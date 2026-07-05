"""Graph construction: currencies -> nodes, FX rates -> directed edges.

The core thesis representation.  Every tick is turned into a PyTorch-Geometric
``Data`` graph with

* ``N`` currency nodes (default 4: USD, EUR, GBP, JPY);
* ``N*(N-1)`` directed edges (default 12) — one per ordered currency pair;
* node features  ``[current_weight, previous_weight, best_arb_rate_from_ccy]``;
* edge features  ``[log_mid_rate, spread_pct, volatility, momentum, arb_signal]``.

Feature normalization
----------------------
The raw features live on wildly different scales — ``log_mid_rate`` has std ~3.6
while ``arb_signal`` / ``momentum`` / ``spread_pct`` have std ~1e-4 to 1e-3, a
~10^4x gap. Un-normalized, the GAT's linear/attention layers are dominated by the
rate level and the decision-relevant signals are numerically invisible (this was a
confirmed reason the agent ignored arbitrage). We therefore multiply each feature
by a fixed scale (``EDGE_FEAT_SCALE`` / ``ARB_NODE_SCALE``) chosen so every channel
has ~unit magnitude. Fixed constants (not running stats) keep it deterministic and
leak-free across train/test.

The per-currency node feature ``best_arb_rate_from_ccy`` hands the agent the best
executable arbitrage rate it could earn by funding a cycle *from that currency* —
delivering the arbitrage opportunity directly instead of forcing the GAT to
reconstruct it from tiny per-edge numbers.

Quote convention
----------------
``get_pair_bid_ask(src, dst)`` returns the *executable* (bid, ask) for converting
``src`` into ``dst``, expressed as **dst units per 1 src unit**.  Selling ``src``
to obtain ``dst`` executes at ``bid``; the reverse direction executes at ``ask``.
When only the inverse pair is quoted we invert: ``bid(A,B) = 1/ask(B,A)`` and
``ask(A,B) = 1/bid(B,A)`` — this preserves ``bid < ask`` and keeps every
conversion spread-aware.

Fix relative to the notebook
----------------------------
The notebook's ``compute_triangular_arbitrage`` multiplied ``get_pair_bid_ask()[0]``
(the bid) for the three legs.  For this quote convention that *happens* to equal
the executable round-trip, but it was implicit (the discarded ``_`` made it look
like a mid/bid ratio) and would silently break if the convention changed.  Here
arbitrage is computed by explicitly *converting* a unit of currency around the
cycle with ``convert()``, which always pays the correct side of the spread.  The
result is the same number, now correct-by-construction and clearly spread-aware.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .config import N_EDGE_FEATS


# Fixed per-feature scales (≈ 1/std measured on synthetic data) so each channel is
# ~unit magnitude for the GAT. Order matches edge features:
# [log_mid_rate, spread_pct, volatility, momentum, arb_signal].
EDGE_FEAT_SCALE = np.array([0.28, 1.0e5, 1.1e4, 670.0, 5000.0], dtype=np.float32)
# Scale for the per-currency arbitrage node feature (same units as edge arb_signal).
ARB_NODE_SCALE = 5000.0


# --------------------------------------------------------------------------
# Quote helpers
# --------------------------------------------------------------------------
def get_pair_bid_ask(src: str, dst: str, tick_data: dict) -> Tuple[float, float]:
    """Executable (bid, ask) for converting ``src`` -> ``dst`` (dst per src).

    Uses the directly quoted pair if available, otherwise inverts the reverse
    quote: ``bid(A,B) = 1/ask(B,A)``, ``ask(A,B) = 1/bid(B,A)``.
    """
    direct = f"{src}{dst}"
    inverse = f"{dst}{src}"
    if direct in tick_data:
        return tick_data[direct]["bid"], tick_data[direct]["ask"]
    if inverse in tick_data:
        return 1.0 / tick_data[inverse]["ask"], 1.0 / tick_data[inverse]["bid"]
    raise KeyError(f"Quote {src}->{dst} not found")


def convert(amount: float, src: str, dst: str, tick_data: dict) -> float:
    """Convert ``amount`` of ``src`` into ``dst`` at the executable rate.

    Selling ``src`` for ``dst`` always executes at ``bid(src, dst)`` — the worse
    side for the trader — so the spread cost is included.
    """
    bid, _ = get_pair_bid_ask(src, dst, tick_data)
    return amount * bid


def enumerate_directed_cycles(currencies: List[str]) -> List[Tuple[str, str, str]]:
    """All distinct directed 3-cycles ``a->b->c->a`` (deduped by rotation).

    Direction matters (``a->b->c`` differs from ``a->c->b``); rotations of the same
    directed cycle are collapsed to one canonical tuple (starting at the currency
    earliest in ``currencies``). For 4 currencies this yields 8 cycles.
    """
    order = {c: i for i, c in enumerate(currencies)}
    seen, cycles = set(), []
    for a, b, c in itertools.permutations(currencies, 3):
        canon = min([(a, b, c), (b, c, a), (c, a, b)], key=lambda t: order[t[0]])
        if canon not in seen:
            seen.add(canon)
            cycles.append(canon)
    return cycles


def cycle_factor(cycle: Tuple[str, str, str], tick_data: dict) -> float:
    """Executable factor (out per 1 unit in) of going once around ``cycle``.

    ``> 1`` means a profitable arbitrage survives spreads. Rotation-invariant, so
    the funding vertex does not change the factor. Returns 0.0 if a quote is missing.
    """
    a, b, c = cycle
    try:
        amt = convert(convert(convert(1.0, a, b, tick_data), b, c, tick_data), c, a, tick_data)
    except KeyError:
        return 0.0
    return amt


def best_cycle_factor_from(ccy: str, cycles: List[Tuple[str, str, str]], tick_data: dict) -> float:
    """Best executable factor among cycles that include ``ccy`` (>= 1.0)."""
    best = 1.0
    for cyc in cycles:
        if ccy in cyc:
            best = max(best, cycle_factor(cyc, tick_data))
    return best


def compute_triangular_arbitrage(
    edge_pairs: List[Tuple[str, str]],
    tick_data: dict,
    currencies: List[str],
) -> Dict[Tuple[str, str], float]:
    """Spread-aware triangular-arbitrage signal per directed edge.

    For every cycle ``a -> b -> c -> a`` we convert one unit of ``a`` around the
    loop paying the spread on each leg; ``cycle_return = final - 1``.  A positive
    value means an *executable* arbitrage profit survives transaction costs.  Each
    edge stores the best (max) cycle return among cycles that traverse it.
    """
    arb: Dict[Tuple[str, str], float] = {pair: 0.0 for pair in edge_pairs}
    for a in currencies:
        for b in currencies:
            if a == b:
                continue
            for c in currencies:
                if c == a or c == b:
                    continue
                try:
                    after_b = convert(1.0, a, b, tick_data)
                    after_c = convert(after_b, b, c, tick_data)
                    back_to_a = convert(after_c, c, a, tick_data)
                    cycle_return = back_to_a - 1.0
                except KeyError:
                    continue
                for edge in [(a, b), (b, c), (c, a)]:
                    if edge in arb:
                        arb[edge] = max(arb[edge], cycle_return)
    return arb


def parse_tick(row: pd.Series) -> dict:
    """Extract ``{pair: {'bid': .., 'ask': ..}}`` from a single DataFrame row."""
    tick_data: dict = {}
    pairs = {col.split("_")[0] for col in row.index if col.endswith("_bid") or col.endswith("_ask")}
    for pair in pairs:
        if f"{pair}_bid" in row.index and f"{pair}_ask" in row.index:
            tick_data[pair] = {
                "bid": float(row[f"{pair}_bid"]),
                "ask": float(row[f"{pair}_ask"]),
            }
    return tick_data


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------
class FxGraphBuilder:
    """Builds a PyG ``Data`` graph for one tick from market window + portfolio.

    The graph *topology* and edge ordering are derived once from ``currencies``,
    so they always match the rest of the pipeline.  Node features are supplied by
    the environment (it owns the portfolio state); the builder only computes
    market-derived edge features.
    """

    def __init__(self, currencies: List[str]):
        self.currencies = list(currencies)
        self.n_nodes = len(self.currencies)

        src_list, dst_list, self.edge_pairs = [], [], []
        for i, src in enumerate(self.currencies):
            for j, dst in enumerate(self.currencies):
                if i != j:
                    src_list.append(i)
                    dst_list.append(j)
                    self.edge_pairs.append((src, dst))

        self.edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        self.n_edges = len(self.edge_pairs)
        self.cycles = enumerate_directed_cycles(self.currencies)  # for arb signals

    def _edge_features(self, df_window: pd.DataFrame, tick_data: dict) -> np.ndarray:
        arb_signals = compute_triangular_arbitrage(self.edge_pairs, tick_data, self.currencies)
        feats = []
        for src, dst in self.edge_pairs:
            try:
                bid, ask = get_pair_bid_ask(src, dst, tick_data)
                mid = (bid + ask) / 2.0
                spread_pct = (ask - bid) / mid if mid > 0 else 0.0

                # Historical mid series for this directed rate (invert if needed).
                fwd, inv = f"{src}{dst}", f"{dst}{src}"
                if f"{fwd}_bid" in df_window.columns:
                    hist = (df_window[f"{fwd}_bid"] + df_window[f"{fwd}_ask"]) / 2.0
                elif f"{inv}_bid" in df_window.columns:
                    hist = 1.0 / ((df_window[f"{inv}_bid"] + df_window[f"{inv}_ask"]) / 2.0)
                else:
                    hist = pd.Series([mid])

                log_rets = np.log(hist / hist.shift(1)).dropna()
                volatility = float(log_rets.std()) if len(log_rets) > 2 else 0.0
                momentum = float((mid / hist.iloc[0]) - 1.0) if len(hist) > 1 else 0.0

                feats.append([
                    float(np.log(max(mid, 1e-6))),  # log_mid_rate
                    float(spread_pct),              # spread_pct
                    volatility,                     # volatility
                    momentum,                       # momentum
                    float(arb_signals[(src, dst)]), # arb_signal (spread-aware)
                ])
            except KeyError:
                feats.append([0.0] * N_EDGE_FEATS)
        # Normalize to comparable scale so the GAT can see all channels.
        return np.asarray(feats, dtype=np.float32) * EDGE_FEAT_SCALE

    def build(
        self,
        df_window: pd.DataFrame,
        current_weights: np.ndarray,
        prev_weights: np.ndarray,
    ) -> Data:
        """Assemble the tick graph.

        Parameters
        ----------
        df_window:
            The trailing window of ticks; the last row is the current quote.
        current_weights, prev_weights:
            Portfolio weights aligned with ``self.currencies`` (sum to 1).  These
            become node features so the agent can reason about turnover from its
            previous allocation — info the notebook tracked but never exposed.

        Node features per currency: ``[current_weight, prev_weight,
        best_arb_rate_from_ccy]`` where the third is the best executable cycle
        rate (factor-1, >=0) the agent could earn by funding an arbitrage cycle
        from that currency — scaled so it is visible to the GAT.
        """
        tick_data = parse_tick(df_window.iloc[-1])
        edge_attr = self._edge_features(df_window, tick_data)

        arb_from = np.array([
            max(0.0, best_cycle_factor_from(c, self.cycles, tick_data) - 1.0)
            for c in self.currencies
        ], dtype=np.float32) * ARB_NODE_SCALE

        node_features = np.stack([
            np.asarray(current_weights, dtype=np.float32),
            np.asarray(prev_weights, dtype=np.float32),
            arb_from,
        ], axis=1)  # shape (n_nodes, 3)

        return Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=self.edge_index,
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        )
