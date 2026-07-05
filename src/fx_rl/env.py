"""The FX portfolio-allocation environment (corrected MDP).

This is a full re-design of the notebook's ``FXArbitrageEnv``.  The notebook had
several contradictions between its docstring, action space, and ``step`` logic;
they are resolved here and each fix is documented inline.

MDP definition
--------------
**State** (Dict observation, consumed by the GAT extractor):
    nodes : (4, 2)  per currency [current_weight, previous_weight]
    edges : (12, 5) per directed pair [log_mid_rate, spread_pct, vol, momentum, arb]
The portfolio weights live in the node features, so the previous allocation is
genuinely part of the observation (the notebook tracked ``prev_weights`` but
never exposed them).

**Action** (Box(-1, 1, shape=(4,))):  a *target portfolio allocation* over
[USD, EUR, GBP, JPY].  The raw action is mapped to long-only weights that sum to
one via a temperature-scaled softmax::

    w = softmax(action * temperature)

This guarantees a valid simplex (no leverage, no shorting) for *any* action the
policy emits, which is much cleaner than clipping.  The agent then rebalances the
portfolio toward ``w`` at executable bid/ask prices.

**Reward** (return-based, comparable units):
    reward_t = reward_scale * log(NAV_{t+1} / NAV_t)  -  turnover_penalty * turnover_t

* ``log(NAV_{t+1}/NAV_t)`` is the realised one-step log return.  Because trades
  execute at bid/ask, the spread / transaction cost is *already* reflected in
  NAV — there is no separate dollar-scale cost term to add (that would double
  count).  ``turnover`` (= 0.5 * L1 weight change) carries a small explicit
  penalty as a behavioural regulariser against churn.
* ``reward_scale`` lifts the ~1e-4/tick log return to O(1e-2) so it is on a
  comparable scale to the turnover penalty — fixing the notebook's problem where
  raw dollar ΔNAV (O(10)) dwarfed every other term.

Changes vs. the notebook (summary)
-----------------------------------
1. Action is 4-dim target weights via softmax, not a 3-dim half-delta/half-target
   hybrid.  The notebook's docstring said "delta in [-1,1]" but the code used a
   3-dim ``Box(0,1)`` interpreted as a target weight capped at ``MAX_TRADE_FRAC``
   — three incompatible definitions.  Now there is exactly one.
2. ``prev_weights`` actually enters the observation (as a node feature).
3. Reward is log-return based with comparable-scale costs; the large ``HOLD_BONUS
   = 0.05`` that invited a do-nothing policy is removed.
4. Spread cost is charged once, via bid/ask execution, not double-counted.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym
import pandas as pd
from gymnasium import spaces

from .config import Config, N_NODE_FEATS, N_EDGE_FEATS
from .graph import (
    FxGraphBuilder, get_pair_bid_ask, parse_tick, convert, cycle_factor,
)


def softmax_weights(action: np.ndarray, temperature: float) -> np.ndarray:
    """Map a raw action vector to long-only portfolio weights summing to 1."""
    logits = np.asarray(action, dtype=np.float64) * temperature
    logits -= logits.max()  # numerical stability
    exp = np.exp(logits)
    return (exp / exp.sum()).astype(np.float32)


def action_to_arb_intensity(raw: float) -> float:
    """Map the raw arbitrage action in [-1, 1] to a commit fraction in [0, 1].

    We use ``max(0, raw)`` rather than ``(raw + 1) / 2`` so that the policy's
    *neutral* output (0) means **no arbitrage**. The earlier affine mapping made
    neutral = 0.5 intensity, i.e. the agent committed half its USD to the best
    cycle every tick and bled spread on the ~95% of ticks with no opportunity —
    which trained the agent to disable arbitrage entirely instead of learning to
    fire selectively inside windows. With this mapping, doing nothing is free and
    the agent can discover when firing pays.
    """
    return float(max(0.0, raw))


class FXArbitrageEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: Config,
        df_data: pd.DataFrame,
        builder: Optional[FxGraphBuilder] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.currencies = cfg.data.currencies
        self.non_usd = [c for c in self.currencies if c != "USD"]
        self.builder = builder or FxGraphBuilder(self.currencies)
        self.df_data = df_data.reset_index(drop=True)

        self.window_size = cfg.env.window_size
        self.initial_balance_usd = cfg.env.initial_balance_usd
        self.max_drawdown = cfg.env.max_drawdown
        self.max_t = len(self.df_data) - 1

        n_ccy = cfg.data.n_currencies
        # Action = [n_ccy allocation logits, 1 arbitrage intensity].
        # The first n_ccy map (softmax) to long-only portfolio weights; the last
        # maps to a [0,1] fraction of USD to route through the best executable
        # USD->X->Y->USD triangular-arbitrage cycle this tick.
        self.n_ccy = n_ccy
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_ccy + 1,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "nodes": spaces.Box(-np.inf, np.inf, shape=(n_ccy, N_NODE_FEATS), dtype=np.float32),
            "edges": spaces.Box(-np.inf, np.inf, shape=(self.builder.n_edges, N_EDGE_FEATS), dtype=np.float32),
        })

        # runtime state (set in reset)
        self.t = self.window_size - 1
        self.balances: dict = {}
        self.weights = np.zeros(n_ccy, dtype=np.float32)
        self.prev_weights = np.zeros(n_ccy, dtype=np.float32)
        self.portfolio_history: list = []

    # ------------------------------------------------------------------
    # portfolio accounting (all valuation at executable bid = liquidation value)
    # ------------------------------------------------------------------
    def _value_usd(self, ccy: str, tick: dict) -> float:
        amount = self.balances.get(ccy, 0.0)
        if ccy == "USD":
            return amount
        if amount <= 1e-12:
            return 0.0
        try:
            bid, _ = get_pair_bid_ask(ccy, "USD", tick)
            return amount * bid
        except KeyError:
            return 0.0

    def _nav(self, tick: dict) -> float:
        return sum(self._value_usd(c, tick) for c in self.currencies)

    def _weights(self, tick: dict, nav: float) -> np.ndarray:
        if nav <= 1e-12:
            w = np.zeros(len(self.currencies), dtype=np.float32)
            w[0] = 1.0
            return w
        return np.array([self._value_usd(c, tick) / nav for c in self.currencies], dtype=np.float32)

    def _execute_trade(self, ccy: str, delta_usd: float, tick: dict) -> None:
        """Move ``delta_usd`` of NAV into (>0) or out of (<0) ``ccy`` via CCY/USD.

        Buys pay ``ask`` and sells receive ``bid`` — the worse side both ways, so
        the bid/ask spread is the realised transaction cost.
        """
        try:
            bid, ask = get_pair_bid_ask(ccy, "USD", tick)
        except KeyError:
            return

        if delta_usd > 1e-9:  # buy ccy
            spend = min(delta_usd, self.balances["USD"])
            if spend > 1e-9:
                self.balances["USD"] -= spend
                self.balances[ccy] = self.balances.get(ccy, 0.0) + spend / ask
        elif delta_usd < -1e-9:  # sell ccy
            want_usd = -delta_usd
            amount_ccy = want_usd / bid
            held = self.balances.get(ccy, 0.0)
            if amount_ccy >= held:  # sell everything we have
                self.balances["USD"] += held * bid
                self.balances[ccy] = 0.0
            else:
                self.balances[ccy] = held - amount_ccy
                self.balances["USD"] += want_usd

    def _rebalance(self, target_weights: np.ndarray, tick: dict, nav: float) -> None:
        """Rebalance holdings toward ``target_weights`` at prices in ``tick``.

        Sells are executed before buys so freed USD funds the purchases.
        """
        targets = {c: float(target_weights[i]) * nav for i, c in enumerate(self.currencies)}
        current = {c: self._value_usd(c, tick) for c in self.currencies}
        for ccy in self.non_usd:  # sells first
            delta = targets[ccy] - current[ccy]
            if delta < -1e-9:
                self._execute_trade(ccy, delta, tick)
        for ccy in self.non_usd:  # then buys
            delta = targets[ccy] - current[ccy]
            if delta > 1e-9:
                self._execute_trade(ccy, delta, tick)

    # ------------------------------------------------------------------
    # triangular arbitrage (multi-currency, funded from any held currency)
    # ------------------------------------------------------------------
    def _profitable_cycles(self, tick: dict):
        """All directed triangles with executable factor > 1, sorted best-first.

        Returns a list of ``(cycle, factor)``. Cycles need not involve USD — any
        of the 8 directed triangles among the 4 currencies qualifies.
        """
        out = [(cyc, cycle_factor(cyc, tick)) for cyc in self.builder.cycles]
        out = [(c, f) for c, f in out if f > 1.0 + 1e-12]
        out.sort(key=lambda cf: -cf[1])
        return out

    def current_best_arb(self):
        """Best ``(cycle, factor)`` at the current decision tick (for policies)."""
        cycles = self._profitable_cycles(parse_tick(self.df_data.iloc[self.t]))
        return cycles[0] if cycles else (None, 1.0)

    @staticmethod
    def _rotate(cycle, start):
        i = cycle.index(start)
        return cycle[i:] + cycle[:i]

    def _execute_arbitrage(self, intensity: float, tick: dict):
        """Execute the best *fundable* profitable arbitrage cycle.

        **Multi-currency, gated.** Only profitable cycles (factor > 1) are
        considered (no knowingly-losing trade). Among them we pick the one with the
        greatest expected profit given current holdings — ``(factor-1) * value the
        agent holds in the cycle's currencies`` — then run it from *every* held
        vertex, each committing ``intensity * balance[vertex]``. Each cycle
        round-trips back to its start currency, so arbitrage leaves the portfolio
        weights essentially unchanged; its P&L flows into NAV.

        Funding is therefore decoupled from USD: whatever the agent holds, if it
        sits on a profitable cycle it can be deployed. Returns ``(details, reason)``
        where details is a dict (or None) describing the executed cycle.
        """
        intensity = float(np.clip(intensity, 0.0, 1.0))
        if intensity <= 1e-9:
            return None, "intensity_zero"
        cycles = self._profitable_cycles(tick)
        if not cycles:
            return None, "no_profitable_cycle"

        # choose cycle with max expected profit given what we hold
        best, best_score = None, 0.0
        for cyc, f in cycles:
            held_val = sum(self._value_usd(c, tick) for c in cyc)
            score = held_val * (f - 1.0)
            if score > best_score:
                best, best_score = (cyc, f), score
        if best is None:
            return None, "no_funds_on_cycle"

        cyc, factor = best
        funded = {}
        for start in cyc:
            commit = intensity * self.balances.get(start, 0.0)
            if commit <= 1e-12:
                continue
            a, b, c = self._rotate(cyc, start)
            self.balances[a] -= commit
            self.balances[a] += convert(convert(convert(commit, a, b, tick), b, c, tick), c, a, tick)
            funded[start] = commit
        if not funded:
            return None, "no_funds_on_cycle"
        return {"cycle": cyc, "factor": factor, "funded": funded}, "executed"

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------
    def _build_obs(self, current_weights: np.ndarray, prev_weights: np.ndarray) -> dict:
        window = self.df_data.iloc[self.t - self.window_size + 1 : self.t + 1]
        graph = self.builder.build(window, current_weights, prev_weights)
        return {
            "nodes": graph.x.numpy().astype(np.float32),
            "edges": graph.edge_attr.numpy().astype(np.float32),
        }

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.window_size - 1
        self.balances = {c: 0.0 for c in self.currencies}
        self.balances["USD"] = self.initial_balance_usd

        tick = parse_tick(self.df_data.iloc[self.t])
        nav = self._nav(tick)
        w = self._weights(tick, nav)  # all-USD => [1, 0, 0, 0]
        self.weights = w
        self.prev_weights = w.copy()
        self.portfolio_history = [nav]

        return self._build_obs(current_weights=w, prev_weights=self.prev_weights), {}

    def step(self, action: np.ndarray):
        """Gym step: split the raw action into target weights + arb intensity."""
        action = np.asarray(action, dtype=np.float32)
        target = softmax_weights(action[: self.n_ccy], self.cfg.env.action_temperature)
        arb_intensity = action_to_arb_intensity(action[self.n_ccy])
        return self.act_on_weights(target, arb_intensity)

    def act_on_weights(self, target: np.ndarray, arb_intensity: float = 0.0):
        """Core transition given a target allocation and an arbitrage intensity.

        ``step`` calls this after splitting the agent's action; baseline policies
        call it directly (with their own weights and intensity), so the agent and
        all baselines share exactly the same dynamics and costs.

        Sequence at the current prices: (1) execute the USD-funded arbitrage cycle
        with the committed intensity, then (2) rebalance toward ``target``. The
        arbitrage round-trips back to USD, so it does not disturb the portfolio
        weights (hence no turnover penalty on it); its P&L flows into NAV.
        """
        cfg = self.cfg.env
        target = np.asarray(target, dtype=np.float32)
        tick = parse_tick(self.df_data.iloc[self.t])
        nav_before = self._nav(tick)
        w_before = self._weights(tick, nav_before)

        # 1. arbitrage first (multi-currency, gated to profitable cycles → arb_profit >= 0)
        profitable_cycles = self._profitable_cycles(tick)
        arb_details, arb_reason = self._execute_arbitrage(arb_intensity, tick)
        nav_after_arb = self._nav(tick)
        arb_profit = nav_after_arb - nav_before  # USD P&L of the arbitrage

        # 2. rebalance to target at current prices
        self._rebalance(target, tick, nav_after_arb)
        nav_after_trade = self._nav(tick)
        w_after_trade = self._weights(tick, nav_after_trade)
        turnover = 0.5 * float(np.abs(w_after_trade - w_before).sum())
        # spread cost of the *rebalance* only (arbitrage P&L reported separately)
        transaction_cost = nav_after_arb - nav_after_trade

        # 3. advance one tick; portfolio is re-valued at new prices
        self.t += 1
        tick_next = parse_tick(self.df_data.iloc[self.t])
        nav_next = self._nav(tick_next)
        w_next = self._weights(tick_next, nav_next)

        # 4. return-based reward (arbitrage P&L + spreads already inside the NAV change)
        log_ret = float(np.log(max(nav_next, 1e-9) / max(nav_before, 1e-9)))
        reward = cfg.reward_scale * log_ret - cfg.turnover_penalty * turnover

        self.portfolio_history.append(nav_next)

        # 5. termination
        terminated = False
        peak = max(self.portfolio_history)
        if peak > 0 and (peak - nav_next) / peak > self.max_drawdown:
            terminated = True
            reward -= cfg.drawdown_penalty
        truncated = self.t >= self.max_t

        # 6. observation: current allocation now, previous = allocation when we acted
        obs = self._build_obs(current_weights=w_next, prev_weights=w_before)
        self.prev_weights = w_before
        self.weights = w_next

        info = {
            "nav": nav_next,
            "log_return": log_ret,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "arb_intensity": float(np.clip(arb_intensity, 0.0, 1.0)),
            "arb_profit": arb_profit,
            "arb_reason": arb_reason,
            "arb_executed": arb_details,
            "n_profitable_cycles": len(profitable_cycles),
            "best_arb_factor": profitable_cycles[0][1] if profitable_cycles else 1.0,
            "weights": w_next,
            "target_weights": target,
        }
        return obs, float(reward), terminated, truncated, info
