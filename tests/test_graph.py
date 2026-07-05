"""Tests for graph construction, quote inversion, and arbitrage."""

import numpy as np
import pytest

from fx_rl.config import Config, N_EDGE_FEATS, N_NODE_FEATS
from fx_rl.data import generate_gbm_data
from fx_rl.graph import (
    FxGraphBuilder,
    get_pair_bid_ask,
    convert,
    compute_triangular_arbitrage,
    parse_tick,
    enumerate_directed_cycles,
    cycle_factor,
    best_cycle_factor_from,
    EDGE_FEAT_SCALE,
)


@pytest.fixture
def cfg():
    return Config()


@pytest.fixture
def df(cfg):
    return generate_gbm_data(cfg.data, seed=123)


@pytest.fixture
def builder(cfg):
    return FxGraphBuilder(cfg.data.currencies)


# --- quote logic ----------------------------------------------------------
def test_direct_and_inverse_consistency(df):
    tick = parse_tick(df.iloc[0])
    # Direct quote EUR->USD exists in the data.
    bid_eur_usd, ask_eur_usd = get_pair_bid_ask("EUR", "USD", tick)
    assert bid_eur_usd < ask_eur_usd  # spread is positive

    # Inverse USD->EUR must be derived and still have bid < ask.
    bid_usd_eur, ask_usd_eur = get_pair_bid_ask("USD", "EUR", tick)
    assert bid_usd_eur < ask_usd_eur

    # Inversion identity: bid(A,B) == 1/ask(B,A).
    assert bid_usd_eur == pytest.approx(1.0 / ask_eur_usd, rel=1e-9)
    assert ask_usd_eur == pytest.approx(1.0 / bid_eur_usd, rel=1e-9)


def test_round_trip_loses_spread(df):
    """Converting A->B->A must lose money (two half-spreads)."""
    tick = parse_tick(df.iloc[0])
    start = 1.0
    mid = convert(start, "EUR", "USD", tick)
    back = convert(mid, "USD", "EUR", tick)
    assert back < start


def test_missing_quote_raises(df):
    tick = parse_tick(df.iloc[0])
    with pytest.raises(KeyError):
        get_pair_bid_ask("USD", "CHF", tick)  # CHF not in the universe


# --- arbitrage ------------------------------------------------------------
def _mid_cycle_arb(edge_pairs, tick, currencies):
    """Naive arbitrage using MID prices (ignores spreads) — for comparison."""
    def mid(src, dst):
        b, a = get_pair_bid_ask(src, dst, tick)
        return (b + a) / 2.0

    arb = {p: 0.0 for p in edge_pairs}
    for a in currencies:
        for b in currencies:
            if a == b:
                continue
            for c in currencies:
                if c in (a, b):
                    continue
                cyc = mid(a, b) * mid(b, c) * mid(c, a) - 1.0
                for edge in [(a, b), (b, c), (c, a)]:
                    arb[edge] = max(arb[edge], cyc)
    return arb


def test_arbitrage_signal_keys(builder, df):
    tick = parse_tick(df.iloc[0])
    arb = compute_triangular_arbitrage(builder.edge_pairs, tick, builder.currencies)
    assert set(arb.keys()) == set(builder.edge_pairs)


def test_arbitrage_is_spread_aware(builder, df):
    """The executable (bid-path) arbitrage must be <= the naive mid-price
    arbitrage: paying the spread on every leg erodes the edge.

    NB: with data v2 the cross rates are no-arbitrage-consistent except inside
    injected windows, so we scan for a tick that actually has a (mid-price) edge."""
    arb_tick = None
    for i in range(len(df)):
        tick = parse_tick(df.iloc[i])
        mid_based = _mid_cycle_arb(builder.edge_pairs, tick, builder.currencies)
        if max(mid_based.values()) > 1e-4:  # an injected window
            arb_tick = (tick, mid_based)
            break
    assert arb_tick is not None, "no arbitrage window found in the path"

    tick, mid_based = arb_tick
    executable = compute_triangular_arbitrage(builder.edge_pairs, tick, builder.currencies)
    for edge in builder.edge_pairs:
        assert executable[edge] <= mid_based[edge] + 1e-12
    assert max(executable.values()) < max(mid_based.values())


# --- cycle helpers & normalization ---------------------------------------
def test_enumerate_directed_cycles(cfg):
    cycles = enumerate_directed_cycles(cfg.data.currencies)
    assert len(cycles) == 8  # 4 currencies -> C(4,3)*2 directed triangles
    assert all(len(c) == 3 and len(set(c)) == 3 for c in cycles)
    assert len(set(cycles)) == 8  # deduped by rotation


def test_cycle_factor_and_best_from(builder, df):
    tick = parse_tick(df.iloc[0])
    # a no-window tick: every cycle factor ~ 1 (consistent crosses, minus spread)
    for cyc in builder.cycles:
        assert cycle_factor(cyc, tick) <= 1.0 + 1e-6
    assert best_cycle_factor_from("USD", builder.cycles, tick) >= 1.0


def test_node_arb_feature_active_in_window(cfg):
    """Inside an injected window the per-currency arb node feature is > 0."""
    cfg.data.n_ticks = 600
    cfg.data.arb_injection_prob = 0.1
    df = generate_gbm_data(cfg.data, seed=5)
    builder = FxGraphBuilder(cfg.data.currencies)
    w = np.full(4, 0.25, dtype=np.float32)
    seen_positive = False
    for i in range(cfg.env.window_size, len(df)):
        g = builder.build(df.iloc[i - cfg.env.window_size : i], w, w)
        if g.x[:, 2].max() > 0:  # node arb feature column
            seen_positive = True
            break
    assert seen_positive


def test_edge_features_normalized(builder, df):
    """Normalization shrinks the scale gap between log_mid_rate and arb_signal
    from ~10^4x toward ~10x."""
    feats = np.stack([
        builder.build(df.iloc[i - 20 : i], np.full(4, 0.25, np.float32), np.full(4, 0.25, np.float32)).edge_attr.numpy()
        for i in range(20, 400)
    ]).reshape(-1, N_EDGE_FEATS)
    stds = feats.std(axis=0)
    gap = stds[0] / max(stds[3], 1e-9)  # log_mid_rate vs momentum std
    assert gap < 50  # was ~2500 unnormalized


# --- graph shapes ---------------------------------------------------------
def test_topology(builder, cfg):
    assert builder.n_nodes == cfg.data.n_currencies == 4
    assert builder.n_edges == cfg.data.n_edges == 12
    assert builder.edge_index.shape == (2, 12)


def test_build_shapes(builder, cfg, df):
    window = df.iloc[: cfg.env.window_size]
    cw = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    pw = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    g = builder.build(window, current_weights=cw, prev_weights=pw)

    assert g.x.shape == (4, N_NODE_FEATS)
    assert g.edge_index.shape == (2, 12)
    assert g.edge_attr.shape == (12, N_EDGE_FEATS)
    # node features carry the supplied weights
    np.testing.assert_allclose(g.x[:, 0].numpy(), cw)
    np.testing.assert_allclose(g.x[:, 1].numpy(), pw)
    assert np.isfinite(g.edge_attr.numpy()).all()
