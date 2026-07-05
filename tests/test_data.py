"""Tests for the data generator: tight spreads, regimes, injected arbitrage."""

import numpy as np
import pytest

from fx_rl.config import Config
from fx_rl.data import generate_gbm_data, make_train_test
from fx_rl.graph import parse_tick, compute_triangular_arbitrage, FxGraphBuilder


@pytest.fixture
def cfg():
    return Config()


def test_output_contract(cfg):
    df = generate_gbm_data(cfg.data, seed=1)
    assert df.shape == (cfg.data.n_ticks, 12)
    for pair in cfg.data.base:
        assert f"{pair}_bid" in df.columns and f"{pair}_ask" in df.columns
        assert (df[f"{pair}_ask"] > df[f"{pair}_bid"]).all()  # positive spread


def test_spreads_are_tight(cfg):
    """EURUSD spread should now be sub-bps (was ~1.4 bps before)."""
    df = generate_gbm_data(cfg.data, seed=1)
    mid = (df["EURUSD_bid"] + df["EURUSD_ask"]) / 2
    spread_pct = (df["EURUSD_ask"] - df["EURUSD_bid"]) / mid
    assert spread_pct.mean() < 1e-4  # < 1 bp


def test_regimes_create_persistent_trends(cfg):
    """With regimes on, realized cumulative moves are much larger than with a
    flat tiny drift (trends accumulate)."""
    cfg.data.n_ticks = 2000
    with_regimes = generate_gbm_data(cfg.data, seed=3)
    cfg.data.use_regimes = False
    flat = generate_gbm_data(cfg.data, seed=3)

    def max_abs_log_move(df, pair):
        mid = (df[f"{pair}_bid"] + df[f"{pair}_ask"]) / 2
        return float(np.abs(np.log(mid / mid.iloc[0])).max())

    # regime version should reach a larger excursion in at least one major
    assert max_abs_log_move(with_regimes, "EURUSD") > max_abs_log_move(flat, "EURUSD")


def test_injected_arbitrage_is_executable(cfg):
    """At least one tick must contain an executable triangular arbitrage that
    survives the (tightened) spreads."""
    cfg.data.n_ticks = 600
    cfg.data.arb_injection_prob = 0.05  # plenty of windows for the test
    df = generate_gbm_data(cfg.data, seed=5)
    builder = FxGraphBuilder(cfg.data.currencies)

    best = 0.0
    for i in range(len(df)):
        tick = parse_tick(df.iloc[i])
        arb = compute_triangular_arbitrage(builder.edge_pairs, tick, builder.currencies)
        best = max(best, max(arb.values()))
        if best > 1e-4:  # > 1 bp executable profit
            break
    assert best > 1e-4


def test_no_arbitrage_injection_when_disabled(cfg):
    """With injection off, executable arbitrage stays negligible."""
    cfg.data.n_ticks = 400
    cfg.data.arb_injection_prob = 0.0
    df = generate_gbm_data(cfg.data, seed=5)
    builder = FxGraphBuilder(cfg.data.currencies)
    best = 0.0
    for i in range(len(df)):
        tick = parse_tick(df.iloc[i])
        arb = compute_triangular_arbitrage(builder.edge_pairs, tick, builder.currencies)
        best = max(best, max(arb.values()))
    assert best < 1e-3  # no large injected windows


def test_train_test_independent(cfg):
    cfg.data.n_ticks = 300
    tr, te = make_train_test(cfg.data, train_seed=42, test_seed=1)
    assert not np.allclose(tr["EURUSD_bid"].values, te["EURUSD_bid"].values)
