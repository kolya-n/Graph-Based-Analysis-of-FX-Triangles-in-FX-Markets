"""Tests for financial metrics and baseline policies."""

import numpy as np
import pytest

from fx_rl.config import Config
from fx_rl.data import generate_gbm_data
from fx_rl.env import FXArbitrageEnv
from fx_rl.metrics import compute_metrics, max_drawdown
from fx_rl.baselines import (
    CashPolicy,
    EqualWeightPolicy,
    BuyAndHoldPolicy,
    RandomPolicy,
    MomentumPolicy,
    default_baselines,
    run_policy,
)


# --- metrics --------------------------------------------------------------
def test_max_drawdown_simple():
    nav = np.array([100, 120, 90, 110, 80])  # peak 120 -> trough 80 => 1/3
    assert max_drawdown(nav) == pytest.approx((120 - 80) / 120)


def test_max_drawdown_monotonic_is_zero():
    assert max_drawdown(np.array([100, 101, 102, 103])) == 0.0


def test_total_return_and_final_nav():
    nav = [100, 110, 121]  # +10% then +10%
    m = compute_metrics(nav, periods_per_year=252)
    assert m["final_nav"] == pytest.approx(121)
    assert m["total_return"] == pytest.approx(0.21)
    assert m["n_periods"] == 2


def test_sharpe_positive_for_steady_gains():
    nav = [100 * (1.001 ** i) for i in range(50)]  # steady positive returns
    m = compute_metrics(nav)
    assert m["sharpe"] > 0
    assert m["max_drawdown"] == 0.0


def test_metrics_with_cost_series():
    m = compute_metrics([100, 101, 102], turnovers=[0.2, 0.1], transaction_costs=[0.5, 0.3])
    assert m["avg_turnover"] == pytest.approx(0.15)
    assert m["total_transaction_cost"] == pytest.approx(0.8)


def test_alpha_decomposition():
    """Total P&L splits into arbitrage + allocation residual."""
    nav = [100, 110, 121]          # total P&L = +21
    m = compute_metrics(nav, arb_profits=[5.0, 8.0])  # arb = 13
    assert m["total_pnl"] == pytest.approx(21)
    assert m["arb_pnl"] == pytest.approx(13)
    assert m["allocation_pnl"] == pytest.approx(8)  # 21 - 13


def test_alpha_decomposition_absent_without_arb():
    m = compute_metrics([100, 110, 121])
    assert "allocation_pnl" not in m


def test_too_short_raises():
    with pytest.raises(ValueError):
        compute_metrics([100])


# --- baselines ------------------------------------------------------------
@pytest.fixture
def env():
    cfg = Config()
    return FXArbitrageEnv(cfg, generate_gbm_data(cfg.data, seed=7))


def test_all_baselines_produce_valid_simplex(env):
    obs, _ = env.reset()
    for policy in default_baselines():
        policy.reset()
        w = policy.predict_weights(obs, env)
        assert w.shape == (4,)
        assert np.all(w >= -1e-6)
        assert w.sum() == pytest.approx(1.0, abs=1e-5)


def test_cash_policy_stays_flat_ish(env):
    """Cash policy never leaves USD => zero turnover after the first step."""
    res = run_policy(env, CashPolicy())
    assert max(res["turnover"]) < 1e-6
    assert sum(res["transaction_cost"]) == pytest.approx(0.0, abs=1e-6)


def test_buy_and_hold_trades_once(env):
    res = run_policy(env, BuyAndHoldPolicy())
    # only the first step should incur meaningful turnover
    assert res["turnover"][0] > 0.1
    assert max(res["turnover"][1:]) < 1e-3


def test_equal_weight_runs_full_episode(env):
    res = run_policy(env, EqualWeightPolicy())
    m = compute_metrics(res["nav"], res["turnover"], res["transaction_cost"])
    assert m["n_periods"] == env.max_t - (env.window_size - 1)
    assert np.isfinite(m["sharpe"])


def test_random_policy_is_reproducible(env):
    r1 = run_policy(env, RandomPolicy(seed=123))
    r2 = run_policy(env, RandomPolicy(seed=123))
    np.testing.assert_allclose(r1["nav"], r2["nav"])
