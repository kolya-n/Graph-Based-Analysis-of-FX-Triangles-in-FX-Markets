"""Financial performance metrics.

Computed from a NAV (net-asset-value) trajectory plus optional per-step turnover
and transaction-cost series produced by the environment.  The notebook reported
only the final NAV and total return; a diploma-level evaluation needs risk-adjusted
and cost metrics, which live here.

All ratios assume a zero risk-free rate.  ``periods_per_year`` annualises the
per-tick statistics — for synthetic tick data there is no real calendar, so we
expose it as a parameter (default 252, i.e. treat one tick ≈ one trading day);
the *relative* ranking of strategies is unaffected by the choice.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def max_drawdown(nav: np.ndarray) -> float:
    """Largest peak-to-trough fractional decline of the NAV curve (>= 0)."""
    nav = np.asarray(nav, dtype=np.float64)
    running_peak = np.maximum.accumulate(nav)
    drawdowns = (running_peak - nav) / running_peak
    return float(np.max(drawdowns)) if len(nav) else 0.0


def compute_metrics(
    nav: Sequence[float],
    turnovers: Optional[Sequence[float]] = None,
    transaction_costs: Optional[Sequence[float]] = None,
    arb_profits: Optional[Sequence[float]] = None,
    periods_per_year: int = 252,
) -> dict:
    """Return a dict of performance metrics for one NAV trajectory.

    Parameters
    ----------
    nav:
        NAV per step including the initial value (length T+1).
    turnovers, transaction_costs, arb_profits:
        Optional per-step series (length T) from ``info``.
    periods_per_year:
        Annualisation factor for Sharpe / Sortino / volatility.

    Notes
    -----
    When ``arb_profits`` is given we report the **alpha decomposition**: total P&L
    split into the arbitrage component (sum of per-step ``arb_profit``) and the
    residual allocation/trading component (``allocation_pnl``). An audit found the
    agent's edge is ~entirely arbitrage while ``allocation_pnl`` is negative, so this
    split is essential to read the results honestly.
    NB: ``sharpe`` here uses ``sqrt(periods_per_year)`` on per-tick returns — it is
    a *convention*, not a real annual Sharpe; ``sortino`` is inflated when downside
    is tiny (near-riskless synthetic arbitrage).
    """
    nav = np.asarray(nav, dtype=np.float64)
    if len(nav) < 2:
        raise ValueError("NAV series must have at least two points.")

    simple_ret = nav[1:] / nav[:-1] - 1.0
    mean_r = float(np.mean(simple_ret))
    std_r = float(np.std(simple_ret, ddof=1)) if len(simple_ret) > 1 else 0.0
    downside = simple_ret[simple_ret < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    ann = np.sqrt(periods_per_year)

    metrics = {
        "final_nav": float(nav[-1]),
        "total_return": float(nav[-1] / nav[0] - 1.0),
        "n_periods": int(len(simple_ret)),
        "mean_return": mean_r,
        "volatility": std_r,
        "annualized_volatility": std_r * ann,
        "sharpe": (mean_r / std_r * ann) if std_r > 0 else 0.0,
        "sortino": (mean_r / downside_std * ann) if downside_std > 0 else 0.0,
        "max_drawdown": max_drawdown(nav),
    }
    if turnovers is not None and len(turnovers):
        metrics["avg_turnover"] = float(np.mean(turnovers))
        metrics["total_turnover"] = float(np.sum(turnovers))
    if transaction_costs is not None and len(transaction_costs):
        metrics["avg_transaction_cost"] = float(np.mean(transaction_costs))
        metrics["total_transaction_cost"] = float(np.sum(transaction_costs))
    if arb_profits is not None and len(arb_profits):
        total_pnl = float(nav[-1] - nav[0])
        total_arb = float(np.sum(arb_profits))
        metrics["total_pnl"] = total_pnl
        metrics["arb_pnl"] = total_arb
        metrics["allocation_pnl"] = total_pnl - total_arb  # residual = directional/trading alpha
    return metrics
