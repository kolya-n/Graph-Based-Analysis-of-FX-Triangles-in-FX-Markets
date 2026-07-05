"""Evaluation: trained TD3 agent vs. baseline policies.

Rolls every policy through the *same* test environment, computes the financial
metrics from ``metrics.py`` for each, writes a metrics CSV, and saves comparison
plots (NAV trajectories + a risk/return bar chart).

This replaces the notebook's evaluation, which ran three *identical* deterministic
rollouts on the same path and reported only NAV / return with no benchmark.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless: save figures, never block on a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config
from .env import FXArbitrageEnv
from .baselines import default_baselines, TD3Policy, run_policy
from .metrics import compute_metrics


def evaluate(
    model,
    cfg: Config,
    test_df: pd.DataFrame,
    periods_per_year: int = 252,
    tag: str = "eval",
) -> pd.DataFrame:
    """Evaluate ``model`` against baselines on ``test_df``.

    Returns a metrics DataFrame (one row per policy) and writes CSV + plots under
    the configured ``results`` directories.
    """
    env = FXArbitrageEnv(cfg, test_df)

    policies = default_baselines(random_seed=cfg.train.eval_seed)
    if model is not None:
        policies.append(TD3Policy(model))

    rows = []
    trajectories = {}
    for policy in policies:
        res = run_policy(env, policy, seed=cfg.train.eval_seed)
        m = compute_metrics(
            res["nav"],
            turnovers=res["turnover"],
            transaction_costs=res["transaction_cost"],
            arb_profits=res.get("arb_profit"),
            periods_per_year=periods_per_year,
        )
        m["policy"] = policy.name
        rows.append(m)
        trajectories[policy.name] = res["nav"]

    metrics_df = pd.DataFrame(rows).set_index("policy")
    # order columns sensibly — alpha decomposition up front (arb vs allocation)
    col_order = [
        "final_nav", "total_return", "arb_pnl", "allocation_pnl", "sharpe", "sortino",
        "max_drawdown", "annualized_volatility", "avg_turnover",
        "avg_transaction_cost", "total_transaction_cost", "n_periods",
    ]
    metrics_df = metrics_df[[c for c in col_order if c in metrics_df.columns]]

    _save_outputs(cfg, metrics_df, trajectories, tag)
    return metrics_df


def _save_outputs(cfg: Config, metrics_df: pd.DataFrame, trajectories: dict, tag: str) -> None:
    os.makedirs(cfg.train.metrics_dir, exist_ok=True)
    os.makedirs(cfg.train.plot_dir, exist_ok=True)

    csv_path = os.path.join(cfg.train.metrics_dir, f"{tag}_metrics.csv")
    metrics_df.to_csv(csv_path)
    print(f"Wrote metrics to {csv_path}")
    print(metrics_df.round(4).to_string())

    # NAV trajectories (panels stacked vertically so each is full-width and legible)
    fig, axes = plt.subplots(2, 1, figsize=(9, 10))
    for name, nav in trajectories.items():
        axes[0].plot(nav, linewidth=1.3, label=name)
    axes[0].axhline(cfg.env.initial_balance_usd, color="grey", ls="--", alpha=0.5)
    axes[0].set_title("NAV trajectory (test)")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("NAV (USD)")
    axes[0].legend(fontsize=8)

    # Sharpe comparison bar
    sharpe = metrics_df["sharpe"].sort_values()
    axes[1].barh(sharpe.index, sharpe.values, color="steelblue")
    axes[1].axvline(0, color="red", ls="--", alpha=0.5)
    axes[1].set_title("Annualised Sharpe by policy")
    axes[1].set_xlabel("Sharpe")

    fig.tight_layout()
    plot_path = os.path.join(cfg.train.plot_dir, f"{tag}_comparison.png")
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Wrote plot to {plot_path}")
