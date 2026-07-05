"""CLI: per-step monitoring of arbitrage opportunities and agent behaviour.

Rolls a policy through the test path and logs, for every step:
  - the agent's allocation (weights per currency),
  - the number of profitable arbitrage cycles available and the best factor,
  - whether arbitrage was executed, which cycle, and WHY (the reason),
  - arbitrage P&L and NAV.

Writes a per-step CSV and prints a summary. Use it to answer "is the agent seeing
and exploiting the arbitrage, and if not, why".

Examples
--------
    python scripts/diagnose.py --policy arbitrage           # the oracle's behaviour
    python scripts/diagnose.py --model results/models/td3_fx_graph --policy td3
    python scripts/diagnose.py --policy momentum --steps 500
"""

import argparse
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from fx_rl.config import Config
from fx_rl.data import make_train_test, load_csv_data
from fx_rl.env import FXArbitrageEnv
from fx_rl.graph import parse_tick
from fx_rl.baselines import (
    CashPolicy, EqualWeightPolicy, BuyAndHoldPolicy, RandomPolicy,
    MomentumPolicy, ArbitragePolicy, TD3Policy,
)

BASELINES = {
    "cash": CashPolicy, "equal_weight": EqualWeightPolicy,
    "buy_and_hold": BuyAndHoldPolicy, "random": RandomPolicy,
    "momentum": MomentumPolicy, "arbitrage": ArbitragePolicy,
}


def _fmt_cycle(executed):
    if not executed:
        return ""
    cyc = ">".join(executed["cycle"])
    funded = "+".join(sorted(executed["funded"]))
    return f"{cyc} (fund:{funded})"


def main():
    p = argparse.ArgumentParser(description="Per-step arbitrage / allocation monitor")
    p.add_argument("--policy", default="arbitrage",
                   choices=list(BASELINES) + ["td3"], help="Policy to roll")
    p.add_argument("--model", default=None, help="Model path (required for --policy td3)")
    p.add_argument("--config", default=None)
    p.add_argument("--csv", default=None)
    p.add_argument("--regime-shift", action="store_true")
    p.add_argument("--steps", type=int, default=None, help="Limit number of steps logged")
    p.add_argument("--tag", default=None, help="CSV filename tag (default: policy name)")
    args = p.parse_args()

    if args.config:
        cfg = Config.from_yaml(args.config)
    elif args.model and os.path.exists(args.model + "_config.yaml"):
        cfg = Config.from_yaml(args.model + "_config.yaml")
    else:
        cfg = Config()

    if args.csv:
        test_df = load_csv_data(args.csv, cfg.data)
    else:
        _, test_df = make_train_test(cfg.data, train_seed=cfg.data.seed,
                                     test_seed=cfg.train.eval_seed,
                                     regime_shift=args.regime_shift)
    env = FXArbitrageEnv(cfg, test_df)

    if args.policy == "td3":
        if not args.model:
            p.error("--policy td3 requires --model")
        from stable_baselines3 import TD3
        policy = TD3Policy(TD3.load(args.model))
    else:
        policy = BASELINES[args.policy]()

    policy.reset()
    obs, _ = env.reset(seed=cfg.train.eval_seed)
    ccys = cfg.data.currencies
    rows = []
    # per-triangle aggregation + alpha decomposition
    tri = lambda c: "/".join(sorted(set(c)))
    avail, best, execed, tri_profit = Counter(), Counter(), Counter(), defaultdict(float)
    init_nav = env.portfolio_history[-1]
    step = 0
    done = False
    while not done:
        profs = env._profitable_cycles(parse_tick(env.df_data.iloc[env.t]))  # before acting
        target = policy.predict_weights(obs, env)
        arb = policy.predict_arb_intensity(obs, env)
        obs, reward, terminated, truncated, info = env.act_on_weights(target, arb)
        w = info["weights"]
        for cyc, _f in profs:
            avail[tri(cyc)] += 1
        if profs:
            best[tri(profs[0][0])] += 1
        if info["arb_executed"]:
            t = tri(info["arb_executed"]["cycle"])
            execed[t] += 1
            tri_profit[t] += float(info["arb_profit"])
        rows.append({
            "step": step,
            "nav": round(float(info["nav"]), 2),
            **{f"w_{c}": round(float(w[i]), 3) for i, c in enumerate(ccys)},
            "n_profitable_cycles": info["n_profitable_cycles"],
            "best_arb_factor": round(float(info["best_arb_factor"]), 6),
            "arb_intensity": round(float(info["arb_intensity"]), 3),
            "arb_reason": info["arb_reason"],
            "arb_executed": _fmt_cycle(info["arb_executed"]),
            "arb_profit": round(float(info["arb_profit"]), 4),
            "turnover": round(float(info["turnover"]), 4),
        })
        step += 1
        done = terminated or truncated or (args.steps is not None and step >= args.steps)

    df = pd.DataFrame(rows)
    os.makedirs(cfg.train.metrics_dir, exist_ok=True)
    tag = args.tag or args.policy
    out = os.path.join(cfg.train.metrics_dir, f"{tag}_diagnostic.csv")
    df.to_csv(out, index=False)

    # --- summary ---
    n = len(df)
    windows = df["n_profitable_cycles"] > 0
    executed = df["arb_reason"] == "executed"
    print(f"Policy: {args.policy}   steps: {n}   (CSV: {out})")
    print(f"  ticks with a profitable cycle available : {int(windows.sum())} ({windows.mean()*100:.1f}%)")
    print(f"  ticks where arbitrage was executed      : {int(executed.sum())} ({executed.mean()*100:.1f}%)")
    print(f"  total arbitrage profit (USD)            : {df['arb_profit'].sum():.2f}")
    print(f"  mean arb_intensity in windows           : {df.loc[windows, 'arb_intensity'].mean():.3f}")
    print(f"  reasons breakdown                       : {df['arb_reason'].value_counts().to_dict()}")
    print(f"  mean allocation [{', '.join(ccys)}]     : "
          f"{[round(float(df[f'w_{c}'].mean()), 3) for c in ccys]}")

    final_nav = float(df["nav"].iloc[-1])
    total_pnl = final_nav - init_nav
    arb_pnl = float(df["arb_profit"].sum())
    print(f"  final NAV                               : {final_nav:.2f}")
    print("  --- alpha decomposition ---")
    print(f"  total P&L     : {total_pnl:+.2f}")
    print(f"  arbitrage P&L : {arb_pnl:+.2f}")
    print(f"  allocation P&L: {total_pnl - arb_pnl:+.2f}  (directional/trading residual)")
    print("  --- per-triangle capture ---")
    print(f"  {'triangle':14s} {'avail':>6s} {'is_best':>7s} {'executed':>8s} {'profit$':>9s}")
    for t in sorted(avail, key=lambda k: -avail[k]):
        print(f"  {t:14s} {avail[t]:>6d} {best.get(t,0):>7d} {execed.get(t,0):>8d} {tri_profit.get(t,0):>9.1f}")


if __name__ == "__main__":
    main()
