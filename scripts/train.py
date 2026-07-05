"""CLI: train the TD3 + GAT agent.

Examples
--------
    python scripts/train.py                                  # defaults
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --timesteps 50000 --seed 1
    python scripts/train.py --csv data/eurusd_ticks.csv      # real data
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fx_rl.config import Config
from fx_rl.data import generate_gbm_data, load_csv_data
from fx_rl.train import train


def main():
    p = argparse.ArgumentParser(description="Train TD3+GAT FX agent")
    p.add_argument("--config", default=None, help="Path to YAML config (optional)")
    p.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
    p.add_argument("--seed", type=int, default=None, help="Override training seed")
    p.add_argument("--csv", default=None, help="Train on a real CSV instead of synthetic GBM")
    p.add_argument("--model-name", default=None, help="Override saved model name")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    if args.timesteps is not None:
        cfg.train.total_timesteps = args.timesteps
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.model_name is not None:
        cfg.train.model_name = args.model_name

    if args.csv:
        train_df = load_csv_data(args.csv, cfg.data)
        print(f"Loaded {len(train_df)} ticks from {args.csv}")
    else:
        train_df = generate_gbm_data(cfg.data, seed=cfg.data.seed)
        print(f"Generated {len(train_df)} synthetic ticks (seed={cfg.data.seed})")

    train(cfg, train_df=train_df, save=True)


if __name__ == "__main__":
    main()
