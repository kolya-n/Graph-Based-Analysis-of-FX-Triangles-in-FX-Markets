"""CLI: evaluate a trained agent against baselines.

Examples
--------
    python scripts/evaluate.py --model results/models/td3_fx_graph
    python scripts/evaluate.py --model results/models/td3_fx_graph --regime-shift
    python scripts/evaluate.py --no-model                      # baselines only
    python scripts/evaluate.py --model ... --csv data/test_ticks.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stable_baselines3 import TD3

from fx_rl.config import Config
from fx_rl.data import make_train_test, load_csv_data
from fx_rl.evaluate import evaluate


def main():
    p = argparse.ArgumentParser(description="Evaluate TD3+GAT agent vs baselines")
    p.add_argument("--config", default=None, help="Path to YAML config (optional)")
    p.add_argument("--model", default=None, help="Path to saved model (without .zip)")
    p.add_argument("--no-model", action="store_true", help="Evaluate baselines only")
    p.add_argument("--csv", default=None, help="Evaluate on a real CSV test set")
    p.add_argument("--regime-shift", action="store_true",
                   help="Use reversed-drift / higher-vol synthetic test regime")
    p.add_argument("--periods-per-year", type=int, default=252)
    p.add_argument("--tag", default="eval", help="Output filename tag")
    args = p.parse_args()

    # Prefer the config saved next to the model so eval matches training exactly.
    if args.config:
        cfg = Config.from_yaml(args.config)
    elif args.model and os.path.exists(args.model + "_config.yaml"):
        cfg = Config.from_yaml(args.model + "_config.yaml")
        print(f"Loaded config saved with the model: {args.model}_config.yaml")
    else:
        cfg = Config()

    if args.csv:
        test_df = load_csv_data(args.csv, cfg.data)
        print(f"Loaded {len(test_df)} test ticks from {args.csv}")
    else:
        _, test_df = make_train_test(
            cfg.data,
            train_seed=cfg.data.seed,
            test_seed=cfg.train.eval_seed,
            regime_shift=args.regime_shift,
        )
        print(f"Generated synthetic test path (seed={cfg.train.eval_seed}, "
              f"regime_shift={args.regime_shift})")

    model = None
    if not args.no_model:
        if not args.model:
            p.error("--model is required unless --no-model is set")
        model = TD3.load(args.model)
        print(f"Loaded model {args.model}.zip")

    evaluate(model, cfg, test_df, periods_per_year=args.periods_per_year, tag=args.tag)


if __name__ == "__main__":
    main()
