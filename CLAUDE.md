# CLAUDE.md

Guidance for working in this repository.

## What this is

A diploma RL project: a GAT-based feature extractor feeding a TD3 agent that
allocates an FX portfolio (USD/EUR/GBP/JPY). The graph state representation is the
thesis's core contribution — keep it. The original Colab scratch is archived at
`notebooks/archive_fx_agent_sketch.py` (reference only; do not depend on it).

## Environment

- Windows; Python venv at `.venv`. Run tools with `./.venv/Scripts/python.exe`.
- Source is not pip-installed. Use `PYTHONPATH=src` (tests/ad-hoc) or rely on the
  `sys.path` bootstrap in `scripts/`.
- CPU-only torch. Training 100k steps is CPU-bound; graph construction per tick
  (the triangular-arbitrage triple loop + pandas window ops) is the hot path.

## Common commands

```bash
PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/ -q     # tests (~3 min)
python scripts/train.py --timesteps 20000                          # quick train
python scripts/evaluate.py --no-model                              # baselines only
tensorboard --logdir results/logs
```

## Architecture & invariants (do not break)

- **Single config object.** `config.Config` (data/env/model/train) drives every
  module. Graph topology is derived from `data.currencies` — never hard-code the
  node/edge counts elsewhere. `N_NODE_FEATS=2`, `N_EDGE_FEATS=5` in `config.py`
  must stay in sync with `graph.py` and the env observation space.
- **Action → weights.** The agent action is raw; the env applies
  `softmax_weights(action, temperature)` to get a long-only simplex. Baselines
  bypass this and pass target weights directly to `env.act_on_weights`. Both paths
  share the same dynamics — keep them unified so comparisons stay fair.
- **Reward.** Return-based (`reward_scale * log(NAV ratio) - turnover_penalty *
  turnover`). Spread cost is charged once, via bid/ask execution in `_execute_trade`
  — do **not** add a separate dollar-scale cost term (double counting). Do not
  reintroduce a large hold bonus (it caused a do-nothing policy in the sketch).
- **Quote convention.** `get_pair_bid_ask(src, dst)` returns the executable
  (bid, ask) for converting src→dst (dst per src); selling src pays `bid`. Arbitrage
  and trades go through `convert()` / `_execute_trade` so spreads are always paid.
- **Reproducibility.** Train and test use *independent* seeds. Keep it that way;
  never evaluate on the training path.

## Conventions

- Match the existing inline-documentation style: when changing math/economics,
  write a short note on what was wrong and how it was fixed (the existing modules
  do this in module docstrings — continue the pattern).
- Add/extend tests in `tests/` for any logic change; run `check_env` after env
  changes.
- Keep `results/` outputs out of version control (see `.gitignore`); the
  `.gitkeep` files preserve the directory layout.


# External memory

This project uses `ai-memory/` as external project memory.

Before substantial work, read:

- `ai-memory/PROJECT_STATE.md`

Rules:

- Do not scan the whole repository unless necessary.
- First read the relevant memory files.
- After major work, update `ai-memory/PROJECT_STATE.md`.