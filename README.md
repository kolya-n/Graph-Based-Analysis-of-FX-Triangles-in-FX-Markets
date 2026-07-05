# FX-RL: Graph-Attention Reinforcement Learning for FX Portfolio Allocation

A reinforcement-learning agent that allocates a USD-denominated portfolio across
four currencies (USD, EUR, GBP, JPY). The market at each tick is encoded as a
**graph** — currencies are nodes, exchange rates are directed edges — and a
**Graph Attention Network (GAT)** produces the state embedding consumed by a
**TD3** continuous-control agent.

This repository is the cleaned, reproducible, diploma-level version of an
original Colab scratch notebook (archived at
[`notebooks/archive_fx_agent_sketch.py`](notebooks/archive_fx_agent_sketch.py)).
See [Changes vs. the original notebook](#changes-vs-the-original-notebook).

## Problem formulation (MDP)

| Component | Definition |
|-----------|------------|
| **State** | A graph with 4 currency nodes and 12 directed edges. Node features `[current_weight, previous_weight]`; edge features `[log_mid_rate, spread_pct, volatility, momentum, arb_signal]`. |
| **Action** | A 4-vector mapped to a long-only portfolio target on the simplex via `w = softmax(action * temperature)` (no leverage, no shorting; always valid). |
| **Transition** | Rebalance toward `w` at executable bid/ask prices (spread is paid), then advance one tick. |
| **Reward** | `reward_scale · log(NAV_{t+1} / NAV_t) − turnover_penalty · turnover_t`. Spread/transaction cost is reflected inside the NAV change (charged once); the turnover term is a small explicit regulariser. |
| **Termination** | Drawdown exceeds `max_drawdown` (terminated, with penalty), or the data path ends (truncated). |

## Project structure

```
src/fx_rl/
  config.py      # typed dataclasses + YAML loading + global seeding
  data.py        # synthetic GBM generator + CSV loader + train/test split
  graph.py       # FxGraphBuilder, quote inversion, spread-aware triangular arbitrage
  env.py         # FXArbitrageEnv — the corrected MDP (Gymnasium)
  extractor.py   # FxGraphExtractor — GAT feature extractor for SB3
  train.py       # TD3 + GAT training, logging, checkpointing
  baselines.py   # cash / equal-weight / buy-and-hold / random / momentum policies
  metrics.py     # return, Sharpe, Sortino, max drawdown, volatility, turnover, costs
  evaluate.py    # TD3 vs. baselines -> metrics CSV + comparison plots
tests/           # pytest suite (graph, env, metrics, baselines)
scripts/         # train.py / evaluate.py CLI entry points
configs/         # default.yaml (full run specification)
results/         # models/ logs/ plots/ metrics/  (run outputs)
notebooks/       # exploration.ipynb + archived original sketch
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell: .venv\Scripts\Activate.ps1)
pip install -r requirements.txt
```

## Usage

Training and evaluation read `src/` via a path bootstrap in the scripts, so no
install step is required.

```bash
# Train (synthetic GBM by default)
python scripts/train.py --config configs/default.yaml

# Shorter smoke run
python scripts/train.py --timesteps 20000 --seed 1

# Evaluate vs. baselines on an independent test path (different seed)
python scripts/evaluate.py --model results/models/td3_fx_graph

# Harder out-of-distribution test: reversed drift + higher volatility
python scripts/evaluate.py --model results/models/td3_fx_graph --regime-shift

# Baselines only (no trained model needed)
python scripts/evaluate.py --no-model

# Plug in real data (same column contract: <PAIR>_bid / <PAIR>_ask)
python scripts/train.py --csv data/train_ticks.csv
python scripts/evaluate.py --model results/models/td3_fx_graph --csv data/test_ticks.csv
```

TensorBoard: `tensorboard --logdir results/logs`.

## Reproducibility

`config.set_global_seeds` seeds Python/NumPy/PyTorch; TD3's `seed` argument seeds
policy init, action sampling, and env reset; data generation takes explicit
seeds. **Train and test use independent seeds** (`data.seed` vs `train.eval_seed`)
so evaluation is genuinely out-of-sample, and `--regime-shift` makes the test
path statistically distinct from training.

## Evaluation methodology

Every policy — the trained agent and all baselines — is rolled through the
*identical* environment dynamics and transaction costs via `env.act_on_weights`,
then scored with the same metrics. Baselines: **cash (USD-only)**,
**equal-weight**, **buy-and-hold**, **random**, **momentum**. Metrics: total
return, annualised Sharpe & Sortino, max drawdown, annualised volatility, average
turnover, and average/total transaction cost.

## Note on the synthetic data

The six quoted pairs are simulated as *independent* GBMs, so cross-rates are not
forced to be no-arbitrage-consistent with the majors. A genuine (small) triangular
arbitrage therefore exists in the data — which is what gives the `arb_signal` edge
feature something real to represent. The signal is **spread-aware**: it is the
return of an *executable* bid/ask conversion loop, so it correctly discounts the
theoretical mid-price edge.
