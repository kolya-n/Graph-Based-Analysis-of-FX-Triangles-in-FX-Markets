"""Data generation and loading.

Provides synthetic Geometric-Brownian-Motion (GBM) tick data as the first
reproducible data source, plus a CSV hook so real bid/ask data can be dropped in
later without touching the rest of the pipeline.

Output contract (used by ``graph.py`` and ``env.py``)
-----------------------------------------------------
A ``pandas.DataFrame`` with one row per tick and two columns per quoted pair::

    EURUSD_bid, EURUSD_ask, GBPUSD_bid, GBPUSD_ask, ...

Market structure (so a smart policy can actually beat cash)
-----------------------------------------------------------
The first version simulated independent GBMs with a tiny constant drift, which —
once spreads were realistic — left nothing worth trading on (the agent rationally
sat in cash).  This version adds two exploitable structures:

* **Trend / momentum regimes:** each pair's drift switches between persistent
  trending regimes (``use_regimes``), so directional bets pay off and the agent
  must read momentum from the graph state to time them.
* **Injected triangular-arbitrage windows:** a cross pair (EURGBP/EURJPY/GBPJPY)
  is transiently perturbed so a USD->X->Y->USD cycle yields an *executable*
  profit exceeding spread cost — a real arbitrage signal to detect and exploit.

Fixes relative to the notebook
------------------------------
* The notebook generated one global ``df`` with a single ``np.random.seed(42)``
  and then sliced ``df[:4000]`` / ``df[4000:]`` for train/test — so the test set
  was the *same* GBM realization as train.  Here ``make_train_test`` draws train
  and test from **independent seeds**, with an optional regime shift.
* Generation takes an explicit ``seed`` / ``np.random.Generator`` instead of
  mutating global RNG state.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import DataConfig


def _regime_drift_series(
    rng: np.random.Generator,
    n: int,
    base_drift: float,
    trend_strength: float,
    switch_prob: float,
) -> np.ndarray:
    """Per-tick drift that switches between persistent trending regimes.

    A regime drift ~ N(0, trend_strength) is held until a switch occurs (prob
    ``switch_prob`` per tick), then redrawn. ``base_drift`` is added as a constant
    baseline, so trends are persistent but mean-zero in expectation.
    """
    drifts = np.empty(n, dtype=np.float64)
    current = rng.normal(0.0, trend_strength)
    for t in range(n):
        if rng.random() < switch_prob:
            current = rng.normal(0.0, trend_strength)
        drifts[t] = base_drift + current
    return drifts


def _inject_arbitrage(rng: np.random.Generator, mids: dict, cfg: DataConfig) -> None:
    """Transiently perturb cross pair(s) to create executable arbitrage windows.

    Each event (probability ``arb_injection_prob`` per tick) has:
      * a random magnitude ``~U(arb_mag_min, arb_magnitude)`` and random sign;
      * a random duration ``~randint(arb_dur_min, arb_duration)``;
      * a decay profile (``arb_decay``) so the mispricing shrinks to 0 over the
        window — loosely modelling arbitrageurs closing the gap, instead of a
        rectangular pulse that snaps back;
      * with probability ``arb_multileg_prob``, it perturbs **two** cross legs (a
        "multi-leg" event) which can make the pure-cross EUR/GBP/JPY triangle the
        best opportunity; otherwise it perturbs one cross pair (lighting up the
        corresponding USD-anchored triangle).

    Because cross rates feed the triangular cycles, this breaks no-arbitrage
    consistency by more than the (tightened) spread cost, so a real executable
    arbitrage exists during the window. Mutates ``mids`` in place.
    """
    cross = [p for p in cfg.base if "USD" not in p]
    if not cross or cfg.arb_injection_prob <= 0 or cfg.arb_magnitude <= 0:
        return
    n = cfg.n_ticks
    for start in range(n):
        if rng.random() >= cfg.arb_injection_prob:
            continue
        mag = rng.uniform(cfg.arb_mag_min, cfg.arb_magnitude)
        dur = int(rng.integers(cfg.arb_dur_min, cfg.arb_duration + 1))
        end = min(start + dur, n)
        length = end - start
        # decay profile: full at detection, linearly to ~0 (arbitraged away)
        profile = (1.0 - np.arange(length) / length) if cfg.arb_decay else np.ones(length)

        if len(cross) >= 2 and rng.random() < cfg.arb_multileg_prob:
            idx = rng.choice(len(cross), size=2, replace=False)
            chosen = [cross[int(i)] for i in idx]
        else:
            chosen = [cross[int(rng.integers(len(cross)))]]

        for pair in chosen:
            sign = 1.0 if rng.random() < 0.5 else -1.0
            mids[pair][start:end] = mids[pair][start:end] * (1.0 + sign * mag * profile)


def _simulate_usd_rates(rng: np.random.Generator, cfg: DataConfig, n: int) -> dict:
    """Simulate USD-per-currency rate drivers (no-arbitrage-consistent base).

    Each non-USD currency ``c`` gets one independent GBM series ``rate_c`` = USD
    per 1 unit of ``c``, seeded from its USD major (``cUSD`` directly, or ``USDc``
    inverted). ``rate_USD = 1``. All quoted pair mids are later derived as
    ``rate[base]/rate[quote]``, so cross rates are arbitrage-consistent by
    construction — unlike the previous version, which simulated all 6 pairs
    independently and thus produced large *unintended* triangular arbitrage.
    """
    rates: dict[str, np.ndarray] = {"USD": np.ones(n, dtype=np.float64)}
    for c in cfg.currencies:
        if c == "USD":
            continue
        if f"{c}USD" in cfg.base:                      # e.g. EURUSD: USD per EUR
            p0, vol, base_drift = cfg.base[f"{c}USD"], cfg.vols[f"{c}USD"], cfg.drifts[f"{c}USD"]
        elif f"USD{c}" in cfg.base:                    # e.g. USDJPY: invert to USD per JPY
            p0 = 1.0 / cfg.base[f"USD{c}"]
            vol = cfg.vols[f"USD{c}"]                  # vol is identical in log space
            base_drift = -cfg.drifts[f"USD{c}"]        # inverse flips drift sign
        else:
            raise ValueError(f"No USD major pair found for currency {c}")

        if cfg.use_regimes:
            drift = _regime_drift_series(rng, n, base_drift, cfg.trend_strength, cfg.regime_switch_prob)
        else:
            drift = np.full(n, base_drift, dtype=np.float64)
        returns = drift + vol * rng.standard_normal(n)
        rates[c] = p0 * np.cumprod(np.exp(returns))
    return rates


def generate_gbm_data(cfg: DataConfig, seed: Optional[int] = None) -> pd.DataFrame:
    """Generate synthetic bid/ask tick data (regime trends + injected arbitrage).

    1. Simulate USD-per-currency rate drivers with regime-switching drift
       (no-arbitrage-consistent base).
    2. Derive every quoted pair mid as ``rate[base]/rate[quote]``.
    3. Inject transient triangular-arbitrage windows into cross pairs.
    4. Apply a constant half-spread to obtain bid/ask.

    A local ``np.random.Generator`` is used so calls are independent and do not
    touch the global RNG.
    """
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    n = cfg.n_ticks

    rates = _simulate_usd_rates(rng, cfg, n)
    mids: dict[str, np.ndarray] = {
        pair: rates[pair[:3]] / rates[pair[3:]] for pair in cfg.base
    }

    _inject_arbitrage(rng, mids, cfg)

    data: dict[str, np.ndarray] = {}
    for pair in cfg.base:
        half_spread = cfg.spreads[pair] / 2.0
        data[f"{pair}_bid"] = mids[pair] - half_spread
        data[f"{pair}_ask"] = mids[pair] + half_spread

    return pd.DataFrame(data)


def load_csv_data(path: str, cfg: Optional[DataConfig] = None) -> pd.DataFrame:
    """Load real tick data from CSV.

    Expected columns: ``<PAIR>_bid`` / ``<PAIR>_ask`` for every quoted pair, e.g.
    ``EURUSD_bid, EURUSD_ask, ...``.  This is the same contract the synthetic
    generator produces, so the rest of the pipeline is data-source agnostic.
    """
    df = pd.read_csv(path)
    if cfg is not None:
        missing = [
            f"{pair}_{side}"
            for pair in cfg.base
            for side in ("bid", "ask")
            if f"{pair}_{side}" not in df.columns
        ]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
    return df.reset_index(drop=True)


def _regime_shift(cfg: DataConfig) -> DataConfig:
    """Return a copy of ``cfg`` with a different market regime for the test set.

    We flip the sign of every drift and inflate volatility by 50%.  This makes the
    test path statistically distinct from train (trend reversal + higher noise),
    so an agent that merely memorised the training trend is exposed.
    """
    import copy

    shifted = copy.deepcopy(cfg)
    shifted.drifts = {k: -v for k, v in cfg.drifts.items()}
    shifted.vols = {k: 1.5 * v for k, v in cfg.vols.items()}
    return shifted


def make_train_test(
    cfg: DataConfig,
    train_seed: int,
    test_seed: int,
    regime_shift: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Produce independent train and test tick paths.

    Parameters
    ----------
    cfg:
        Base data configuration.
    train_seed, test_seed:
        Independent seeds — the two paths are different GBM realizations.
    regime_shift:
        If True, the test path uses reversed drift and higher volatility (see
        ``_regime_shift``) for a harder out-of-distribution evaluation.
    """
    train_df = generate_gbm_data(cfg, seed=train_seed)
    test_cfg = _regime_shift(cfg) if regime_shift else cfg
    test_df = generate_gbm_data(test_cfg, seed=test_seed)
    return train_df, test_df
