"""Central configuration for the FX-RL project.

Everything that the original scratch notebook hard-coded inline (currency list,
GBM parameters, window size, reward coefficients, TD3 hyper-parameters, seeds)
lives here as typed dataclasses.  A single ``Config`` object is threaded through
``data`` / ``graph`` / ``env`` / ``train`` / ``evaluate`` so that one YAML file
fully determines a run.

Design notes / fixes relative to the notebook
---------------------------------------------
* The notebook scattered ``CURRENCIES``, ``WINDOW_SIZE``, ``N_NODE_FEATS`` … as
  module globals and re-created ``builder`` / ``train_df`` many times.  Here the
  graph topology is *derived* from ``DataConfig.currencies`` so it can never get
  out of sync with the data.
* ``N_NODE_FEATS`` is now 2 = ``[current_weight, previous_weight]`` per currency
  (see ``env.py``), which is how the previous allocation enters the observation.
* ``N_EDGE_FEATS`` is 5 = ``[log_mid_rate, spread_pct, volatility, momentum,
  arb_signal]`` (unchanged count, but ``arb_signal`` is now spread-aware — see
  ``graph.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from typing import Dict, List, Tuple
import hashlib
import itertools
import json
import random
import warnings

import numpy as np

try:  # PyYAML is optional at import time; only needed for from_yaml/to_yaml.
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


# Number of features the rest of the pipeline expects.  Kept as constants so
# observation spaces and the GAT extractor agree on dimensions.
N_NODE_FEATS = 3  # [current_weight, previous_weight, best_arb_rate_from_this_currency]
N_EDGE_FEATS = 5  # [log_mid_rate, spread_pct, volatility, momentum, arb_signal]
# NB: edge/node features are normalized to comparable scale in graph.py — without
# this the absolute log-rate (std ~3.6) drowns the arb/momentum signals (std ~1e-4)
# by ~10^4x and the GAT cannot see them.


@dataclass
class DataConfig:
    """Synthetic-data (GBM) generation parameters.

    The ``base`` / ``vols`` / ``spreads`` / ``drifts`` dicts are keyed by the
    *quoted* pair (e.g. ``"EURUSD"``).  Only these pairs are simulated; any other
    directed rate (e.g. USD->EUR) is derived by inversion in ``graph.py``.
    """

    currencies: List[str] = field(default_factory=lambda: ["USD", "EUR", "GBP", "JPY"])
    quoted_pairs: List[str] = field(
        default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY", "EURGBP", "EURJPY", "GBPJPY"]
    )
    n_ticks: int = 5000
    seed: int = 42

    base: Dict[str, float] = field(default_factory=lambda: {
        "EURUSD": 1.0850, "GBPUSD": 1.2720, "USDJPY": 149.80,
        "EURGBP": 0.8527, "EURJPY": 162.50, "GBPJPY": 190.60,
    })
    vols: Dict[str, float] = field(default_factory=lambda: {
        "EURUSD": 0.00020, "GBPUSD": 0.00025, "USDJPY": 0.00030,
        "EURGBP": 0.00015, "EURJPY": 0.00035, "GBPJPY": 0.00040,
    })
    # Spreads tightened toward realistic FX-major levels (~0.2-0.6 bps) so that
    # genuine signals (trends, injected arbitrage) become exploitable. The old
    # values (~1.4 bps on EURUSD) were wide enough that the tiny drift never paid
    # for a round trip, so the agent rationally sat in cash.
    spreads: Dict[str, float] = field(default_factory=lambda: {
        "EURUSD": 0.00003, "GBPUSD": 0.00004, "USDJPY": 0.004,
        "EURGBP": 0.00003, "EURJPY": 0.006, "GBPJPY": 0.008,
    })
    # Baseline (constant) drift; when ``use_regimes`` is on this is augmented by a
    # regime-switching trend component (see data.py).
    drifts: Dict[str, float] = field(default_factory=lambda: {
        "EURUSD": 0.000002, "GBPUSD": 0.000003, "USDJPY": -0.000005,
        "EURGBP": -0.000001, "EURJPY": 0.000003, "GBPJPY": 0.000004,
    })

    # --- trend / momentum regimes ---------------------------------------
    # Each pair's drift switches between persistent trending regimes, so a smart
    # policy must read momentum from the graph state to time directional bets.
    use_regimes: bool = True
    regime_switch_prob: float = 0.005   # per-tick switch prob (~1/200 ticks per regime)
    trend_strength: float = 0.00004     # std of regime drift (~0.4 bps/tick when trending)

    # --- injected triangular-arbitrage windows --------------------------
    # Transiently perturb cross pair(s) so a triangular cycle yields an *executable*
    # profit exceeding spread cost. Modelled to (loosely) resemble real mispricings:
    # random size, random duration, and a decay profile (the edge is "arbitraged
    # away" over the window rather than snapping back). Single-pair events light up a
    # USD-anchored triangle; multi-leg events can light up the pure-cross triangle.
    arb_injection_prob: float = 0.01    # per-tick prob a new window starts
    arb_mag_min: float = 0.0005         # min mispricing (~5 bps)
    arb_magnitude: float = 0.0025       # max mispricing (~25 bps); drawn U(min, max)
    arb_dur_min: int = 3                # min window length (ticks)
    arb_duration: int = 10              # max window length; drawn randint(min, max)
    arb_decay: bool = True              # mispricing decays to 0 over the window
    arb_multileg_prob: float = 0.3      # prob an event perturbs 2 cross legs (richer triangles)

    @property
    def n_currencies(self) -> int:
        return len(self.currencies)

    @property
    def directed_edges(self) -> List[Tuple[str, str]]:
        """All ordered currency pairs (i != j) — the directed graph edges."""
        return [(a, b) for a, b in itertools.permutations(self.currencies, 2)]

    @property
    def n_edges(self) -> int:
        return len(self.directed_edges)


@dataclass
class EnvConfig:
    """Environment / MDP parameters."""

    window_size: int = 20
    initial_balance_usd: float = 10_000.0
    max_drawdown: float = 0.50          # episode terminates past this drawdown
    drawdown_penalty: float = 1.0       # reward penalty on drawdown bust (in return units)
    turnover_penalty: float = 0.005     # lambda on L1 turnover (return units per unit turnover)
    # Spread/transaction cost is charged implicitly by trading at bid/ask in the
    # graph quotes; ``turnover_penalty`` is a small *additional* explicit nudge.
    reward_scale: float = 100.0         # log-return is ~1e-4/tick; scale so it is O(1e-2)
    action_temperature: float = 3.0     # softmax temperature mapping action -> weights
    # temperature 3 lets the agent concentrate up to ~99% in one currency
    # (softmax([3,-3,-3,-3])); temperature 1 caps concentration near 71%.


@dataclass
class ModelConfig:
    """GAT extractor + TD3 hyper-parameters."""

    features_dim: int = 128
    gat_hidden: int = 16
    gat_heads: int = 4
    gat_out: int = 32
    net_arch: List[int] = field(default_factory=lambda: [128, 64])

    learning_rate: float = 1e-4
    batch_size: int = 256
    buffer_size: int = 100_000
    learning_starts: int = 2000
    train_freq: int = 1
    gradient_steps: int = 1
    action_noise_sigma: float = 0.15   # raised from 0.1 for more exploration (less sticky)
    gamma: float = 0.99


@dataclass
class TrainConfig:
    total_timesteps: int = 100_000
    seed: int = 0
    log_every: int = 500
    eval_seed: int = 1          # different seed -> different test GBM path
    train_frac: float = 0.8     # fraction of synthetic ticks used for training
    model_dir: str = "results/models"
    log_dir: str = "results/logs"
    plot_dir: str = "results/plots"
    metrics_dir: str = "results/metrics"
    model_name: str = "td3_fx_graph"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # ---- reproducibility ----------------------------------------------
    def data_fingerprint(self) -> str:
        """Stable short hash of the data-generation config.

        Saved with the model so a loaded config can be checked against the data it
        was trained on. Catches the silent-drift bug where new data-gen fields were
        added with defaults, so an old saved config no longer reproduces its dataset.
        """
        payload = json.dumps(asdict(self.data), sort_keys=True, default=str)
        return hashlib.sha1(payload.encode()).hexdigest()[:12]

    # ---- serialization -------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        if yaml is None:  # pragma: no cover
            raise RuntimeError("PyYAML is required to load YAML configs.")
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        stored_fp = raw.pop("_data_fingerprint", None)
        cfg = cls.from_dict(raw)
        cfg._warn_if_irreproducible(raw, path, stored_fp)
        return cfg

    def _warn_if_irreproducible(self, raw: dict, path: str, stored_fp) -> None:
        """Warn loudly if a *saved* config can't reproduce its original run.

        A saved config is meant to be complete; missing fields (schema evolved
        since save) or a fingerprint mismatch mean the rebuilt config will generate
        different data / behave differently. (``from_dict`` itself stays silent —
        partial dicts are a legitimate way to override a few keys.)
        """
        for name, obj in (("data", self.data), ("env", self.env),
                          ("model", self.model), ("train", self.train)):
            section = raw.get(name)
            if section is None:
                continue
            missing = {f.name for f in fields(obj)} - set(section)
            if missing:
                warnings.warn(
                    f"Saved config '{path}' section '{name}' is missing {sorted(missing)} "
                    f"(schema evolved since it was saved); current defaults were used, so it "
                    f"may NOT reproduce the original run.",
                    stacklevel=3,
                )
        if stored_fp is not None and self.data_fingerprint() != stored_fp:
            warnings.warn(
                f"Data-config fingerprint mismatch in '{path}': stored {stored_fp} != "
                f"rebuilt {self.data_fingerprint()} — data generation differs from the "
                f"original run. Re-train to refresh canonical numbers.",
                stacklevel=3,
            )

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        """Build a Config, overriding only the keys present in ``raw``."""
        cfg = cls()
        for section_name, section_obj in (
            ("data", cfg.data), ("env", cfg.env),
            ("model", cfg.model), ("train", cfg.train),
        ):
            section_raw = raw.get(section_name, {}) or {}
            for key, value in section_raw.items():
                if not hasattr(section_obj, key):
                    raise KeyError(f"Unknown config key '{section_name}.{key}'")
                setattr(section_obj, key, value)
        return cfg

    def to_dict(self) -> dict:
        return {
            "data": asdict(self.data),
            "env": asdict(self.env),
            "model": asdict(self.model),
            "train": asdict(self.train),
        }

    def to_yaml(self, path: str) -> None:
        if yaml is None:  # pragma: no cover
            raise RuntimeError("PyYAML is required to write YAML configs.")
        payload = self.to_dict()
        payload["_data_fingerprint"] = self.data_fingerprint()  # repro guard on reload
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True)


def set_global_seeds(seed: int) -> None:
    """Seed every RNG that affects a run.

    The notebook only called ``np.random.seed(42)``; torch, the Python ``random``
    module, and (downstream) the gym env / SB3 were left unseeded, so training was
    not reproducible.  This seeds all of them.  Gymnasium and SB3 are seeded
    separately at construction time (``env.reset(seed=...)`` and ``model = TD3(..,
    seed=..)``) — see ``train.py``.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass
