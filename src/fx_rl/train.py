"""TD3 + GAT training.

Wires the corrected environment, the GAT feature extractor, and SB3's TD3 into a
single reproducible ``train(cfg)`` entry point.  Everything is driven by the
``Config`` object so a run is fully described by one YAML file plus the seeds.

Reproducibility (fixing the notebook, which only set ``np.random.seed(42)``):
``set_global_seeds`` seeds python/numpy/torch; the TD3 ``seed`` argument seeds the
policy initialisation, the action space sampling, and the env reset; the action
noise is constructed deterministically.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from .config import Config, set_global_seeds
from .data import generate_gbm_data
from .env import FXArbitrageEnv
from .extractor import FxGraphExtractor


class PortfolioCallback(BaseCallback):
    """Logs NAV / turnover / reward to TensorBoard and keeps per-episode arrays."""

    def __init__(self, log_every: int = 500, total_timesteps: int = 100_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_every = log_every
        self.total_timesteps = total_timesteps
        self.episode_rewards: list[float] = []
        self.episode_navs: list[float] = []
        self._ep_reward = 0.0
        self._recent_navs: list[float] = []

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        reward = float(self.locals["rewards"][0])
        nav = float(info.get("nav", 0.0))

        self._ep_reward += reward
        self._recent_navs.append(nav)

        # stream scalars to TensorBoard
        self.logger.record("portfolio/nav", nav)
        self.logger.record("portfolio/turnover", float(info.get("turnover", 0.0)))
        self.logger.record("portfolio/transaction_cost", float(info.get("transaction_cost", 0.0)))

        if self.num_timesteps % self.log_every == 0:
            window = self._recent_navs[-self.log_every:]
            ep_avg = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0.0
            if self.verbose:
                print(
                    f"[{self.num_timesteps:>7}/{self.total_timesteps}] "
                    f"NAV avg ${np.mean(window):,.2f} "
                    f"[{np.min(window):,.0f}-{np.max(window):,.0f}]  "
                    f"R_ep(last10): {ep_avg:+.4f}  episodes: {len(self.episode_rewards)}"
                )

        if self.locals["dones"][0]:
            self.episode_rewards.append(self._ep_reward)
            self.episode_navs.append(nav)
            self._ep_reward = 0.0
        return True


def make_env(cfg: Config, df: pd.DataFrame):
    """Factory returning a Monitor-wrapped FXArbitrageEnv (for DummyVecEnv)."""
    def _init():
        return Monitor(FXArbitrageEnv(cfg, df))
    return _init


def build_model(cfg: Config, train_env) -> TD3:
    policy_kwargs = dict(
        features_extractor_class=FxGraphExtractor,
        features_extractor_kwargs=dict(
            features_dim=cfg.model.features_dim,
            gat_hidden=cfg.model.gat_hidden,
            gat_heads=cfg.model.gat_heads,
            gat_out=cfg.model.gat_out,
        ),
        net_arch=cfg.model.net_arch,
    )
    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=cfg.model.action_noise_sigma * np.ones(n_actions),
    )
    return TD3(
        "MultiInputPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=cfg.model.learning_rate,
        batch_size=cfg.model.batch_size,
        buffer_size=cfg.model.buffer_size,
        learning_starts=cfg.model.learning_starts,
        train_freq=cfg.model.train_freq,
        gradient_steps=cfg.model.gradient_steps,
        gamma=cfg.model.gamma,
        action_noise=action_noise,
        tensorboard_log=cfg.train.log_dir,
        seed=cfg.train.seed,
        verbose=0,
    )


def train(cfg: Config, train_df: Optional[pd.DataFrame] = None, save: bool = True):
    """Run a full TD3+GAT training and (optionally) save the model + config.

    Returns ``(model, callback, train_df)``.
    """
    set_global_seeds(cfg.train.seed)

    if train_df is None:
        train_df = generate_gbm_data(cfg.data, seed=cfg.data.seed)

    train_env = DummyVecEnv([make_env(cfg, train_df)])
    model = build_model(cfg, train_env)

    callback = PortfolioCallback(
        log_every=cfg.train.log_every,
        total_timesteps=cfg.train.total_timesteps,
        verbose=1,
    )
    model.learn(total_timesteps=cfg.train.total_timesteps, callback=callback)

    if save:
        os.makedirs(cfg.train.model_dir, exist_ok=True)
        model_path = os.path.join(cfg.train.model_dir, cfg.train.model_name)
        model.save(model_path)
        cfg.to_yaml(model_path + "_config.yaml")
        print(f"Saved model to {model_path}.zip and config to {model_path}_config.yaml")

    return model, callback, train_df
