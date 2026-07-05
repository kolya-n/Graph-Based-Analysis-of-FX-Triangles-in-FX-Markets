"""Baseline policies and a shared rollout driver.

A diploma-level evaluation needs reference strategies to judge the RL agent
against.  Each policy implements ``predict_weights(obs, env) -> target weights``
(a point on the long-only simplex).  Every policy — baselines and the trained
TD3 agent alike — is rolled out through ``env.act_on_weights`` so they all face
identical dynamics, prices, and transaction costs.

Baselines
---------
* ``CashPolicy``        — hold 100% USD (the risk-free-ish benchmark).
* ``EqualWeightPolicy`` — rebalance to 1/N across all currencies every tick.
* ``BuyAndHoldPolicy``  — buy an equal-weight basket once, then never trade.
* ``RandomPolicy``      — a random simplex point each tick (seeded).
* ``MomentumPolicy``    — rule-based: tilt toward currencies appreciating vs USD.

This module is an addition to the structure in the task spec; baseline policies
were requested in step 6 but did not have a designated file, so they live here
rather than being crammed into ``metrics.py``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .env import softmax_weights, action_to_arb_intensity


class Policy:
    name: str = "policy"

    def reset(self) -> None:  # optional state reset between episodes
        pass

    def predict_weights(self, obs: dict, env) -> np.ndarray:
        raise NotImplementedError

    def predict_arb_intensity(self, obs: dict, env) -> float:
        """Fraction of USD to route through the best arbitrage cycle (default: none)."""
        return 0.0


class CashPolicy(Policy):
    name = "cash_usd"

    def predict_weights(self, obs, env):
        w = np.zeros(env.cfg.data.n_currencies, dtype=np.float32)
        w[0] = 1.0  # USD is index 0
        return w


class EqualWeightPolicy(Policy):
    name = "equal_weight"

    def predict_weights(self, obs, env):
        n = env.cfg.data.n_currencies
        return np.full(n, 1.0 / n, dtype=np.float32)


class BuyAndHoldPolicy(Policy):
    """Buy an equal-weight basket on the first step, then hold (never rebalance).

    After the initial purchase we target the *current* drifted weights, so
    ``_rebalance`` issues no trades and the basket simply rides the market.
    """

    name = "buy_and_hold"

    def __init__(self):
        self._initialised = False

    def reset(self):
        self._initialised = False

    def predict_weights(self, obs, env):
        n = env.cfg.data.n_currencies
        if not self._initialised:
            self._initialised = True
            return np.full(n, 1.0 / n, dtype=np.float32)
        return env.weights.astype(np.float32)  # hold: target == current


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, seed: Optional[int] = 0):
        self.rng = np.random.default_rng(seed)
        self._seed = seed

    def reset(self):
        self.rng = np.random.default_rng(self._seed)

    def predict_weights(self, obs, env):
        n = env.cfg.data.n_currencies
        return self.rng.dirichlet(np.ones(n)).astype(np.float32)


class MomentumPolicy(Policy):
    """Rule-based: allocate to currencies appreciating against USD.

    Uses the per-edge ``momentum`` feature (index 3) of the ``(ccy, USD)`` edge.
    Weight is proportional to positive momentum; if no currency has positive
    momentum, stay in USD.
    """

    name = "momentum"

    def predict_weights(self, obs, env):
        edges = obs["edges"]
        edge_pairs = env.builder.edge_pairs
        n = env.cfg.data.n_currencies
        scores = np.zeros(n, dtype=np.float64)
        for i, ccy in enumerate(env.currencies):
            if ccy == "USD":
                continue
            try:
                idx = edge_pairs.index((ccy, "USD"))
                scores[i] = max(0.0, float(edges[idx, 3]))  # positive momentum only
            except ValueError:
                scores[i] = 0.0
        if scores.sum() <= 1e-12:
            w = np.zeros(n, dtype=np.float32)
            w[0] = 1.0  # all USD
            return w
        return (scores / scores.sum()).astype(np.float32)


class ArbitragePolicy(Policy):
    """Oracle arbitrageur: hold a diversified (equal-weight) book and fully exploit
    every profitable cycle.

    With multi-currency funding, capturing a cycle requires holding its currencies,
    so a cash-only book could only fund USD cycles. Holding equal weights lets this
    oracle fund any of the 8 directed triangles. It commits full intensity every
    tick (the env gates to profitable cycles), showing the practical arbitrage
    ceiling for a diversified book.
    """

    name = "arbitrage"

    def predict_weights(self, obs, env):
        n = env.cfg.data.n_currencies
        return np.full(n, 1.0 / n, dtype=np.float32)

    def predict_arb_intensity(self, obs, env):
        return 1.0  # env executes only profitable cycles


class TD3Policy(Policy):
    """Wraps a trained SB3 TD3 model; splits its action into weights + arb intensity."""

    name = "td3_gat"

    def __init__(self, model, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic
        self._last_action = None

    def predict_weights(self, obs, env):
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        self._last_action = np.asarray(action, dtype=np.float32)
        return softmax_weights(self._last_action[: env.n_ccy], env.cfg.env.action_temperature)

    def predict_arb_intensity(self, obs, env):
        # reuse the action cached by predict_weights (same obs, deterministic)
        return action_to_arb_intensity(self._last_action[env.n_ccy])


def default_baselines(random_seed: int = 0):
    """The standard baseline set used in evaluation."""
    return [
        CashPolicy(),
        EqualWeightPolicy(),
        BuyAndHoldPolicy(),
        RandomPolicy(seed=random_seed),
        MomentumPolicy(),
        ArbitragePolicy(),
    ]


def run_policy(env, policy: Policy, seed: Optional[int] = None) -> dict:
    """Roll a policy through one full episode; return per-step series."""
    policy.reset()
    obs, _ = env.reset(seed=seed)
    navs = [env.portfolio_history[-1]]
    turnovers, costs, rewards, arb_profits = [], [], [], []
    done = False
    while not done:
        target = policy.predict_weights(obs, env)
        arb_intensity = policy.predict_arb_intensity(obs, env)
        obs, reward, terminated, truncated, info = env.act_on_weights(target, arb_intensity)
        navs.append(info["nav"])
        turnovers.append(info["turnover"])
        costs.append(info["transaction_cost"])
        arb_profits.append(info["arb_profit"])
        rewards.append(reward)
        done = terminated or truncated
    return {
        "name": policy.name,
        "nav": navs,
        "turnover": turnovers,
        "transaction_cost": costs,
        "arb_profit": arb_profits,
        "reward": rewards,
    }
