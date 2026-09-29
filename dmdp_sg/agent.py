"""Actor-critic network of the PPO-Lagrangian learner (Appendix D).

The actor, the reward critic and the cost critic are separate multilayer perceptrons with two
hidden tanh layers. The actor outputs the mean of a diagonal Gaussian distribution with a learned,
state-independent standard deviation; the mean action is used for evaluation. Inputs are
normalized with running estimates of their mean and variance (``RunningMeanStd``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


class RunningMeanStd:
    """Running mean and variance of the policy input, used to normalize observations."""

    def __init__(self, shape: int | tuple[int, ...], epsilon: float = 1e-4):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        """Merge a batch ``x`` (first axis = samples) into the running statistics."""
        x = np.asarray(x, np.float64)
        batch_mean, batch_var, batch_count = x.mean(0), x.var(0), x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot
        m2 = self.var * self.count + batch_var * batch_count + delta ** 2 * self.count * batch_count / tot
        self.var = m2 / tot
        self.count = tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Standardized input, clipped to [-10, 10], as float32."""
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10.0, 10.0).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return dict(mean=self.mean, var=self.var, count=self.count)

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.mean, self.var, self.count = np.asarray(d["mean"]), np.asarray(d["var"]), float(d["count"])


def _layer(n_in: int, n_out: int, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    """Linear layer with orthogonal weight initialization."""
    layer = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


def _mlp(n_in: int, n_out: int, hidden: int, out_std: float) -> nn.Sequential:
    """Two hidden tanh layers of width ``hidden``."""
    return nn.Sequential(_layer(n_in, hidden), nn.Tanh(), _layer(hidden, hidden), nn.Tanh(),
                         _layer(hidden, n_out, std=out_std))


class Agent(nn.Module):
    """Gaussian actor with separate reward and cost critics.

    The attribute names are the keys of the saved state dict (``policy.pt``) and must not change.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.critic = _mlp(obs_dim, 1, hidden, 1.0)
        self.cost_critic = _mlp(obs_dim, 1, hidden, 1.0)
        self.actor_mean = _mlp(obs_dim, act_dim, hidden, 0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Reward value V_r(x)."""
        return self.critic(x).squeeze(-1)

    def cost_value(self, x: torch.Tensor) -> torch.Tensor:
        """Cost value V_c(x)."""
        return self.cost_critic(x).squeeze(-1)

    def dist(self, x: torch.Tensor) -> torch.distributions.Normal:
        """Action distribution N(mean(x), diag(exp(logstd))^2)."""
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        return torch.distributions.Normal(mean, std)

    def act(self, x: torch.Tensor, deterministic: bool = False
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sampled (or mean) action, its log-probability and the entropy of the distribution."""
        d = self.dist(x)
        a = d.mean if deterministic else d.sample()
        return a, d.log_prob(a).sum(-1), d.entropy().sum(-1)
