"""Evaluation rollouts of a trained policy under several sensor-noise levels (Section 7.1, Appendix D).

For every noise level nu, ``EVAL_EPISODES`` episodes are run with the deterministic (mean) action.
The true forward velocity v_k, the reward, the cost and the training dissimilarity are recorded at
every step k; entries after a termination are NaN. The distributions rho_k are then estimated
across episodes (see ``metrics``).

The results are stored in ``evaluation.npz`` with keys ``"nu{nu:.2f}/{name}"``, e.g.
``"nu0.20/velocity"``, each an array of shape (episodes, horizon).
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import torch

from .agent import Agent, RunningMeanStd
from .config import EPISODE_LENGTH, TrainConfig
from .envs import make_env

EVAL_SEED = 12345        # episode j is reset with the seed EVAL_SEED + 7919 * j
RECORDED = ("velocity", "reward", "cost", "dissimilarity")


def load_policy(path: str) -> tuple[Agent, RunningMeanStd, TrainConfig]:
    """Loads a ``policy.pt`` written by ``train.train``."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**ckpt["cfg"])
    agent = Agent(ckpt["obs_dim"], ckpt["act_dim"], cfg.hidden)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()
    rms = RunningMeanStd(ckpt["obs_dim"])
    rms.load_state_dict(ckpt["rms"])
    return agent, rms, cfg


@torch.no_grad()
def rollout(agent: Agent, rms: RunningMeanStd, cfg: TrainConfig, nu: float, episodes: int,
            horizon: int = EPISODE_LENGTH, seed: int = EVAL_SEED) -> dict[str, np.ndarray]:
    """Runs ``episodes`` episodes at the fixed noise level ``nu`` with the deterministic policy.

    Returns a dict with the arrays ``velocity``, ``reward``, ``cost`` and ``dissimilarity`` of
    shape (episodes, horizon); entries after the end of an episode are NaN.
    """
    envs = [make_env(cfg.env_id, nu, seed + 7919 * j, cfg.window, cfg.policy_input, cfg.mu_star, cfg.sd_star,
                     cfg.vel_index)
            for j in range(episodes)]
    obs = np.stack([e.reset(seed=seed + 7919 * j)[0] for j, e in enumerate(envs)])
    alive = np.ones(episodes, bool)
    rec = {k: np.full((episodes, horizon), np.nan) for k in RECORDED}
    low, high = envs[0].action_space.low, envs[0].action_space.high
    for k in range(horizon):
        if not alive.any():
            break
        x = torch.as_tensor(rms.normalize(obs))
        action, _, _ = agent.act(x, deterministic=True)
        action = np.clip(action.numpy(), low, high)
        for j, e in enumerate(envs):
            if not alive[j]:
                continue
            o, r, terminated, truncated, info = e.step(action[j])
            obs[j] = o
            rec["velocity"][j, k] = info["x_velocity"]
            rec["reward"][j, k] = r
            rec["cost"][j, k] = info["cost"]
            rec["dissimilarity"][j, k] = info["dissimilarity"]
            if terminated or truncated:
                alive[j] = False
    for e in envs:
        e.close()
    return rec


def evaluate_policy(path: str, noises: Iterable[float], episodes: int, out_path: str | None = None,
                    horizon: int = EPISODE_LENGTH) -> dict[str, np.ndarray]:
    """Evaluates the policy saved at ``path`` at every noise level in ``noises``.

    Returns (and, if ``out_path`` is given, saves as a compressed ``.npz``) a dict with the keys
    ``"nu{nu:.2f}/{name}"`` for ``name`` in ``RECORDED``.
    """
    agent, rms, cfg = load_policy(path)
    torch.set_num_threads(1)
    result = {}
    for nu in noises:
        rec = rollout(agent, rms, cfg, nu, episodes, horizon=horizon)
        for key, arr in rec.items():
            result[f"nu{nu:.2f}/{key}"] = arr
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        np.savez_compressed(out_path, **result)
    return result
