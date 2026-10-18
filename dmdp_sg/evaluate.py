"""Evaluation rollouts of a trained policy under several sensor-noise levels (Section 5).

For every noise level nu, ``config.EVAL_EPISODES`` episodes are run with the deterministic (mean) action.
The true forward velocity v_k, the reward r_k + r_kD (reward r_k = v_k - (kappa c_k)^2 - 0.1 ||a_k||^2 of the run, the
benchmark reward for kappa = 0, plus the threshold reward r_kD, which is 0 for the baselines), the cost, the training dissimilarity, the threshold reward r_kD alone ("bonus") and the pitch
angle of the torso are recorded at every step k; entries after a termination are NaN. The distributions rho_k are then estimated
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
from .config import EPISODE_LENGTH, TrainConfig, config_from_dict
from .envs import env_settings, make_env

EVAL_SEED = 12345        # episode j is reset with the seed EVAL_SEED + 7919 * j
RECORDED = ("velocity", "reward", "cost", "dissimilarity", "bonus", "pitch")


def load_policy(path: str) -> tuple[Agent, RunningMeanStd, TrainConfig]:
    """Loads a ``policy.pt`` written by ``train.train`` (PPO) or ``sac.train_sac`` (SAC).

    Returns the agent, the normalizer of the observations and the configuration of the run. For a SAC run these are a
    ``sac.SACAgent`` and a ``sac.IdentityNormalizer``, which have the interface of ``agent.Agent`` and
    ``agent.RunningMeanStd`` that ``rollout`` uses.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = config_from_dict(ckpt["cfg"])
    if cfg.learner == "sac":
        from .sac import IdentityNormalizer, SACAgent
        space = make_env(nu=0.0, seed=0, **env_settings(cfg)).action_space
        sac_agent = SACAgent(ckpt["obs_dim"], ckpt["act_dim"], cfg.sac_hidden, space.low, space.high, cfg.n_quantiles)
        sac_agent.load_state_dict(ckpt["agent"])
        sac_agent.eval()
        return sac_agent, IdentityNormalizer(), cfg
    agent = Agent(ckpt["obs_dim"], ckpt["act_dim"], cfg.hidden, n_quantiles=cfg.n_quantiles)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()
    rms = RunningMeanStd(ckpt["obs_dim"])
    rms.load_state_dict(ckpt["rms"])
    return agent, rms, cfg


@torch.no_grad()
def rollout(agent: Agent, rms: RunningMeanStd, cfg: TrainConfig, nu: float, episodes: int,
            horizon: int = EPISODE_LENGTH, seed: int = EVAL_SEED) -> dict[str, np.ndarray]:
    """Runs ``episodes`` episodes at the fixed noise level ``nu`` with the deterministic policy.

    ``agent`` and ``rms`` are the first two results of ``load_policy``. Returns a dict with the arrays ``velocity``,
    ``reward``, ``cost``, ``dissimilarity``, ``bonus`` and ``pitch`` of shape (episodes, horizon); entries after the end
    of an episode are NaN.
    """
    envs = [make_env(nu=nu, seed=seed + 7919 * j, **env_settings(cfg)) for j in range(episodes)]
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
            rec["bonus"][j, k] = info["r_tD"]
            rec["pitch"][j, k] = info["pitch"]
            if terminated or truncated:
                alive[j] = False
    for e in envs:
        e.close()
    return rec


def evaluate_policy(path: str, noises: Iterable[float], episodes: int, out_path: str | None = None,
                    horizon: int = EPISODE_LENGTH) -> dict[str, np.ndarray]:
    """Evaluates the policy saved at ``path`` at every noise level in ``noises``.

    Returns (and, if ``out_path`` is given, saves as a compressed ``.npz``) a dict with the keys
    ``"nu{nu:.2f}/{name}"`` for ``name`` in ``RECORDED``. Noise levels already stored in an existing
    ``out_path`` are kept and not evaluated again, so adding a noise level to ``noises`` evaluates only
    that level. This reproduces a full evaluation because the rollouts at one noise level depend only on
    the seeds and on the number of episodes, which share one batched forward pass of the policy (the
    floating-point result of the batched pass depends on the batch size). A stored file whose arrays
    do not have the shape (episodes, horizon) is therefore discarded and every level is evaluated again.
    """
    result: dict[str, np.ndarray] = {}
    if out_path is not None and os.path.exists(out_path):
        with np.load(out_path) as z:
            result = {k: z[k] for k in z.files}
        if any(arr.shape != (episodes, horizon) for arr in result.values()):
            result = {}
    missing = [nu for nu in noises if f"nu{nu:.2f}/velocity" not in result]
    if not missing:
        return result
    agent, rms, cfg = load_policy(path)
    torch.set_num_threads(1)
    for nu in missing:
        rec = rollout(agent, rms, cfg, nu, episodes, horizon=horizon)
        for key, arr in rec.items():
            result[f"nu{nu:.2f}/{key}"] = arr
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        tmp = out_path[:-len(".npz")] + ".tmp.npz"
        np.savez_compressed(tmp, **result)
        os.replace(tmp, out_path)
    return result
