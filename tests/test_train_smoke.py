"""Smoke test of training and evaluation: one PPO update with 2 environments x 64 steps."""

import json

import numpy as np
import pytest
import torch

from dmdp_sg.analysis import load_evaluation, run_dir
from dmdp_sg.config import make_config
from dmdp_sg.evaluate import evaluate_policy, load_policy
from dmdp_sg.train import _update_lagrange_multiplier, compute_gae, train


def test_compute_gae_without_discounting():
    rewards = torch.ones(4, 1)
    values = torch.zeros(4, 1)
    dones = torch.zeros(4, 1)
    adv, ret = compute_gae(rewards, values, dones, torch.zeros(1), torch.zeros(1), gamma=1.0, lam=1.0)
    assert adv[:, 0].tolist() == [4.0, 3.0, 2.0, 1.0]
    assert torch.equal(ret, adv)


def test_lagrange_multiplier_ascends_and_is_projected():
    lam = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([lam], lr=0.1)
    _update_lagrange_multiplier(lam, opt, episodic_cost=40.0, cost_limit=25.0)    # constraint violated
    assert lam.item() > 0.0
    for _ in range(5):
        _update_lagrange_multiplier(lam, opt, episodic_cost=0.0, cost_limit=25.0)  # constraint satisfied
    assert lam.item() == 0.0


@pytest.mark.parametrize("method", ["DMDP", "PPO"])
def test_training_smoke(tmp_path, method):
    cfg = make_config("HalfCheetah", method, 0, total_steps=128, num_envs=2, num_steps=64)
    out = run_dir(str(tmp_path), cfg.env, cfg.method, cfg.seed)
    history = train(cfg, out, verbose=False)
    assert history == []                                     # no episode of H = 1000 steps is completed

    for name in ("policy.pt", "training.csv", "config.json"):
        assert (tmp_path / cfg.env / cfg.method / "seed0" / name).exists()
    with open(f"{out}/config.json") as f:
        assert json.load(f) == cfg.to_dict()

    agent, rms, loaded = load_policy(f"{out}/policy.pt")
    assert loaded == cfg and rms.mean.shape == (19 if method == "DMDP" else 17,)

    ev = evaluate_policy(f"{out}/policy.pt", [0.0, 0.2], episodes=2, out_path=f"{out}/evaluation.npz", horizon=5)
    assert sorted(ev) == sorted(f"nu{nu:.2f}/{k}" for nu in (0.0, 0.2)
                                for k in ("velocity", "reward", "cost", "dissimilarity"))
    stored = load_evaluation(str(tmp_path), cfg.env, cfg.method, cfg.seed)
    assert set(stored) == {0.0, 0.2}
    assert stored[0.2]["velocity"].shape == (2, 5) and np.all(np.isfinite(stored[0.2]["velocity"]))


def test_truncated_episode_is_bootstrapped():
    from dmdp_sg.agent import Agent, RunningMeanStd
    from dmdp_sg.train import _bootstrap_truncated
    torch.manual_seed(0)
    agent, rms = Agent(3, 1), RunningMeanStd(3)
    final = np.array([[0.5, -1.0, 2.0], [1.0, 1.0, 1.0]])
    infos = {"final_observation": np.array([final[0], final[1], None], dtype=object)}
    rewards, costs = torch.ones(3), torch.zeros(3)
    terminated, truncated = np.array([False, True, False]), np.array([True, True, False])
    _bootstrap_truncated(rewards, costs, agent, rms, infos, terminated, truncated, gamma=0.9)
    x = torch.as_tensor(rms.normalize(final[:1]))
    with torch.no_grad():
        assert torch.allclose(rewards[0], 1.0 + 0.9 * agent.value(x)[0])
        assert torch.allclose(costs[0], 0.9 * agent.cost_value(x)[0])
    assert rewards[1:].tolist() == [1.0, 1.0] and costs[1:].tolist() == [0.0, 0.0]   # terminated or running
