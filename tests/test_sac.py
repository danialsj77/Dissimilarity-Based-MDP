"""Tests of the SAC learner (sac.py): configuration, buffer, reward scaling, truncation, Lagrangian update and a smoke run."""

import copy
import json
import os

import numpy as np
import pytest
import torch

from dmdp_sg.analysis import load_episodes, load_evaluation, load_training, run_dir
from dmdp_sg.config import SAC_NUM_ENVS, SAC_TOTAL_STEPS, make_config
from dmdp_sg.evaluate import evaluate_policy, load_policy
from dmdp_sg.sac import (IdentityNormalizer, ReplayBuffer, ReturnStatistics, SACAgent, _next_observations, sac_update,
                         train_sac)


def test_config_of_the_sac_learner():
    cfg = make_config("HalfCheetah", "SAC-Lag", 0)
    assert cfg.learner == "sac" and cfg.lagrangian and cfg.beta == 0.0 and cfg.d_bonus == 0.0
    assert cfg.total_steps == SAC_TOTAL_STEPS and cfg.num_envs == SAC_NUM_ENVS
    dmdp = make_config("HalfCheetah", "DMDP-Lag", 0, learner="sac")
    assert dmdp.learner == "sac" and dmdp.beta > 0 and dmdp.policy_input == "hybrid"
    assert make_config("HalfCheetah", "DMDP", 0).learner == "ppo"                      # the default learner is PPO
    assert make_config("HalfCheetah", "SAC", 0, total_steps=1000).total_steps == 1000   # overrides still apply
    for method, learner in (("SAC", "ppo"), ("PPO", "sac"), ("PPO-Lag", "sac")):
        with pytest.raises(ValueError):
            make_config("HalfCheetah", method, 0, learner=learner)


def test_replay_buffer_wraps_around():
    buf = ReplayBuffer(3, obs_dim=2, act_dim=1, seed=0)
    for t in range(5):
        buf.add(np.full((1, 2), t), np.zeros((1, 1)), [t], [0.0], [0.0], np.full((1, 2), t + 1))
    assert buf.size == 3 and sorted(buf.reward.tolist()) == [2.0, 3.0, 4.0]
    batch = buf.sample(8)
    assert batch["obs"].shape == (8, 2) and set(batch["reward"].tolist()) <= {2.0, 3.0, 4.0}
    assert torch.all(batch["next_obs"] == batch["obs"] + 1)


def test_next_observation_of_a_finished_episode_is_its_final_observation():
    next_obs = np.zeros((3, 2), np.float32)                     # reset observations of the vector environment
    infos = {"final_observation": np.array([None, np.ones(2), None], dtype=object)}
    real = _next_observations(next_obs, infos, np.array([False, True, False]))
    assert real[1].tolist() == [1.0, 1.0] and real[0].tolist() == [0.0, 0.0] and next_obs[1].tolist() == [0.0, 0.0]


def test_return_statistics_scale_by_the_std_of_the_discounted_return():
    stats = ReturnStatistics(2, gamma=0.9)
    rng = np.random.default_rng(0)
    returns, seen = np.zeros(2), []
    for t in range(200):
        r, done = rng.normal(3.0, 2.0, 2), np.array([t % 50 == 49, False])
        stats.update(r, done)
        returns = 0.9 * returns + r
        seen.extend(returns)
        returns[done] = 0.0
    scaled = stats.scale(torch.tensor([1.0, 1e6]))
    assert scaled[0].item() == pytest.approx(1.0 / np.std(seen), rel=1e-3) and scaled[1].item() == 10.0
    assert ReturnStatistics(1, 0.9, enabled=False).scale(torch.tensor([5.0])).item() == 5.0


def _batch(n=32, obs_dim=5, act_dim=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(obs=torch.randn(n, obs_dim, generator=g), act=torch.rand(n, act_dim, generator=g) * 2 - 1,
                reward=torch.randn(n, generator=g), cost=(torch.rand(n, generator=g) < 0.3).float(),
                terminated=torch.zeros(n), next_obs=torch.randn(n, obs_dim, generator=g))


def _optimizers(agent):
    return dict(q=torch.optim.Adam(list(agent.q1.parameters()) + list(agent.q2.parameters()), lr=1e-3),
                qc=torch.optim.Adam(agent.qc.parameters(), lr=1e-3),
                actor=torch.optim.Adam(agent.actor.parameters(), lr=1e-3),
                alpha=torch.optim.Adam([agent.log_alpha], lr=1e-3))


@pytest.mark.parametrize("lagrangian", [False, True])
def test_cost_critic_is_trained_only_with_the_constraint(lagrangian):
    torch.manual_seed(0)
    agent = SACAgent(5, 2, 16, -np.ones(2), np.ones(2))
    qc_before = copy.deepcopy(agent.qc.state_dict())
    cfg = make_config("HalfCheetah", "SAC-Lag" if lagrangian else "SAC", 0)
    sac_update(agent, _optimizers(agent), _batch(), ReturnStatistics(1, 0.99), 1.0, cfg, target_entropy=-2.0)
    changed = any(not torch.equal(qc_before[k], v) for k, v in agent.qc.state_dict().items())
    assert changed == lagrangian


def test_lagrange_multiplier_enters_the_actor_loss():
    cfg = make_config("HalfCheetah", "SAC-Lag", 0)
    actors = []
    for lam in (0.0, 10.0):
        torch.manual_seed(0)
        agent = SACAgent(5, 2, 16, -np.ones(2), np.ones(2))
        torch.manual_seed(1)
        sac_update(agent, _optimizers(agent), _batch(), ReturnStatistics(1, 0.99), lam, cfg, target_entropy=-2.0)
        actors.append(agent.actor.state_dict())
    assert any(not torch.equal(actors[0][k], actors[1][k]) for k in actors[0])


def test_actions_respect_the_bounds():
    agent = SACAgent(3, 2, 8, np.array([-1.0, 0.0]), np.array([1.0, 2.0]))
    a, _, _ = agent.act(torch.randn(100, 3) * 10)
    assert torch.all(a[:, 0].abs() <= 1.0) and torch.all((a[:, 1] >= 0.0) & (a[:, 1] <= 2.0))
    mean, _, _ = agent.act(torch.zeros(1, 3), deterministic=True)
    assert mean.shape == (1, 2)
    assert IdentityNormalizer().normalize(np.array([[12.5]])).dtype == np.float32


@pytest.mark.parametrize("method", ["DMDP-Lag", "SAC", "Dist-SAC", "Implicit-SAC", "Dist-SAC-Lag", "Implicit-SAC-Lag"])
def test_sac_training_smoke(tmp_path, method):
    # two environments for 1050 steps each: one episode of H = 1000 steps per environment is completed
    cfg = make_config("HalfCheetah", method, 0, learner="sac", total_steps=2100, num_envs=2, sac_learning_starts=1900,
                      sac_batch_size=16, sac_hidden=16, sac_lambda_interval=500)
    out = run_dir(str(tmp_path), cfg.env, cfg.method, cfg.seed)
    history = train_sac(cfg, out, verbose=False)
    assert len(history) == 1 and history[0]["step"] == 2000 and history[0]["ep_length"] == 1000
    assert {"alpha", "lagrange_multiplier", "ep_task_return"} <= set(history[0])
    assert len(load_episodes(str(tmp_path), cfg.env, cfg.method, cfg.seed)) == 2
    assert len(load_training(str(tmp_path), cfg.env, cfg.method, cfg.seed)) == 1
    with open(f"{out}/config.json") as f:
        assert json.load(f) == cfg.to_dict()

    agent, norm, loaded = load_policy(f"{out}/policy.pt")
    assert loaded == cfg and isinstance(agent, SACAgent) and isinstance(norm, IdentityNormalizer)
    assert agent.q1(torch.zeros(3, agent.actor.trunk[0].in_features), torch.zeros(3, 6)).shape == \
        ((3, 32) if method.startswith(("Dist-", "Implicit-")) else (3,))                  # distributional critics: 32 quantiles
    ev = evaluate_policy(f"{out}/policy.pt", [0.0, 0.2], episodes=2, out_path=f"{out}/evaluation.npz", horizon=5)
    assert len(ev) == 12
    stored = load_evaluation(str(tmp_path), cfg.env, cfg.method, cfg.seed)
    assert stored[0.2]["velocity"].shape == (2, 5) and np.all(np.isfinite(stored[0.0]["dissimilarity"]))


def test_sac_multiplier_uses_lagrangian_lr(tmp_path, monkeypatch):
    import dmdp_sg.sac as sac
    rates, update = [], sac._update_lagrange_multiplier

    def spy(multiplier, optimizer, episodic_cost, cost_limit):
        rates.append(optimizer.param_groups[0]["lr"])
        update(multiplier, optimizer, episodic_cost, cost_limit)
    monkeypatch.setattr(sac, "_update_lagrange_multiplier", spy)
    # random actions only (no gradient steps); the multiplier is updated at 500 and 1000 steps
    cfg = make_config("HalfCheetah", "SAC-Lag", 0, total_steps=1200, num_envs=2, sac_learning_starts=10 ** 9,
                      sac_lambda_interval=500, lagrangian_lr=0.005)
    train_sac(cfg, run_dir(str(tmp_path), cfg.env, cfg.method, cfg.seed), verbose=False)
    assert rates == [0.005, 0.005]


def test_config_is_written_before_the_policy(tmp_path, monkeypatch):
    # run_all.py takes an existing policy.pt for a finished run, so policy.pt must be the last file written
    from dmdp_sg.sac import _save_run
    present, save = [], torch.save
    monkeypatch.setattr(torch, "save", lambda obj, path: (present.append(sorted(os.listdir(tmp_path))), save(obj, path))[1])
    _save_run(str(tmp_path), SACAgent(3, 1, 8, -np.ones(1), np.ones(1)), make_config("HalfCheetah", "SAC", 0), 3, 1, [], [])
    assert present == [["config.json", "episodes.csv", "training.csv"]] and (tmp_path / "policy.pt").exists()


def test_stored_configs_with_the_piloted_multiplier_rate():
    from dmdp_sg.config import config_from_dict
    unconstrained = make_config("HalfCheetah", "DMDP", 0, learner="sac").to_dict()
    assert config_from_dict({**unconstrained, "sac_lagrangian_lr": 0.005}) == make_config("HalfCheetah", "DMDP", 0, learner="sac")
    constrained = make_config("HalfCheetah", "SAC-Lag", 0).to_dict()
    assert config_from_dict({**constrained, "sac_lagrangian_lr": constrained["lagrangian_lr"]}).lagrangian
    with pytest.raises(ValueError):
        config_from_dict({**constrained, "sac_lagrangian_lr": 0.005})
