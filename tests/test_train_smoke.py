"""Smoke test of training and evaluation: one PPO update with 2 environments x 64 steps."""

import json
import os

import numpy as np
import pytest
import torch

from dmdp_sg.analysis import load_episodes, load_evaluation, run_dir
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


def test_multiplier_stays_at_zero_before_the_first_episode_while_the_adam_moments_fill():
    # no episode completed: J_c_hat = 0, the gradient c_bar > 0 pushes lambda below 0, and the projection keeps it at 0
    lam = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([lam], lr=0.035)
    for k in range(1, 4):
        _update_lagrange_multiplier(lam, opt, episodic_cost=0.0, cost_limit=5.0)
        assert lam.item() == 0.0
        assert opt.state[lam]["exp_avg"].item() == pytest.approx(5.0 * (1.0 - 0.9 ** k))


def test_config_is_written_before_the_policy(tmp_path, monkeypatch):
    # run_all.py takes an existing policy.pt for a finished run, so policy.pt must be the last file written
    from dmdp_sg.agent import Agent, RunningMeanStd
    from dmdp_sg.train import _save_run
    present, save = [], torch.save
    monkeypatch.setattr(torch, "save", lambda obj, path: (present.append(sorted(os.listdir(tmp_path))), save(obj, path))[1])
    _save_run(str(tmp_path), Agent(3, 1), RunningMeanStd(3), make_config("HalfCheetah", "PPO", 0), 3, 1, [], [])
    assert present == [["config.json", "episodes.csv", "training.csv"]] and (tmp_path / "policy.pt").exists()


@pytest.mark.parametrize("method", ["DMDP", "PPO", "Dist-PPO", "Implicit-PPO", "Dist-PPO-Lag", "Implicit-PPO-Lag"])
def test_training_smoke(tmp_path, method):
    cfg = make_config("HalfCheetah", method, 0, total_steps=128, num_envs=2, num_steps=64)
    out = run_dir(str(tmp_path), cfg.env, cfg.method, cfg.seed)
    history = train(cfg, out, verbose=False)
    assert history == []                                     # no episode of H = 1000 steps is completed

    for name in ("policy.pt", "training.csv", "episodes.csv", "config.json"):
        assert (tmp_path / cfg.env / cfg.method / "seed0" / name).exists()
    with open(f"{out}/config.json") as f:
        assert json.load(f) == cfg.to_dict()

    agent, rms, loaded = load_policy(f"{out}/policy.pt")
    assert loaded == cfg and rms.mean.shape == (21 if method == "DMDP" or method.startswith("Implicit") else 17,)   # omega_hat_t, o_t
    assert agent.critic[-1].out_features == (32 if method.startswith(("Dist-", "Implicit-")) else 1)   # distributional critic: 32 quantiles

    ev = evaluate_policy(f"{out}/policy.pt", [0.0, 0.2], episodes=2, out_path=f"{out}/evaluation.npz", horizon=5)
    assert sorted(ev) == sorted(f"nu{nu:.2f}/{k}" for nu in (0.0, 0.2)
                                for k in ("velocity", "reward", "cost", "dissimilarity", "bonus", "pitch"))
    stored = load_evaluation(str(tmp_path), cfg.env, cfg.method, cfg.seed)
    assert set(stored) == {0.0, 0.2}
    assert stored[0.2]["velocity"].shape == (2, 5) and np.all(np.isfinite(stored[0.2]["velocity"]))


def test_return_scaler_divides_by_the_std_of_the_discounted_return():
    from dmdp_sg.train import ReturnScaler
    rng = np.random.default_rng(0)
    scaler, returns, seen, scaled = ReturnScaler(3, gamma=0.9), np.zeros(3), [], None
    for t in range(200):
        reward, done = rng.normal(-5.0, 2.0, 3), np.array([False, t % 50 == 49, False])
        returns = 0.9 * returns + reward
        seen.append(returns.copy())
        scaled = scaler(reward, done)
        returns[done] = 0.0                              # the discounted return restarts with a new episode
        np.testing.assert_allclose(scaled, np.clip(reward / np.sqrt(np.var(np.concatenate(seen)) + 1e-8), -10, 10), rtol=1e-3)
    off = ReturnScaler(3, gamma=0.9, enabled=False)
    assert off(reward, done) is reward


def test_combined_advantage_does_not_depend_on_the_reward_scale():
    from dmdp_sg.train import combine_advantages
    torch.manual_seed(0)
    adv_r, adv_c = torch.randn(16, 4) * 3.0 + 1.0, torch.rand(16, 4)
    a = combine_advantages(adv_r, adv_c, 2.0)
    assert torch.allclose(a, combine_advantages(100.0 * adv_r + 7.0, adv_c + 0.3, 2.0), atol=1e-5)   # scale and offsets drop out
    expected = ((adv_r - adv_r.mean()) / adv_r.std() - 2.0 * (adv_c - adv_c.mean())) / 3.0
    assert torch.allclose(a, expected, atol=1e-5)
    unconstrained = combine_advantages(adv_r, adv_c, 0.0)                     # lambda = 0: the standardized reward advantage
    assert torch.allclose(unconstrained, (adv_r - adv_r.mean()) / adv_r.std(), atol=1e-5)


def test_factor_of_the_combined_advantage_cancels_in_the_minibatch_standardization():
    # _ppo_update standardizes the combined advantage again, so 1 / (1 + lam) does not change the policy step
    from dmdp_sg.train import combine_advantages
    torch.manual_seed(0)
    adv_r, adv_c = torch.randn(16, 4), torch.rand(16, 4)

    def standardize(x):
        return (x - x.mean()) / (x.std() + 1e-8)
    for lam in (0.0, 1.0, 5.0):
        without_factor = (adv_r - adv_r.mean()) / adv_r.std() - lam * (adv_c - adv_c.mean())
        torch.testing.assert_close(standardize(combine_advantages(adv_r, adv_c, lam)), standardize(without_factor),
                                   rtol=1e-4, atol=1e-5)


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


def test_episode_log(tmp_path):
    # 2 environments x 1024 steps: each environment completes one episode of H = 1000 steps
    cfg = make_config("HalfCheetah", "PPO", 0, total_steps=2048, num_envs=2, num_steps=1024)
    out = run_dir(str(tmp_path), cfg.env, cfg.method, cfg.seed)
    history = train(cfg, out, verbose=False)
    episodes = load_episodes(str(tmp_path), cfg.env, cfg.method, cfg.seed)
    assert list(episodes.columns) == ["step", "ret", "task_ret", "cost", "dis", "dis_median", "within", "pitch", "flipped"]
    assert len(episodes) == 2 and (episodes.step == 2048).all()
    assert history[-1]["ep_flipped"] == pytest.approx(episodes.flipped.mean()) and (episodes.flipped <= 1.0).all()
    assert history[-1]["ep_return"] == pytest.approx(episodes.ret.mean())
    assert history[-1]["ep_task_return"] == pytest.approx(episodes.task_ret.mean())
    np.testing.assert_allclose(episodes.ret, episodes.task_ret)          # PPO: no threshold reward
    assert history[-1]["ep_cost"] == pytest.approx(episodes.cost.mean())


def test_episode_log_separates_the_threshold_reward(tmp_path):
    # with an infinite threshold every step is within, so r_tD = +10 at every step of DMDP
    cfg = make_config("HalfCheetah", "DMDP", 0, total_steps=2048, num_envs=2, num_steps=1024, d_threshold=1e9)
    out = run_dir(str(tmp_path), cfg.env, cfg.method, cfg.seed)
    history = train(cfg, out, verbose=False)
    episodes = load_episodes(str(tmp_path), cfg.env, cfg.method, cfg.seed)
    assert (episodes.within == 1.0).all()
    np.testing.assert_allclose(episodes.ret - episodes.task_ret, 10.0 * 1000)
    assert history[-1]["ep_task_return"] == pytest.approx(episodes.task_ret.mean())
    assert history[-1]["ep_return"] == pytest.approx(episodes.ret.mean())


def test_evaluation_uses_the_configured_cost_penalty(tmp_path):
    from dmdp_sg.evaluate import rollout
    cfg = make_config("HalfCheetah", "PPO", 0, total_steps=128, num_envs=2, num_steps=64, cost_penalty=2.0)   # as in the pilot
    out = run_dir(str(tmp_path), cfg.env, cfg.method, cfg.seed)
    train(cfg, out, verbose=False)
    agent, rms, loaded = load_policy(f"{out}/policy.pt")
    rec0 = rollout(agent, rms, loaded, 0.0, 2, horizon=50)
    loaded.cost_penalty = 0.0
    rec1 = rollout(agent, rms, loaded, 0.0, 2, horizon=50)
    np.testing.assert_allclose(rec0["velocity"], rec1["velocity"])          # the penalty does not change the dynamics
    np.testing.assert_allclose(rec0["reward"], rec1["reward"] - 4.0 * rec1["cost"])


def test_legacy_config_without_cost_penalty_is_read_as_unpenalized():
    from dmdp_sg.config import config_from_dict
    stored = make_config("HalfCheetah", "DMDP-Lag", 0, cost_penalty=2.0).to_dict()     # a run of the design pilot
    del stored["cost_penalty"]
    assert config_from_dict(stored).cost_penalty == 0.0
    assert config_from_dict(stored) != make_config("HalfCheetah", "DMDP-Lag", 0, cost_penalty=2.0)


def test_shaped_reward_subtracts_beta_times_the_dissimilarity(monkeypatch):
    # training reward r_t + r_tD - beta * D (here with beta = 2 for DMDP); r_t for PPO
    import dmdp_sg.train as tr
    from dmdp_sg.agent import Agent, RunningMeanStd
    from dmdp_sg.envs import make_vector_env
    for method, beta, d_threshold in (("DMDP", 2.0, 0.14), ("PPO", 0.0, 0.14), ("DMDP-Lag", 0.5, 1e9)):
        cfg = make_config("HalfCheetah", method, 0, num_envs=2, num_steps=8, beta=beta, d_threshold=d_threshold,
                          reward_normalization=False)
        envs = make_vector_env(cfg, nu=0.0)
        obs, _ = envs.reset(seed=[0, 1])
        rms = RunningMeanStd(obs.shape[1])
        agent = Agent(obs.shape[1], 6)
        buf = tr.RolloutBuffer.zeros(8, 2, obs.shape[1], 6)
        rewards, dis, r_td = [], [], []
        step = envs.step
        def spy(action):
            out = step(action)
            rewards.append(out[1].copy())
            dis.append(tr._info_at_step(out[4], "dissimilarity", 2))
            r_td.append(tr._info_at_step(out[4], "r_tD", 2))
            return out
        monkeypatch.setattr(envs, "step", spy)
        tr._collect_rollout(envs, agent, rms, cfg, buf, torch.as_tensor(rms.normalize(obs)), torch.zeros(2), [])
        expected = np.array(rewards) - beta * np.array(dis)
        np.testing.assert_allclose(buf.rewards.numpy(), expected, rtol=1e-5, atol=1e-5)
        if d_threshold > 1:                     # every window is within the infinite threshold of this DMDP-Lag case
            np.testing.assert_array_equal(np.array(r_td), 10.0)
        if method == "PPO":                     # d = 0
            np.testing.assert_array_equal(np.array(r_td), 0.0)
        envs.close()


def test_rollout_stores_the_scaled_training_reward(monkeypatch):
    import dmdp_sg.train as tr
    from dmdp_sg.agent import Agent, RunningMeanStd
    from dmdp_sg.envs import make_vector_env
    # beta = 1 gives shaped rewards large enough that the first steps are scaled down
    cfg = make_config("HalfCheetah", "DMDP", 0, num_envs=2, num_steps=8, beta=1.0)
    assert cfg.reward_normalization
    envs = make_vector_env(cfg, nu=0.0)
    obs, _ = envs.reset(seed=[0, 1])
    rms, agent = RunningMeanStd(obs.shape[1]), Agent(obs.shape[1], 6)
    buf = tr.RolloutBuffer.zeros(8, 2, obs.shape[1], 6)
    shaped, step = [], envs.step
    def spy(action):
        out = step(action)
        shaped.append(out[1] - cfg.beta * tr._info_at_step(out[4], "dissimilarity", 2))
        return out
    monkeypatch.setattr(envs, "step", spy)
    tr._collect_rollout(envs, agent, rms, cfg, buf, torch.as_tensor(rms.normalize(obs)), torch.zeros(2), [],
                        tr.ReturnScaler(2, cfg.gamma))
    reference = tr.ReturnScaler(2, cfg.gamma)
    expected = np.array([reference(r, np.zeros(2, bool)) for r in shaped])
    np.testing.assert_allclose(buf.rewards.numpy(), expected, rtol=1e-5, atol=1e-6)
    assert np.abs(expected).max() < np.abs(np.array(shaped)).max()            # the reward was scaled down
    envs.close()


def test_training_uses_the_multiplier_and_the_reward_scaler(tmp_path, monkeypatch):
    import dmdp_sg.train as tr
    seen = {"lam": [], "scaler": []}
    combine, collect = tr.combine_advantages, tr._collect_rollout
    monkeypatch.setattr(tr, "combine_advantages", lambda a, c, lam: (seen["lam"].append(lam), combine(a, c, lam))[1])
    monkeypatch.setattr(tr, "_collect_rollout", lambda *args: (seen["scaler"].append(args[-1].enabled), collect(*args))[1])
    # cost limit below every attainable cost: the multiplier must grow as soon as an episode is completed
    cfg = make_config("HalfCheetah", "DMDP-Lag", 0, total_steps=4096, num_envs=2, num_steps=1024, cost_limit=-1.0)
    history = train(cfg, run_dir(str(tmp_path), cfg.env, cfg.method, cfg.seed), verbose=False)
    assert seen["lam"][-1] > 0.0 and history[-1]["lagrange_multiplier"] == pytest.approx(seen["lam"][-1])
    assert seen["scaler"] == [True, True]
    seen["lam"].clear()
    cfg = make_config("HalfCheetah", "DMDP", 0, total_steps=2048, num_envs=2, num_steps=1024, reward_normalization=False)
    train(cfg, run_dir(str(tmp_path), "x", cfg.method, cfg.seed), verbose=False)
    assert seen["lam"] == [0.0] and seen["scaler"][-1] is False            # no constraint: lambda = 0
