"""Evaluation of the threshold reward, of the returns, and of the evaluation reward from recorded arrays (analysis, Section 5)."""

import numpy as np
import pytest

from dmdp_sg.analysis import (_finite_fraction, dissimilarity_threshold, episode_returns, summarize_run,
                              threshold_reward)
from dmdp_sg.config import D_BONUS, D_THRESHOLD, V_LIMIT, TrainConfig, config_from_dict, make_config
from dmdp_sg.metrics import distance_to_target


def test_threshold_reward_from_the_recorded_dissimilarity():
    cfg = make_config("HalfCheetah", "DMDP", 0).to_dict()
    assert dissimilarity_threshold(cfg) == D_THRESHOLD == 0.1 and dissimilarity_threshold({}) == D_THRESHOLD
    rec = {"dissimilarity": np.array([[0.05, 0.10, 0.15, 3.0]])}
    np.testing.assert_array_equal(threshold_reward(rec, cfg), [[D_BONUS, D_BONUS, 0.0, 0.0]])
    np.testing.assert_array_equal(threshold_reward(rec, {"d_threshold": 0.07, "d_bonus": 1.0}, d=1.0), [[1.0, 0.0, 0.0, 0.0]])
    # a run without r_tD (a baseline) stored with the earlier D_thr = 0.14 is measured with the current threshold
    assert dissimilarity_threshold({"d_threshold": 0.14, "d_bonus": 0.0}) == D_THRESHOLD


@pytest.mark.parametrize("method", ["DMDP", "PPO"])
def test_summary_reports_the_task_return_and_the_threshold_reward_of_every_method(method):
    cfg = make_config("HalfCheetah", method, 0).to_dict()
    rng = np.random.default_rng(0)
    episodes, horizon = 8, 1000
    task = rng.normal(1.0, 0.5, (episodes, horizon))
    dissimilarity = rng.uniform(0.0, 0.3, (episodes, horizon))
    bonus = np.where(dissimilarity <= D_THRESHOLD, cfg["d_bonus"], 0.0)          # recorded r_tD (0 for PPO)
    rec = dict(velocity=rng.uniform(0.0, 3.29, (episodes, horizon)), reward=task + bonus,
               cost=np.zeros((episodes, horizon)), dissimilarity=dissimilarity, bonus=bonus)
    rows, _ = summarize_run({0.0: rec}, cfg)
    row = rows[0]
    # reported return: the return of the reward the method is trained on (r_t + r_tD for DMDP, r_t for PPO)
    np.testing.assert_allclose(episode_returns({0.0: rec}, 0.0), (task + bonus).sum(1))
    assert row["ret"] == pytest.approx((task + bonus).sum(1).mean())
    assert row["ret_benchmark"] == pytest.approx(task.sum(1).mean())        # no cost, so no penalty to add back
    assert row["within"] == pytest.approx((dissimilarity[:, 500:] <= D_THRESHOLD).mean())


@pytest.mark.parametrize("method, kappa", [("DMDP", 0.0), ("PPO", 0.0), ("DMDP-Lag", 2.0)])
def test_reported_and_benchmark_returns(method, kappa):
    # the recorded reward is r_t + r_tD with r_t = v_t - (kappa c_t)^2 - 0.1 ||a_t||^2 (kappa = 0, d = 10 for DMDP; 0 and 0
    # for PPO; kappa = 2 in the design pilot of DMDP-Lag): the reported return keeps it as it is, the benchmark return removes
    # r_tD and the cost penalty
    cfg = make_config("HalfCheetah", method, 0, cost_penalty=kappa).to_dict()
    assert cfg["d_bonus"] == (10.0 if method.startswith("DMDP") else 0.0)
    rng = np.random.default_rng(1)
    benchmark = rng.normal(2.0, 0.5, (4, 1000))
    cost = (rng.uniform(size=(4, 1000)) < 0.1).astype(float)
    bonus = np.where(rng.uniform(size=(4, 1000)) < 0.5, cfg["d_bonus"], 0.0)
    reward = benchmark - cfg["cost_penalty"] ** 2 * cost + bonus
    rec = dict(reward=reward, cost=cost, bonus=bonus)
    from dmdp_sg.analysis import benchmark_returns
    np.testing.assert_allclose(episode_returns({0.0: rec}, 0.0), reward.sum(1))
    np.testing.assert_allclose(benchmark_returns({0.0: rec}, 0.0, cfg), benchmark.sum(1))


@pytest.mark.parametrize("method, kappa", [("DMDP", 0.0), ("PPO", 0.0), ("DMDP-Lag", 2.0)])
def test_evaluation_reward_is_the_benchmark_reward_plus_r_tD_at_safe_steps_and_minus_v_limit_otherwise(method, kappa):
    # r^eva_t = r^b_t + r_tD if v_t <= v_limit (c_t = 0) and -v_limit if v_t > v_limit (c_t = 1), for every method: r_tD is
    # recomputed from the recorded dissimilarity (also for PPO, which is trained without it), and the cost penalty of a run of
    # the design pilot is added back
    from dmdp_sg.analysis import benchmark_returns, evaluation_penalty, evaluation_returns, evaluation_rewards
    cfg = make_config("HalfCheetah", method, 0, cost_penalty=kappa).to_dict()
    assert evaluation_penalty(cfg) == cfg["v_limit"] == V_LIMIT and evaluation_penalty({}) == V_LIMIT
    rng = np.random.default_rng(2)
    benchmark = rng.normal(2.0, 0.5, (4, 1000))
    cost = (rng.uniform(size=(4, 1000)) < 0.1).astype(float)
    dissimilarity = rng.uniform(0.0, 0.3, (4, 1000))
    within = dissimilarity <= D_THRESHOLD
    bonus = np.where(within, cfg["d_bonus"], 0.0)                               # recorded r_tD (0 for PPO)
    rec = dict(reward=benchmark - cfg["cost_penalty"] ** 2 * cost + bonus, cost=cost, bonus=bonus, dissimilarity=dissimilarity)
    r_td = np.where(within, D_BONUS, 0.0)
    expected = np.where(cost > 0.5, -V_LIMIT, benchmark + r_td)
    np.testing.assert_allclose(evaluation_rewards(rec, cfg), expected)
    np.testing.assert_allclose(evaluation_returns({0.0: rec}, 0.0, cfg), expected.sum(1))
    if method.startswith("DMDP"):          # the recomputed r_tD of a DMDP method is the recorded one
        np.testing.assert_allclose(evaluation_rewards(rec, cfg), np.where(cost > 0.5, -V_LIMIT, rec["reward"] + kappa ** 2 * cost))
    # an unsafe step lowers the evaluation return below the benchmark return by r^b_t + v_limit, a safe step within D_thr raises it by d
    np.testing.assert_allclose(benchmark_returns({0.0: rec}, 0.0, cfg) - evaluation_returns({0.0: rec}, 0.0, cfg),
                               (cost * (benchmark + V_LIMIT) - (1.0 - cost) * r_td).sum(1))


def test_evaluation_return_ignores_the_steps_after_the_end_of_an_episode_and_is_reported_in_the_summary():
    from dmdp_sg.analysis import evaluation_returns, evaluation_rewards
    cfg = make_config("HalfCheetah", "PPO", 0).to_dict()
    rng = np.random.default_rng(3)
    episodes, horizon = 8, 1000
    benchmark = rng.normal(2.0, 0.5, (episodes, horizon))
    cost = (rng.uniform(size=(episodes, horizon)) < 0.2).astype(float)
    dissimilarity = rng.uniform(0.0, 0.3, (episodes, horizon))
    benchmark[0, 600:], cost[0, 600:], dissimilarity[0, 600:] = np.nan, np.nan, np.nan     # the first episode ends after 600 steps
    rec = dict(velocity=rng.uniform(0.0, 4.0, (episodes, horizon)), reward=benchmark.copy(), cost=cost,
               dissimilarity=dissimilarity, bonus=np.zeros((episodes, horizon)))
    safe_reward = benchmark + np.where(dissimilarity <= D_THRESHOLD, D_BONUS, 0.0)
    expected = np.nansum(np.where(np.nan_to_num(cost, nan=0.0) > 0.5, -V_LIMIT, safe_reward), axis=1)
    np.testing.assert_allclose(evaluation_returns({0.0: rec}, 0.0, cfg), expected)
    assert np.isnan(evaluation_rewards(rec, cfg)[0, 600:]).all()
    row = summarize_run({0.0: rec}, cfg)[0][0]
    assert row["ret_eva"] == pytest.approx(expected.mean()) and row["ret_benchmark"] == pytest.approx(np.nansum(benchmark, 1).mean())


def test_pitch_summary_and_recorded_pitch(tmp_path):
    from dmdp_sg.analysis import run_dir
    from dmdp_sg.evaluate import load_policy, rollout
    from dmdp_sg.train import train
    cfg = make_config("HalfCheetah", "DMDP", 0).to_dict()
    rng = np.random.default_rng(1)
    pitch = np.concatenate([rng.uniform(-0.2, 0.2, (6, 1000)), np.full((2, 1000), 3.0)])     # 2 of 8 episodes flipped
    rec = dict(velocity=rng.uniform(2.25, 3.05, (8, 1000)), reward=np.ones((8, 1000)), cost=np.zeros((8, 1000)),
               dissimilarity=np.full((8, 1000), 0.05), bonus=np.zeros((8, 1000)), pitch=pitch)
    row = summarize_run({0.0: rec}, cfg)[0][0]
    assert row["flipped"] == pytest.approx(0.25) and row["pitch"] == pytest.approx(np.abs(pitch[:, 500:]).mean())
    assert row["D_target_pitch"] == pytest.approx(0.25 * 3.0, abs=0.02)     # W1 to U[-0.2, 0.2]: a quarter of the mass at 3
    assert np.isnan(summarize_run({0.0: {k: v for k, v in rec.items() if k != "pitch"}}, cfg)[0][0]["flipped"])
    # the evaluation records the wrapped true pitch
    c = make_config("HalfCheetah", "PPO", 0, total_steps=128, num_envs=2, num_steps=64)
    out = run_dir(str(tmp_path), c.env, c.method, c.seed)
    train(c, out, verbose=False)
    agent, rms, loaded = load_policy(f"{out}/policy.pt")
    r = rollout(agent, rms, loaded, 0.0, 2, horizon=20)
    assert r["pitch"].shape == (2, 20) and np.all(np.abs(r["pitch"]) <= np.pi) and np.all(np.abs(r["pitch"][:, 0]) < 0.3)


def _upright_record(rng, episodes=8, horizon=1000):
    return dict(velocity=rng.uniform(2.25, 3.05, (episodes, horizon)), reward=np.ones((episodes, horizon)),
                cost=np.zeros((episodes, horizon)), dissimilarity=np.full((episodes, horizon), 0.05),
                bonus=np.zeros((episodes, horizon)), pitch=rng.uniform(-0.2, 0.2, (episodes, horizon)))


def test_fractions_ignore_the_steps_after_a_termination():
    cfg = make_config("HalfCheetah", "DMDP", 0).to_dict()
    rec = _upright_record(np.random.default_rng(2))
    rec["pitch"][:2] = 3.0                                      # episodes 0 and 1 are flipped throughout
    for name in rec:
        rec[name][:4, 700:] = np.nan                            # episodes 0-3 end at step 700, the others run to 1000
    row = summarize_run({0.0: rec}, cfg)[0][0]
    recorded = 4 * 200 + 4 * 500                                # steps k >= 500 that were recorded
    assert row["within"] == 1.0                                 # every recorded step is within D_thr
    assert row["flipped"] == pytest.approx(2 * 200 / recorded)
    assert row["pitch"] == pytest.approx(np.nanmean(np.abs(rec["pitch"][:, 500:])))
    rec["dissimilarity"][4:, 600:700] = 1.0                     # 400 of the recorded steps are not within D_thr
    assert summarize_run({0.0: rec}, cfg)[0][0]["within"] == pytest.approx(1.0 - 400 / recorded)
    assert np.isnan(_finite_fraction(np.full((3, 4), np.nan), lambda x: x > 0.0))


def test_fractions_without_termination_are_plain_means():
    cfg = make_config("HalfCheetah", "DMDP", 0).to_dict()
    rec = _upright_record(np.random.default_rng(3))
    rec["pitch"][:, 750:] = 3.0
    rec["dissimilarity"][:, 900:] = 1.0
    row = summarize_run({0.0: rec}, cfg)[0][0]
    assert row["flipped"] == 0.5 and row["within"] == 0.8


def test_missing_target_family_is_read_as_in_the_config():
    cfg = make_config("HalfCheetah", "DMDP", 0).to_dict()
    del cfg["target_family"]                                    # a stored run without the entry
    family = config_from_dict(cfg).target_family                # the family with which the run is loaded
    assert family == TrainConfig().target_family != "gaussian"
    rec = _upright_record(np.random.default_rng(4))
    row = summarize_run({0.0: rec}, cfg)[0][0]
    mu, sd = cfg["mu_star"], cfg["sd_star"]
    assert row["D_target"] == distance_to_target(rec["velocity"], mu, sd, k0=500, family=family)
    assert row["D_target"] != distance_to_target(rec["velocity"], mu, sd, k0=500, family="gaussian")
    cfg["target_family"] = "gaussian"                           # a stored entry is used as it is
    assert summarize_run({0.0: rec}, cfg)[0][0]["D_target"] == distance_to_target(rec["velocity"], mu, sd, k0=500)
