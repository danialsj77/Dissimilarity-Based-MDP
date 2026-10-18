"""Environment wrappers on a real HalfCheetah environment (Sections 6.1-6.3)."""

import gymnasium as gym
import numpy as np
import pytest

from dmdp_sg.config import V_LIMIT, make_config
from dmdp_sg.envs import gaussian_w2_squared, make_env
from dmdp_sg.targets import Target, empirical_w2_squared

CFG = make_config("HalfCheetah", "DMDP", 0)
TARGET = Target(CFG.target_family, CFG.mu_star, CFG.sd_star)      # velocity target U[0.70 v_lim, 0.95 v_lim]
PITCH_TARGET = Target("uniform", CFG.pitch_mu_star, CFG.pitch_sd_star)   # pitch target U[-0.2, 0.2]
WIDE_TARGET = Target("uniform", 1.646, 0.9503)                    # U[0, 3.29], the target of the earlier designs
OBS_DIM = 17


def _make(policy_input, nu=0.0, window=10, seed=0):
    return make_env(CFG.env_id, nu, seed, window, policy_input, CFG.mu_star, CFG.sd_star, CFG.vel_index)


def test_velocity_threshold_of_the_simulator_is_the_configured_one():
    # the simulator produces the cost c_t = 1[v_t > threshold]; the configured v_limit must be the same number
    env = _make("state")
    assert env.unwrapped._velocity_threshold == CFG.v_limit == V_LIMIT
    env.close()
    raw = gym.make(CFG.env_id)
    assert raw.unwrapped._velocity_threshold == CFG.v_limit
    raw.close()


def test_gaussian_w2_squared():
    assert gaussian_w2_squared(2.0, 0.5, 2.0, 0.5) == 0.0
    assert gaussian_w2_squared(1.0, 0.2, 2.0, 0.5) == pytest.approx(1.0 + 0.09)


def test_velocity_only_hybrid_input_has_two_window_statistics():
    env = make_env(CFG.env_id, 0.0, 0, 10, "hybrid", CFG.mu_star, CFG.sd_star, CFG.vel_index, pitch_weight=0.0)
    assert env.observation_space.shape == (OBS_DIM + 2,)
    env.reset(seed=0)
    _, _, _, _, info = env.step(env.action_space.sample())
    assert info["dissimilarity_pitch"] == 0.0 and info["dissimilarity"] == info["dissimilarity_velocity"]
    env.close()


@pytest.mark.parametrize("policy_input, dim", [("state", OBS_DIM), ("hybrid", OBS_DIM + 4)])
def test_feature_dimensions(policy_input, dim):
    env = _make(policy_input)
    assert env.observation_space.shape == (dim,)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (dim,) and obs.dtype == np.float32
    obs, *_ = env.step(env.action_space.sample())
    assert obs.shape == (dim,) and obs.dtype == np.float32
    env.close()


def test_noise_free_state_input_equals_benchmark_observation():
    env = _make("state", nu=0.0)
    raw = gym.make(CFG.env_id)
    obs, _ = env.reset(seed=4)
    raw_obs, _ = raw.reset(seed=4)
    np.testing.assert_array_equal(obs, raw_obs.astype(np.float32))
    action = env.action_space.sample()
    np.testing.assert_array_equal(env.step(action)[0], raw.step(action)[0].astype(np.float32))
    env.close()
    raw.close()


def test_hybrid_window_statistics_and_dissimilarity():
    window = 5
    env = _make("hybrid", nu=0.2, window=window, seed=1)
    obs, _ = env.reset(seed=1)
    # o_t follows omega_hat_t = (window mean and std of the velocity readings, window mean and std of the pitch readings)
    readings = [float(obs[4 + CFG.vel_index])]
    pitch_readings = [float(obs[4 + CFG.pitch_index])]
    # at reset, the windows are filled with the first reading: omega_hat_0 = (o^v_0, 0, o^theta_0, 0)
    assert obs[0] == pytest.approx(readings[0], rel=1e-6) and obs[1] == pytest.approx(0.0, abs=1e-6)
    assert obs[2] == pytest.approx(pitch_readings[0], rel=1e-6) and obs[3] == pytest.approx(0.0, abs=1e-6)
    velocities, pitches = [], []
    for _ in range(8):
        obs, _, _, _, info = env.step(env.action_space.sample())
        readings.append(float(obs[4 + CFG.vel_index]))
        pitch_readings.append(float(obs[4 + CFG.pitch_index]))
        velocities.append(info["x_velocity"])
        pitches.append(info["pitch"])
        assert info["pitch"] == pytest.approx(float(env.unwrapped.data.qpos[CFG.pitch_index + 1]))   # |pitch| < pi here
        for k, r in ((0, readings), (2, pitch_readings)):
            w = np.array(([r[0]] * (window - 1) + r)[-window:])
            assert obs[k] == pytest.approx(w.mean(), rel=1e-5, abs=1e-6)
            assert obs[k + 1] == pytest.approx(w.std(), rel=1e-5, abs=1e-6)
        v, th = np.array(velocities[-window:]), np.array(pitches[-window:])   # true values, available steps only
        assert info["dissimilarity_velocity"] == pytest.approx(empirical_w2_squared(v, TARGET))
        assert info["dissimilarity_pitch"] == pytest.approx(empirical_w2_squared(th, PITCH_TARGET))
        assert info["dissimilarity"] == pytest.approx(info["dissimilarity_velocity"] + info["dissimilarity_pitch"])
    env.close()


def test_pitch_is_wrapped_and_a_flipped_torso_is_far_from_the_pitch_target():
    from dmdp_sg.envs import wrap_angle
    assert wrap_angle(2 * np.pi + 0.1) == pytest.approx(0.1) and wrap_angle(-np.pi - 0.1) == pytest.approx(np.pi - 0.1)
    env = _make("hybrid")
    env.reset(seed=0)
    env.unwrapped.data.qpos[CFG.pitch_index + 1] = np.pi          # torso flipped (upside down)
    _, _, _, _, info = env.step(np.zeros(6, np.float32))
    assert abs(info["pitch"]) > 3.0 and info["dissimilarity_pitch"] > 8.0
    env.close()


def test_reward_is_the_benchmark_reward_plus_the_threshold_reward():
    # reward of DMDP: r_t + r_tD with r_t = v_t - (kappa c_t)^2 - 0.1 ||a_t||^2, kappa = 0 (the benchmark reward), and
    # r_tD = +10 if D <= 0.1, else 0; the default kappa of make_env is that of TrainConfig, so the environment below uses kappa = 0
    assert CFG.cost_penalty == 0.0 and CFG.d_bonus == 10.0 and CFG.d_threshold == 0.1
    # random actions keep D far above 0.1, so a larger threshold is used to reach both branches of r_tD
    env = make_env(CFG.env_id, 0.0, 0, 10, "state", CFG.mu_star, CFG.sd_star, CFG.vel_index, d_threshold=7.5)
    raw = gym.make(CFG.env_id)
    env.reset(seed=3)
    raw.reset(seed=3)
    rng = np.random.default_rng(0)
    branches = set()
    for _ in range(300):
        action = rng.uniform(-1.0, 1.0, 6).astype(np.float32)
        _, r, _, _, info = env.step(action)
        _, r_raw, _, _, info_raw = raw.step(action)
        assert info["benchmark_reward"] == pytest.approx(r_raw)
        assert info["benchmark_reward"] == pytest.approx(info["x_velocity"] - 0.1 * float(np.sum(action ** 2)), abs=1e-9)
        assert info["d_threshold"] == 7.5
        assert info["r_tD"] == (10.0 if info["dissimilarity"] <= info["d_threshold"] else 0.0)
        assert r == pytest.approx(r_raw - CFG.cost_penalty ** 2 * info_raw["cost"] + info["r_tD"])
        branches.add(info["r_tD"])
    assert branches == {0.0, 10.0}
    env.close()
    raw.close()


def test_threshold_reward_magnitude_and_episode_statistics():
    env = make_env(CFG.env_id, 0.0, 0, 10, "state", CFG.mu_star, CFG.sd_star, CFG.vel_index, d_bonus=0.25, d_threshold=7.5)
    env.reset(seed=1)
    D, within, ret, task_ret = [], 0, 0.0, 0.0
    for t in range(1000):
        _, r, terminated, truncated, info = env.step(env.action_space.sample())
        assert info["r_tD"] == (0.25 if info["dissimilarity"] <= info["d_threshold"] else 0.0)
        assert r == pytest.approx(info["benchmark_reward"] - CFG.cost_penalty ** 2 * info["cost"] + info["r_tD"])
        D.append(info["dissimilarity"])
        within += info["dissimilarity"] <= info["d_threshold"]
        ret += r
        task_ret += info["benchmark_reward"] - CFG.cost_penalty ** 2 * info["cost"]
        if terminated or truncated:
            break
    stats = info["episode_stats"]
    assert 100 < within < 900                                    # both branches of r_tD occur
    assert stats["ret"] == pytest.approx(ret) and stats["within"] == within / 1000
    assert stats["task_ret"] == pytest.approx(task_ret) and stats["ret"] - stats["task_ret"] == pytest.approx(0.25 * within)
    assert stats["dis"] == pytest.approx(np.mean(D)) and stats["dis_median"] == pytest.approx(np.median(D))
    assert 0.0 < stats["flipped"] <= 1.0 and stats["pitch"] > 0.0   # random torques flip the torso in part of the episode
    env.close()
    env = make_env(CFG.env_id, 0.0, 0, 10, "state", CFG.mu_star, CFG.sd_star, CFG.vel_index, d_bonus=0.0)
    env.reset(seed=1)
    _, r, _, _, info = env.step(env.action_space.sample())
    assert info["r_tD"] == 0.0 and r == pytest.approx(info["benchmark_reward"] - CFG.cost_penalty ** 2 * info["cost"])
    env.close()


@pytest.mark.parametrize("kappa", [None, 0.5, 2.0])
def test_cost_penalty_above_the_threshold(kappa):
    # a pushed-forward torso exceeds v_lim, so the penalty (kappa c_t)^2 is subtracted (default kappa = 0: the benchmark reward;
    # kappa = 2 in the design pilot: penalty 4)
    extra = {} if kappa is None else {"cost_penalty": kappa}
    env = make_env(CFG.env_id, 0.0, 0, 10, "state", CFG.mu_star, CFG.sd_star, CFG.vel_index, **extra)
    env.reset(seed=0)
    env.unwrapped.data.qvel[0] = 10.0
    _, r, _, _, info = env.step(np.zeros(6, np.float32))
    assert info["cost"] == 1.0 and info["x_velocity"] > CFG.v_limit
    assert r == pytest.approx(info["benchmark_reward"] - (0.0 if kappa is None else kappa ** 2) + info["r_tD"])
    env.close()


def test_configured_cost_penalty_reaches_the_training_environments():
    from dmdp_sg.envs import CostPenalty, make_vector_env
    envs = make_vector_env(make_config("HalfCheetah", "PPO", 0, num_envs=2, cost_penalty=0.7), nu=0.0)
    for e in envs.envs:
        while not isinstance(e, CostPenalty):
            e = e.env
        assert e.kappa == 0.7
    envs.close()


def test_configured_threshold_reward_reaches_the_training_environments():
    from dmdp_sg.envs import DistributionEstimator, ThresholdReward, make_vector_env
    envs = make_vector_env(make_config("HalfCheetah", "DMDP", 0, num_envs=2, d_threshold=0.3, d_bonus=2.5), nu=0.0)
    for e in envs.envs:
        while not isinstance(e, ThresholdReward):
            e = e.env
        assert e.d == 2.5
        while not isinstance(e, DistributionEstimator):
            e = e.env
        assert e.d_threshold == 0.3
    envs.close()


def test_sensor_noise_is_the_bimodal_gaussian_mixture():
    from dmdp_sg.config import NOISE_OFFSET, NOISE_SD
    from dmdp_sg.envs import SensorNoise
    assert (NOISE_OFFSET, NOISE_SD) == (0.8, 0.6) and NOISE_OFFSET ** 2 + NOISE_SD ** 2 == pytest.approx(1.0)
    noise = SensorNoise(gym.make(CFG.env_id), nu=1.0, seed=3)
    xi = np.stack([noise.sample((OBS_DIM,)) for _ in range(20_000)])
    # entries with mean 0, variance m^2 + s^2 = 1 and the kurtosis m^4 + 6 m^2 s^2 + 3 s^4 = 2.18 of the mixture
    assert abs(xi.mean()) < 0.01 and xi.var() == pytest.approx(1.0, abs=0.01)
    assert np.mean(xi ** 4) == pytest.approx(0.8 ** 4 + 6 * 0.8 ** 2 * 0.6 ** 2 + 3 * 0.6 ** 4, abs=0.03)
    # bimodal: half of the entries come from each component, and the density at 0 is below that of N(0, 1)
    assert np.mean(xi > 0) == pytest.approx(0.5, abs=0.01) and np.mean(np.abs(xi) < 0.2) < 0.13
    assert abs(np.corrcoef(xi[:, 0], xi[:, 1])[0, 1]) < 0.03            # independent entries
    noise.close()


def test_gaussian_noise_reproduces_the_earlier_runs():
    from dmdp_sg.envs import SensorNoise
    noise = SensorNoise(gym.make(CFG.env_id), nu=0.3, seed=10_007, offset=0.0, sd=1.0)
    obs = np.linspace(-1.0, 1.0, OBS_DIM)
    rng = np.random.default_rng(10_007)
    for _ in range(3):        # the earlier code drew o_t = s_t + nu * N(0, I) from the same generator
        np.testing.assert_array_equal(noise.observation(obs), (obs + 0.3 * rng.standard_normal(OBS_DIM)).astype(np.float32))
    noise.close()


def test_configured_noise_mixture_reaches_the_training_environments():
    from dmdp_sg.envs import SensorNoise, make_vector_env
    envs = make_vector_env(make_config("HalfCheetah", "PPO", 0, num_envs=2, noise_offset=0.5, noise_sd=0.7), nu=0.1)
    for e in envs.envs:
        while not isinstance(e, SensorNoise):
            e = e.env
        assert (e.offset, e.sd, e.nu) == (0.5, 0.7, 0.1)
    envs.close()


class _StubEnv(gym.Env):
    """Returns reward 1 and the given window dissimilarities (threshold 0.14), then truncates."""

    observation_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)

    def __init__(self, dissimilarities):
        self.dissimilarities = list(dissimilarities)
        self.t = 0

    def reset(self, *, seed=None, options=None):
        self.t = 0
        return np.zeros(1, np.float32), {}

    def step(self, action):
        info = {"dissimilarity": self.dissimilarities[self.t], "d_threshold": 0.14, "cost": 0.0}
        self.t += 1
        return np.zeros(1, np.float32), 1.0, False, self.t == len(self.dissimilarities), info


def test_threshold_reward_branches_and_task_return():
    # r_tD = +10 if D <= 0.14 (including D = 0.14) and 0 otherwise; the task return excludes r_tD
    from dmdp_sg.envs import EpisodeStats, ThresholdReward
    env = EpisodeStats(ThresholdReward(_StubEnv([0.14, np.nextafter(0.14, 1.0), 0.0, 5.0]), d=10.0))
    env.reset()
    r_td, rewards = [], []
    for _ in range(4):
        _, r, _, truncated, info = env.step(np.zeros(1, np.float32))
        r_td.append(info["r_tD"])
        rewards.append(r)
    assert r_td == [10.0, 0.0, 10.0, 0.0] and rewards == [11.0, 1.0, 11.0, 1.0] and truncated
    stats = info["episode_stats"]
    assert stats["ret"] == 24.0 and stats["task_ret"] == 4.0 and stats["within"] == 0.5


def test_episode_statistics_at_truncation():
    env = _make("state")
    env.reset(seed=0)
    cost = 0.0
    for t in range(1000):
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        cost += info["cost"]
        if terminated or truncated:
            break
    assert t == 999 and truncated and not terminated     # HalfCheetah: H = 1000, no early termination
    stats = info["episode_stats"]
    assert stats["length"] == 1000 and stats["cost"] == cost
    env.close()


def test_empirical_w2_is_exact_and_reduces_to_the_gaussian_closed_form():
    rng = np.random.default_rng(0)
    u = (np.arange(200_000) + 0.5) / 200_000                     # fine quadrature of the quantile coupling
    for target in (Target("gaussian", 2.23, 0.5), Target("uniform", 1.646, 0.9503)):
        for n in (1, 2, 5, 10):
            x = rng.normal(2.0, 1.3, n)
            quadrature = np.mean((np.sort(x)[np.minimum((u * n).astype(int), n - 1)] - target.quantile(u)) ** 2)
            assert empirical_w2_squared(x, target) == pytest.approx(quadrature, rel=1e-4)
        assert empirical_w2_squared(np.array([1.7]), target) == pytest.approx((1.7 - target.mu_star) ** 2 + target.sd_star ** 2)
    x = rng.normal(2.1, 1.0, 100_000)
    assert empirical_w2_squared(x, Target("gaussian", 2.23, 0.5)) == pytest.approx(
        gaussian_w2_squared(x.mean(), x.std(), 2.23, 0.5), abs=2e-3)
    uniform = Target("uniform", 1.646, 0.9503)
    assert empirical_w2_squared(uniform.sample(rng, 100_000), uniform) == pytest.approx(0.0, abs=1e-3)


def test_empirical_w2_penalizes_a_bimodal_window_with_the_target_moments():
    x = np.array([2.23 - 0.5, 2.23 + 0.5] * 5)                  # mean and standard deviation of the target
    assert gaussian_w2_squared(x.mean(), x.std(), 2.23, 0.5) == pytest.approx(0.0, abs=1e-12)
    assert empirical_w2_squared(x, Target("gaussian", 2.23, 0.5)) > 0.05


def test_gaussian_closed_form_is_rejected_for_the_uniform_target():
    cfg = make_config("HalfCheetah", "DMDP", 0)
    with pytest.raises(ValueError):
        make_env(cfg.env_id, 0.0, 0, 10, "hybrid", cfg.mu_star, cfg.sd_star, cfg.vel_index, dissimilarity="gaussian_w2")


def test_window_quantile_is_the_median_window_dissimilarity_under_the_target():
    from dmdp_sg.targets import empirical_w2_squared_batch, window_dissimilarity_quantile
    rng = np.random.default_rng(1)
    for n in (1, 3, 10):
        thr = window_dissimilarity_quantile(TARGET, n, 0.5)
        d = empirical_w2_squared_batch(TARGET.sample(rng, 50_000 * n).reshape(50_000, n), TARGET)
        assert np.mean(d <= thr) == pytest.approx(0.5, abs=0.01)
        x = TARGET.sample(rng, n)
        assert empirical_w2_squared_batch(x[None], TARGET)[0] == pytest.approx(empirical_w2_squared(x, TARGET))
    # the threshold decreases with the window length (sampling error of the empirical distribution)
    assert window_dissimilarity_quantile(TARGET, 10) < window_dissimilarity_quantile(TARGET, 5) < window_dissimilarity_quantile(TARGET, 1)


def test_info_reports_the_fixed_threshold_for_every_window_length():
    env = _make("hybrid", window=4)
    env.reset(seed=0)
    for n in range(1, 7):
        _, _, _, _, info = env.step(env.action_space.sample())
        assert info["d_threshold"] == CFG.d_threshold == 0.1
    env.close()


def test_probability_that_an_iid_window_lies_within_the_threshold():
    from dmdp_sg.targets import empirical_w2_squared_batch, window_within_probability
    rng = np.random.default_rng(2)
    for n in (5, 10):
        d = empirical_w2_squared_batch(WIDE_TARGET.sample(rng, 50_000 * n).reshape(50_000, n), WIDE_TARGET)
        assert window_within_probability(WIDE_TARGET, n, 0.14) == pytest.approx(np.mean(d <= 0.14), abs=0.01)
    assert window_within_probability(WIDE_TARGET, 1, 0.14) == 0.0     # a single sample: D >= sd_star^2 > 0.14
    assert window_within_probability(WIDE_TARGET, 10, 0.14) == pytest.approx(0.53, abs=0.01)
    assert window_within_probability(TARGET, 10, 0.14) == 1.0         # the narrow velocity target: sd_star^2 = 0.054


def test_weights_of_the_velocity_and_pitch_terms():
    env = make_env(CFG.env_id, 0.0, 0, 10, "hybrid", CFG.mu_star, CFG.sd_star, CFG.vel_index, velocity_weight=0.5, pitch_weight=3.0)
    env.reset(seed=0)
    for _ in range(5):
        _, _, _, _, info = env.step(env.action_space.sample())
        assert info["dissimilarity_velocity"] > 0 and info["dissimilarity_pitch"] > 0
        assert info["dissimilarity"] == pytest.approx(0.5 * info["dissimilarity_velocity"] + 3.0 * info["dissimilarity_pitch"])
    env.close()


def test_pitch_after_a_full_turn_counts_as_upright():
    env = _make("hybrid", window=3)
    env.reset(seed=0)
    for _ in range(3):
        env.unwrapped.data.qpos[CFG.pitch_index + 1] = 2 * np.pi + 0.05
        env.unwrapped.data.qvel[:] = 0.0
        obs, _, _, _, info = env.step(np.zeros(6, np.float32))
    assert abs(info["pitch"]) < 0.5 and info["dissimilarity_pitch"] < 0.3          # wrapped true pitch
    assert abs(obs[2]) < 0.5                                                       # wrapped window mean of the readings
    stats_env = env
    while not hasattr(stats_env, "_flipped"):
        stats_env = stats_env.env
    assert stats_env._flipped == 0
    env.close()


def test_windows_restart_after_a_reset():
    env = _make("hybrid")
    env.reset(seed=0)
    for _ in range(15):
        env.step(env.action_space.sample())
    obs, _ = env.reset(seed=1)
    assert obs[1] == 0.0 and obs[3] == 0.0                                         # windows of readings refilled
    _, _, _, _, info = env.step(np.zeros(6, np.float32))
    v, th = info["x_velocity"], info["pitch"]                                      # windows of one true value
    assert info["dissimilarity_velocity"] == pytest.approx((v - CFG.mu_star) ** 2 + CFG.sd_star ** 2)
    assert info["dissimilarity_pitch"] == pytest.approx((th - CFG.pitch_mu_star) ** 2 + CFG.pitch_sd_star ** 2)
    env.close()


def test_flipped_fraction_counts_steps_beyond_a_quarter_turn():
    from dmdp_sg.envs import EpisodeStats, ThresholdReward

    class Pitches(_StubEnv):
        def step(self, action):
            obs, r, te, tr, info = super().step(action)
            info["pitch"] = [0.1, -1.5, 1.7, -3.0][self.t - 1]
            return obs, r, te, tr, info

    env = EpisodeStats(ThresholdReward(Pitches([5.0] * 4), d=10.0))
    env.reset()
    for _ in range(4):
        _, _, _, _, info = env.step(np.zeros(1, np.float32))
    stats = info["episode_stats"]
    assert stats["flipped"] == 0.5 and stats["pitch"] == pytest.approx((0.1 + 1.5 + 1.7 + 3.0) / 4)
