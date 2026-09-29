"""Environment wrappers on a real HalfCheetah environment (Sections 6.1-6.3)."""

import gymnasium as gym
import numpy as np
import pytest

from dmdp_sg.config import make_config
from dmdp_sg.envs import gaussian_w2_squared, make_env

CFG = make_config("HalfCheetah", "DMDP", 0)
OBS_DIM = 17


def _make(policy_input, nu=0.0, window=10, seed=0):
    return make_env(CFG.env_id, nu, seed, window, policy_input, CFG.mu_star, CFG.sd_star, CFG.vel_index)


def test_gaussian_w2_squared():
    assert gaussian_w2_squared(2.0, 0.5, 2.0, 0.5) == 0.0
    assert gaussian_w2_squared(1.0, 0.2, 2.0, 0.5) == pytest.approx(1.0 + 0.09)


@pytest.mark.parametrize("policy_input, dim", [("state", OBS_DIM), ("window", OBS_DIM + 10), ("hybrid", OBS_DIM + 2)])
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
    readings = [float(obs[2 + CFG.vel_index])]           # o^v_t is part of o_t, which follows omega_hat_t
    # at reset, the window is filled with the first reading: omega_hat_0 = (o^v_0, 0)
    assert obs[0] == pytest.approx(readings[0], rel=1e-6)
    assert obs[1] == pytest.approx(0.0, abs=1e-6)
    velocities = []
    for _ in range(8):
        obs, _, _, _, info = env.step(env.action_space.sample())
        readings.append(float(obs[2 + CFG.vel_index]))
        velocities.append(info["x_velocity"])
        w = np.array(([readings[0]] * (window - 1) + readings)[-window:])
        assert obs[0] == pytest.approx(w.mean(), rel=1e-5, abs=1e-6)
        assert obs[1] == pytest.approx(w.std(), rel=1e-5, abs=1e-6)
        v = np.array(velocities[-window:])                  # true velocities, available steps only
        assert info["dissimilarity"] == pytest.approx(gaussian_w2_squared(v.mean(), v.std(), CFG.mu_star, CFG.sd_star))
    env.close()


def test_window_input_contains_raw_readings():
    window = 4
    env = _make("window", nu=0.1, window=window)
    obs, _ = env.reset(seed=2)
    readings = [obs[CFG.vel_index]] * window
    for _ in range(6):
        obs, *_ = env.step(env.action_space.sample())
        readings = readings[1:] + [obs[CFG.vel_index]]
        np.testing.assert_array_equal(obs[OBS_DIM:], np.array(readings, np.float32))
    env.close()


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
