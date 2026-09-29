"""Environment wrappers that implement the practical DMDP interface (Section 6).

The wrappers are stacked as  gym env -> SensorNoise -> DistributionEstimator -> EpisodeStats:

SensorNoise            noisy readings o_t = s_t + nu * xi_t, xi_t ~ N(0, I)           (Section 6.2)
DistributionEstimator  policy input and training dissimilarity:
                       omega_hat_t = (window mean, window std) of the last n_w readings o^v_t,
                       i.e. the maximum-likelihood parameter of rho_omega = N(mu, sd^2) fitted to
                       the window (Section 6.2, Appendix C.1), and the stage dissimilarity
                       D(rho_{omega_bar_t} || rho_star), computed from the window of TRUE
                       velocities v_t (Section 6.3)
EpisodeStats           episodic return, cost and mean dissimilarity for logging
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
import safety_gymnasium  # noqa: F401  (registers the Safety-Gymnasium environments)

from .config import TrainConfig

POLICY_INPUTS = ("state", "window", "hybrid")


def gaussian_w2_squared(mu: float, sd: float, mu_star: float, sd_star: float) -> float:
    """Squared 2-Wasserstein distance between N(mu, sd^2) and N(mu_star, sd_star^2) (Section 6.1).

    D(rho_omega || rho_star) = (mu - mu_star)^2 + (sd - sd_star)^2.
    """
    return float((mu - mu_star) ** 2 + (sd - sd_star) ** 2)


class SensorNoise(gym.ObservationWrapper):
    """Additive Gaussian sensor noise on every observation channel, o_t = s_t + nu * xi_t.

    With ``nu_max > 0`` the noise level is redrawn at every reset as nu ~ U[0, nu_max]
    (training); otherwise the fixed level ``nu`` is used (evaluation, or training with a
    fixed noise level through ``TrainConfig.train_noise``).
    """

    def __init__(self, env: gym.Env, nu: float, seed: int, nu_max: float = 0.0):
        super().__init__(env)
        self.nu = float(nu)
        self.nu_max = float(nu_max)
        self.rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        if self.nu_max > 0.0:
            self.nu = float(self.rng.uniform(0.0, self.nu_max))
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def observation(self, obs: np.ndarray) -> np.ndarray:
        if self.nu <= 0.0:
            return obs.astype(np.float32)
        return (obs + self.nu * self.rng.standard_normal(obs.shape)).astype(np.float32)


class DistributionEstimator(gym.Wrapper):
    """Sliding-window estimate of the velocity distribution and the dissimilarity stage cost.

    Policy input (observation returned by ``reset`` and ``step``), for d observation channels:

    * ``"state"``:  o_t                                      (d features)
    * ``"window"``: (o_t, o^v_{t-n_w+1}, ..., o^v_t)         (d + n_w features)
    * ``"hybrid"``: (omega_hat_t, o_t), omega_hat_t = (mu_hat_t, sd_hat_t) the window mean and
      standard deviation of the last n_w velocity readings  (d + 2 features)

    At the start of an episode the window of readings is filled with the first reading.

    ``info["dissimilarity"]`` is D(rho_{omega_bar_t} || rho_star), where omega_bar_t is the window
    mean and standard deviation of the TRUE velocities v_{t-n_w+1}, ..., v_t (over the available
    steps at the start of an episode). It is used only as a training signal (Section 6.3).
    The wrapped environment provides ``info["x_velocity"]`` (v_t) and ``info["cost"]``
    (c_t = 1[v_t > v_limit]).
    """

    def __init__(self, env: gym.Env, window: int, policy_input: str, mu_star: float, sd_star: float,
                 vel_index: int):
        super().__init__(env)
        if policy_input not in POLICY_INPUTS:
            raise ValueError(f"policy_input must be one of {POLICY_INPUTS}, got {policy_input!r}")
        self.window = int(window)
        self.policy_input = policy_input
        self.mu_star = float(mu_star)
        self.sd_star = float(sd_star)
        self.vel_index = int(vel_index)
        self._readings = deque(maxlen=self.window)    # noisy readings o^v_t (available to the policy)
        self._velocities = deque(maxlen=self.window)  # true velocities v_t (training dissimilarity only)
        d = env.observation_space.shape[0]
        dims = {"state": d, "window": d + self.window, "hybrid": d + 2}[policy_input]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (dims,), np.float32)

    def _features(self, obs: np.ndarray) -> np.ndarray:
        reading = float(obs[self.vel_index])
        if not self._readings:                   # at reset, fill the window with the first reading
            for _ in range(self.window - 1):
                self._readings.append(reading)
        self._readings.append(reading)
        if self.policy_input == "state":
            return obs.astype(np.float32)
        readings = np.asarray(self._readings)
        if self.policy_input == "window":
            return np.concatenate([obs, readings]).astype(np.float32)
        omega_hat = np.array([readings.mean(), readings.std()])
        return np.concatenate([omega_hat, obs]).astype(np.float32)

    def _dissimilarity(self) -> float:
        velocities = np.asarray(self._velocities)
        return gaussian_w2_squared(velocities.mean(), velocities.std(), self.mu_star, self.sd_star)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._readings.clear()
        self._velocities.clear()
        return self._features(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._velocities.append(float(info["x_velocity"]))
        info["dissimilarity"] = self._dissimilarity()
        return self._features(obs), reward, terminated, truncated, info


class EpisodeStats(gym.Wrapper):
    """Accumulates the episodic return, cost and mean dissimilarity.

    At the end of an episode, ``info["episode_stats"]`` holds ``ret``, ``cost``, ``dis``
    (mean dissimilarity per step) and ``length``.
    """

    def reset(self, **kwargs):
        self._ret = 0.0
        self._cost = 0.0
        self._dis = 0.0
        self._len = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ret += float(reward)
        self._cost += float(info.get("cost", 0.0))
        self._dis += float(info.get("dissimilarity", 0.0))
        self._len += 1
        if terminated or truncated:
            info["episode_stats"] = dict(ret=self._ret, cost=self._cost, dis=self._dis / max(self._len, 1),
                                         length=self._len)
        return obs, reward, terminated, truncated, info


def make_env(env_id: str, nu: float, seed: int, window: int, policy_input: str, mu_star: float,
             sd_star: float, vel_index: int, nu_max: float = 0.0) -> gym.Env:
    """One Safety-Gymnasium environment with sensor noise, distribution estimator and statistics.

    ``seed`` seeds the action space; the sensor noise uses the seed ``seed + 10_000``. The
    simulator itself is seeded by the ``seed`` argument of ``reset``.
    """
    env = gym.make(env_id)
    env = SensorNoise(env, nu=nu, seed=seed + 10_000, nu_max=nu_max)
    env = DistributionEstimator(env, window=window, policy_input=policy_input, mu_star=mu_star,
                                sd_star=sd_star, vel_index=vel_index)
    env = EpisodeStats(env)
    env.action_space.seed(seed)
    return env


def make_vector_env(cfg: TrainConfig, nu: float, nu_max: float = 0.0) -> gym.vector.SyncVectorEnv:
    """``cfg.num_envs`` training environments; environment i uses the seed ``1000 * cfg.seed + i``."""
    def thunk(i: int):
        return lambda: make_env(cfg.env_id, nu, cfg.seed * 1000 + i, cfg.window, cfg.policy_input,
                                cfg.mu_star, cfg.sd_star, cfg.vel_index, nu_max=nu_max)
    return gym.vector.SyncVectorEnv([thunk(i) for i in range(cfg.num_envs)])
