"""Environment wrappers that implement the practical DMDP interface (Section 4.5 and Appendix E).

The wrappers are stacked as  gym env -> CostPenalty -> SensorNoise -> DistributionEstimator -> ThresholdReward -> EpisodeStats,
so that the reward is r_t + r_tD with the reward r_t = v_t - (kappa * c_t)^2 - 0.1 * ||a_t||^2 of the run (r_tD = 0 for d = 0):

CostPenalty            reward r_t = v_t - (kappa * c_t)^2 - 0.1 * ||a_t||^2: the benchmark reward r^b_t = v_t - 0.1 * ||a_t||^2
                       for kappa = 0 (every reported method); kappa = COST_PENALTY = 2 only in the design pilot  (Appendix E)
SensorNoise            noisy readings o_t = s_t + nu * xi_t, the entries of xi_t independent draws from the bimodal
                       Gaussian mixture 0.5 N(-m, s^2) + 0.5 N(m, s^2) with m^2 + s^2 = 1   (Appendix E)
DistributionEstimator  policy input and training dissimilarity:
                       omega_hat_t = (window mean, window std) of the last n_w readings of the velocity and of
                       the torso pitch, i.e. the method-of-moments parameters of the location-scale families of
                       the two marginals of rho_star (Appendix E), and the stage dissimilarity
                       D(rho_bar_t || rho_star), the sum of the squared 2-Wasserstein distances between the
                       empirical distributions of the windows of TRUE velocities and pitch angles and their
                       targets (Appendix E), and its threshold D_thr (Appendix E)
ThresholdReward        adds the threshold reward r_tD = +d if D(rho_bar_t || rho_star) <= D_thr and 0 otherwise
EpisodeStats           episodic return (with and without r_tD), cost, mean and median dissimilarity, fraction of steps
                       within D_thr, mean absolute pitch and fraction of steps with the torso flipped (|pitch| > pi/2)
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
import safety_gymnasium  # noqa: F401  (registers the Safety-Gymnasium environments)

from .config import NOISE_OFFSET, NOISE_SD, TrainConfig
from .targets import Target, empirical_w2_squared

POLICY_INPUTS = ("state", "hybrid")
DISSIMILARITIES = ("gaussian_w2", "empirical_w2")


def gaussian_w2_squared(mu: float, sd: float, mu_star: float, sd_star: float) -> float:
    """Squared 2-Wasserstein distance between N(mu, sd^2) and N(mu_star, sd_star^2) (Appendix E).

    D(rho_omega || rho_star) = (mu - mu_star)^2 + (sd - sd_star)^2.
    """
    return float((mu - mu_star) ** 2 + (sd - sd_star) ** 2)


class CostPenalty(gym.Wrapper):
    """Subtracts the cost penalty (kappa * c_t)^2 from the benchmark reward: r_t = v_t - (kappa * c_t)^2 - 0.1 * ||a_t||^2.

    The benchmark reward v_t - 0.1 * ||a_t||^2 is kept in ``info["benchmark_reward"]``. Because c_t = 1[v_t > v_limit]
    is 0 or 1, the penalty is kappa^2 at every step above the threshold. Every reported method uses kappa = 0, i.e. the
    benchmark reward; only the runs of the design pilot (results/pilot/) used kappa = COST_PENALTY = 2.
    """

    def __init__(self, env: gym.Env, kappa: float):
        super().__init__(env)
        self.kappa = float(kappa)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["benchmark_reward"] = float(reward)
        return obs, float(reward) - (self.kappa * float(info["cost"])) ** 2, terminated, truncated, info


class SensorNoise(gym.ObservationWrapper):
    """Additive Gaussian-mixture sensor noise on every observation channel, o_t = s_t + nu * xi_t.

    The entries of xi_t are independent draws from the bimodal mixture 0.5 N(-offset, sd^2) + 0.5 N(offset, sd^2): a
    component mean -offset or +offset with probability 1/2 each, plus Gaussian noise with standard deviation sd. The
    mixture has mean 0 and variance offset^2 + sd^2, which is 1 for the defaults (NOISE_OFFSET, NOISE_SD), so nu is the
    standard deviation of the noise. With ``offset = 0`` and ``sd = 1`` the noise is Gaussian, xi_t ~ N(0, I), and no
    component is drawn, which reproduces the runs made before the mixture was introduced.

    With ``nu_max > 0`` the noise level is redrawn at every reset as nu ~ U[0, nu_max]
    (training); otherwise the fixed level ``nu`` is used (evaluation, or training with a
    fixed noise level through ``TrainConfig.train_noise``).
    """

    def __init__(self, env: gym.Env, nu: float, seed: int, nu_max: float = 0.0, offset: float = NOISE_OFFSET,
                 sd: float = NOISE_SD):
        super().__init__(env)
        self.nu = float(nu)
        self.nu_max = float(nu_max)
        self.offset = float(offset)
        self.sd = float(sd)
        self.rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        if self.nu_max > 0.0:
            self.nu = float(self.rng.uniform(0.0, self.nu_max))
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def sample(self, shape: tuple[int, ...]) -> np.ndarray:
        """One draw of xi_t with independent entries from the noise mixture."""
        xi = self.sd * self.rng.standard_normal(shape)
        if self.offset > 0.0:
            xi += self.offset * (2.0 * self.rng.integers(0, 2, size=shape) - 1.0)   # component mean -offset or +offset
        return xi

    def observation(self, obs: np.ndarray) -> np.ndarray:
        if self.nu <= 0.0:
            return obs.astype(np.float32)
        return (obs + self.nu * self.sample(obs.shape)).astype(np.float32)


def wrap_angle(angle: float) -> float:
    """The angle in (-pi, pi]; the pitch of a torso that has turned a full revolution counts as upright."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


class DistributionEstimator(gym.Wrapper):
    """Sliding-window estimate of the distribution of the regulated features and the dissimilarity stage cost.

    The regulated features are the forward velocity v_t and, if ``pitch_weight > 0``, the pitch angle theta_t of the
    torso. Policy input (observation returned by ``reset`` and ``step``), for d observation channels:

    * ``"state"``:  o_t                                      (d features)
    * ``"hybrid"``: (omega_hat_t, o_t), omega_hat_t the window mean and standard deviation of the last n_w velocity
      readings and, if ``pitch_weight > 0``, of the last n_w pitch readings  (d + 2 or d + 4 features)

    At the start of an episode the windows of readings are filled with the first reading.

    ``info["dissimilarity"]`` is the training dissimilarity

        D_t = velocity_weight * W_2^2(window of v, velocity target) + pitch_weight * W_2^2(window of theta, pitch target),

    computed from the TRUE velocities v_{t-n_w+1}, ..., v_t and pitch angles (over the available steps at the start of
    an episode); it is used only as a training signal (Appendix E). The two terms are kept in
    ``info["dissimilarity_velocity"]`` and ``info["dissimilarity_pitch"]``, and ``info["pitch"]`` is theta_t in
    (-pi, pi]. With ``dissimilarity="gaussian_w2"`` the velocity term is D(rho_{omega_bar_t} || rho_star), where
    omega_bar_t is the window mean and standard deviation; with ``"empirical_w2"`` it is the squared 2-Wasserstein
    distance between the empirical distribution of the window and the target (``targets.empirical_w2_squared``). The
    velocity target is the member of ``target_family`` (``targets.TARGET_FAMILIES``) with mean mu_star and standard
    deviation sd_star; the pitch target is the uniform distribution with mean pitch_mu_star and standard deviation
    pitch_sd_star. ``info["d_threshold"]`` is the threshold D_thr of the threshold reward r_tD (Appendix E), the
    same for every number of samples in the window.
    The wrapped environment provides ``info["x_velocity"]`` (v_t) and ``info["cost"]`` (c_t = 1[v_t > v_limit]); the
    true pitch is ``qpos[pitch_index + 1]`` of the simulator (the observation omits the horizontal position).
    """

    def __init__(self, env: gym.Env, window: int, policy_input: str, mu_star: float, sd_star: float,
                 vel_index: int, dissimilarity: str = TrainConfig.dissimilarity,
                 target_family: str = TrainConfig.target_family, d_threshold: float = TrainConfig.d_threshold,
                 pitch_index: int = TrainConfig.pitch_index, pitch_mu_star: float = TrainConfig.pitch_mu_star,
                 pitch_sd_star: float = TrainConfig.pitch_sd_star, velocity_weight: float = TrainConfig.velocity_weight,
                 pitch_weight: float = TrainConfig.pitch_weight):
        super().__init__(env)
        if policy_input not in POLICY_INPUTS:
            raise ValueError(f"policy_input must be one of {POLICY_INPUTS}, got {policy_input!r}")
        if dissimilarity not in DISSIMILARITIES:
            raise ValueError(f"dissimilarity must be one of {DISSIMILARITIES}, got {dissimilarity!r}")
        if target_family != "gaussian" and dissimilarity == "gaussian_w2":
            raise ValueError("the closed form gaussian_w2 applies to the Gaussian family only; use empirical_w2")
        self.dissimilarity = dissimilarity
        self.target = Target(target_family, mu_star, sd_star)
        self.pitch_target = Target("uniform", pitch_mu_star, pitch_sd_star)
        self.window = int(window)
        self.policy_input = policy_input
        self.mu_star = float(mu_star)
        self.sd_star = float(sd_star)
        self.vel_index = int(vel_index)
        self.pitch_index = int(pitch_index)
        self.velocity_weight = float(velocity_weight)
        self.pitch_weight = float(pitch_weight)
        self._readings = deque(maxlen=self.window)        # noisy velocity readings o^v_t (available to the policy)
        self._pitch_readings = deque(maxlen=self.window)  # noisy pitch readings (available to the policy)
        self._velocities = deque(maxlen=self.window)      # true velocities v_t (training dissimilarity only)
        self._pitches = deque(maxlen=self.window)         # true pitch angles theta_t (training dissimilarity only)
        self._w2_weights = [self.target.w2_weights(n) for n in range(1, self.window + 1)]
        self._pitch_w2_weights = [self.pitch_target.w2_weights(n) for n in range(1, self.window + 1)]
        self.d_threshold = float(d_threshold)
        d = env.observation_space.shape[0]
        dims = {"state": d, "hybrid": d + (4 if self.pitch_weight > 0.0 else 2)}[policy_input]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (dims,), np.float32)

    def _features(self, obs: np.ndarray) -> np.ndarray:
        reading, pitch_reading = float(obs[self.vel_index]), wrap_angle(float(obs[self.pitch_index]))
        if not self._readings:                   # at reset, fill the windows with the first reading
            for _ in range(self.window - 1):
                self._readings.append(reading)
                self._pitch_readings.append(pitch_reading)
        self._readings.append(reading)
        self._pitch_readings.append(pitch_reading)
        if self.policy_input == "state":
            return obs.astype(np.float32)
        readings = np.asarray(self._readings)
        omega_hat = [readings.mean(), readings.std()]
        if self.pitch_weight > 0.0:
            pitch_readings = np.asarray(self._pitch_readings)
            omega_hat += [pitch_readings.mean(), pitch_readings.std()]
        return np.concatenate([np.array(omega_hat), obs]).astype(np.float32)

    def _dissimilarity(self) -> tuple[float, float]:
        """Velocity term and pitch term of the window dissimilarity (the pitch term is 0 if pitch_weight = 0)."""
        velocities = np.asarray(self._velocities)
        if self.dissimilarity == "empirical_w2":
            d_velocity = empirical_w2_squared(velocities, self.target, self._w2_weights[velocities.size - 1])
        else:
            d_velocity = gaussian_w2_squared(velocities.mean(), velocities.std(), self.mu_star, self.sd_star)
        if self.pitch_weight <= 0.0:
            return d_velocity, 0.0
        pitches = np.asarray(self._pitches)
        return d_velocity, empirical_w2_squared(pitches, self.pitch_target, self._pitch_w2_weights[pitches.size - 1])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._readings.clear()
        self._pitch_readings.clear()
        self._velocities.clear()
        self._pitches.clear()
        return self._features(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pitch = wrap_angle(float(self.env.unwrapped.data.qpos[self.pitch_index + 1]))
        self._velocities.append(float(info["x_velocity"]))
        self._pitches.append(pitch)
        d_velocity, d_pitch = self._dissimilarity()
        info["pitch"] = pitch
        info["dissimilarity_velocity"] = d_velocity
        info["dissimilarity_pitch"] = d_pitch
        info["dissimilarity"] = self.velocity_weight * d_velocity + self.pitch_weight * d_pitch
        info["d_threshold"] = self.d_threshold
        return self._features(obs), reward, terminated, truncated, info


class ThresholdReward(gym.Wrapper):
    """Adds the threshold reward r_tD to the reward: +d if D(rho_bar_t || rho_star) <= D_thr and 0 otherwise.

    D and D_thr are ``info["dissimilarity"]`` and ``info["d_threshold"]`` of ``DistributionEstimator``; r_tD is kept in
    ``info["r_tD"]``. The reward is r_t + r_tD with the reward r_t = v_t - (kappa * c_t)^2 - 0.1 * ||a_t||^2 of the run
    (``CostPenalty``; the benchmark reward for kappa = 0) (Appendix E); d = 0 for the baselines.
    """

    def __init__(self, env: gym.Env, d: float):
        super().__init__(env)
        self.d = float(d)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        r_td = self.d if info["dissimilarity"] <= info["d_threshold"] else 0.0
        info["r_tD"] = r_td
        return obs, float(reward) + r_td, terminated, truncated, info


class EpisodeStats(gym.Wrapper):
    """Accumulates the episodic return, the cost and statistics of the window dissimilarity.

    At the end of an episode, ``info["episode_stats"]`` holds ``ret`` (return of the training reward r_t + r_tD),
    ``task_ret`` (return of the reward r_t of the run, without r_tD; the benchmark return for kappa = 0), ``cost``, ``dis`` (mean dissimilarity per step),
    ``dis_median`` (median over the steps), ``within`` (fraction of the steps with D <= D_thr), ``pitch`` (mean
    absolute pitch angle of the torso), ``flipped`` (fraction of the steps with |pitch| > pi/2, i.e. the torso is
    upside down) and ``length``.
    """

    def reset(self, **kwargs):
        self._ret = 0.0
        self._task_ret = 0.0
        self._cost = 0.0
        self._dis: list[float] = []
        self._within = 0
        self._pitch = 0.0
        self._flipped = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ret += float(reward)
        self._task_ret += float(reward) - float(info.get("r_tD", 0.0))
        self._cost += float(info.get("cost", 0.0))
        self._dis.append(float(info.get("dissimilarity", 0.0)))
        self._within += int(info.get("dissimilarity", 0.0) <= info.get("d_threshold", float("inf")))
        self._pitch += abs(float(info.get("pitch", 0.0)))
        self._flipped += int(abs(float(info.get("pitch", 0.0))) > np.pi / 2)
        if terminated or truncated:
            n = len(self._dis)
            info["episode_stats"] = dict(ret=self._ret, task_ret=self._task_ret, cost=self._cost, dis=float(np.mean(self._dis)),
                                         dis_median=float(np.median(self._dis)), within=self._within / n,
                                         pitch=self._pitch / n, flipped=self._flipped / n, length=n)
        return obs, reward, terminated, truncated, info


def env_settings(cfg: TrainConfig) -> dict:
    """The keyword arguments of ``make_env`` that are stored in a ``TrainConfig`` (all except nu, seed and nu_max)."""
    return dict(env_id=cfg.env_id, window=cfg.window, policy_input=cfg.policy_input, mu_star=cfg.mu_star,
                sd_star=cfg.sd_star, vel_index=cfg.vel_index, dissimilarity=cfg.dissimilarity,
                target_family=cfg.target_family, cost_penalty=cfg.cost_penalty, d_threshold=cfg.d_threshold,
                d_bonus=cfg.d_bonus, pitch_index=cfg.pitch_index, pitch_mu_star=cfg.pitch_mu_star,
                pitch_sd_star=cfg.pitch_sd_star, velocity_weight=cfg.velocity_weight, pitch_weight=cfg.pitch_weight,
                noise_offset=cfg.noise_offset, noise_sd=cfg.noise_sd)


def make_env(env_id: str, nu: float, seed: int, window: int, policy_input: str, mu_star: float,
             sd_star: float, vel_index: int, nu_max: float = 0.0, dissimilarity: str = TrainConfig.dissimilarity,
             target_family: str = TrainConfig.target_family, cost_penalty: float = TrainConfig.cost_penalty,
             d_threshold: float = TrainConfig.d_threshold, d_bonus: float = TrainConfig.d_bonus,
             pitch_index: int = TrainConfig.pitch_index, pitch_mu_star: float = TrainConfig.pitch_mu_star,
             pitch_sd_star: float = TrainConfig.pitch_sd_star, velocity_weight: float = TrainConfig.velocity_weight,
             pitch_weight: float = TrainConfig.pitch_weight, noise_offset: float = TrainConfig.noise_offset,
             noise_sd: float = TrainConfig.noise_sd) -> gym.Env:
    """One Safety-Gymnasium environment with cost penalty, sensor noise, distribution estimator, threshold reward and statistics.

    ``seed`` seeds the action space; the sensor noise uses the seed ``seed + 10_000``. The
    simulator itself is seeded by the ``seed`` argument of ``reset``.
    """
    env = gym.make(env_id)
    env = CostPenalty(env, kappa=cost_penalty)
    env = SensorNoise(env, nu=nu, seed=seed + 10_000, nu_max=nu_max, offset=noise_offset, sd=noise_sd)
    env = DistributionEstimator(env, window=window, policy_input=policy_input, mu_star=mu_star,
                                sd_star=sd_star, vel_index=vel_index, dissimilarity=dissimilarity,
                                target_family=target_family, d_threshold=d_threshold, pitch_index=pitch_index,
                                pitch_mu_star=pitch_mu_star, pitch_sd_star=pitch_sd_star,
                                velocity_weight=velocity_weight, pitch_weight=pitch_weight)
    env = ThresholdReward(env, d=d_bonus)
    env = EpisodeStats(env)
    env.action_space.seed(seed)
    return env


def make_vector_env(cfg: TrainConfig, nu: float, nu_max: float = 0.0) -> gym.vector.SyncVectorEnv:
    """``cfg.num_envs`` training environments; environment i uses the seed ``1000 * cfg.seed + i``."""
    def thunk(i: int):
        return lambda: make_env(nu=nu, seed=cfg.seed * 1000 + i, nu_max=nu_max, **env_settings(cfg))
    return gym.vector.SyncVectorEnv([thunk(i) for i in range(cfg.num_envs)])
