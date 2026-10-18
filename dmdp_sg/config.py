"""Experiment configuration (Section 4.5 and Appendix E of the paper).

Design of the study: the regulated features are the forward velocity and the pitch angle of the torso, and the target
rho_star is the product of the velocity band U[0.70 v_limit, 0.95 v_limit] and the pitch target U[-0.2, 0.2]. PPO, PPO-Lag,
SAC and SAC-Lag are trained on the benchmark reward r^b_t = v_t - 0.1 * ||a_t||^2, and so are DMDP and DMDP-Lag, which add the threshold
reward (d = 10, D_thr = 0.1) and the dissimilarity term -beta * D (beta = 0.5). Distributional PPO and Distributional SAC are the
distributional-RL baselines: PPO and SAC on the benchmark reward whose critics learn the distribution of the return by quantile
regression (N_QUANTILES). Implicit PPO and Implicit SAC (Implicit-PPO, Implicit-SAC) are Distributional PPO and SAC
whose policy receives the window statistics omega_hat_t of the DMDP methods: distribution feedback without a target.
Distributional PPO-Lag, Distributional SAC-Lag, Implicit PPO-Lag and Implicit SAC-Lag add the cost constraint of PPO-Lag and
SAC-Lag to these four baselines. PPO-Lag, SAC-Lag, the four distributional -Lag baselines and DMDP-Lag enforce the cost limit
c_bar = 10. The initial standard deviation of the PPO actions is 0.37 for every method
(INIT_LOG_STD = -1).

Notation, as in the paper:

    s_t          observation vector of the benchmark
    o_t          noisy sensor reading, o_t = s_t + nu * xi_t, where the entries of xi_t are independent draws from the
                 bimodal Gaussian mixture 0.5 N(-m, s^2) + 0.5 N(m, s^2), m = NOISE_OFFSET, s = NOISE_SD, with mean 0 and
                 variance m^2 + s^2 = 1, so that nu is the standard deviation of the noise   (Appendix E)
    v_t          true forward velocity, the first regulated feature               (Appendix E)
    theta_t      true pitch angle of the torso, the second regulated feature      (Appendix E)
    c_t          safety cost 1[v_t > v_limit] of the benchmark
    r^b_t        benchmark reward v_t - 0.1 * ||a_t||^2, the reward of every reported method (kappa = 0 in the reward
                 r^b_t - (kappa * c_t)^2 of envs.CostPenalty). Only the runs of the design pilot (results/pilot/) were trained
                 with the cost penalty kappa = COST_PENALTY. The reported return R is that of the reward the method is trained
                 on, without -beta * D and -lambda * c_t (analysis.episode_returns)
    o^v_t        noisy reading of the forward velocity, o_t[vel_index]
    rho_k        distribution of v_k at step k across evaluation episodes
    rho_star     target distribution of (v, theta): the product of the velocity target U[lo * v_limit, hi * v_limit]
                 (TARGET_BAND; mean mu_star, standard deviation sd_star) and the pitch target
                 U[-PITCH_HALF_WIDTH, PITCH_HALF_WIDTH] (pitch_mu_star, pitch_sd_star)          (Appendix E)
    rho_bar_t    empirical distribution of the last n_w true velocities and pitch angles (training only)
    omega_hat_t  window mean and std of the last n_w readings of the velocity and of the pitch (policy input, Appendix E)
    D            training dissimilarity: the sum of the squared 2-Wasserstein distances W_2^2 of the velocity window and
                 of the pitch window to their targets (weights velocity_weight and pitch_weight)
    D_thr        threshold of the window dissimilarity (D_THRESHOLD = 0.1, the same for every window length)
    r_tD         threshold reward: +d if D <= D_thr and 0 if D > D_thr, d = D_BONUS = 10
    beta         weight of D: DMDP is trained on r^b_t + r_tD - beta * D and DMDP-Lag on
                 r^b_t + r_tD - beta * D - lambda * c_t (Appendix E)
    c_bar_0      standard episodic cost limit of the benchmark (BENCHMARK_COST_LIMIT = 25; targets of earlier designs only)
    c_bar        cost limit of the Lagrangian constraint of every -Lag method (COST_LIMIT = 10)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

from .targets import target_mean as _target_mean

EPISODE_LENGTH = 1000   # H, steps per episode (HalfCheetah never terminates early)
V_LIMIT = 3.2096        # v_limit: velocity threshold of the cost c_t = 1[v_t > v_limit], fixed by the simulator (which produces the cost)
BENCHMARK_COST_LIMIT = 25.0   # c_bar_0, standard episodic cost limit of the benchmark (used only by the targets of earlier designs)
COST_LIMIT = 10.0       # c_bar: episodic cost limit of every -Lag method (chosen in the design pilot on seeds 10-12)
COST_PENALTY = 2.0      # kappa of the cost penalty (kappa * c_t)^2 of the design pilot (results/pilot/); every reported method uses kappa = 0
D_THRESHOLD = 0.1       # D_thr of the threshold reward, the same for every window length (0.14 in the design pilot)
TARGET_BAND = (0.70, 0.95)   # velocity target U[lo * v_limit, hi * v_limit]: fast, with a margin of 5% below the threshold
PITCH_HALF_WIDTH = 0.2  # pitch target U[-0.2, 0.2] rad: upright torso
INIT_LOG_STD = -1.0     # initial log std of the actions (std 0.37): with the PPO default 0 (std 1), the random actions of the
                        # first rollouts flip the torso, and most runs of PPO and PPO-Lag keep the flipped gait
D_BONUS = 10.0          # d of the threshold reward r_tD = +d if D <= D_thr and 0 otherwise (DMDP and DMDP-Lag)
LEARNERS = ("ppo", "sac")   # PPO (train.py; the main study) and SAC (sac.py; the second learner, Section 5.1)
SAC_TOTAL_STEPS = 1_000_000  # environment steps per SAC run
SAC_NUM_ENVS = 4             # parallel environments of the SAC learner (one gradient step per environment step)
NOISE_OFFSET = 0.8           # m: the entries of xi_t are drawn from 0.5 N(-m, s^2) + 0.5 N(m, s^2) (mean 0, variance m^2 + s^2 = 1)
NOISE_SD = 0.6               # s: standard deviation of each component of the noise mixture
N_QUANTILES = 32             # quantiles of the return distribution learned by the critics of Distributional PPO and SAC


def target_mean(v_limit: float, sd_star: float, cost_limit: float = BENCHMARK_COST_LIMIT,
                horizon: int = EPISODE_LENGTH, family: str = "gaussian") -> float:
    """Mean of the cost-derived target of an earlier design (not reported in the paper).

    rho_star is the fastest member of ``family`` (``targets.TARGET_FAMILIES``) with standard deviation sd_star whose
    expected episodic cost H * P(v > v_limit) equals the benchmark cost limit c_bar_0 (default 25):

        mu_star = v_limit - q_{1 - c_bar_0 / H} * sd_star,

    where q_p is the p-quantile of the standardized member of the family (z_p for the Gaussian family).
    """
    return _target_mean(v_limit, sd_star, cost_limit, horizon, family)


def uniform_target(v_limit: float, cost_limit: float = BENCHMARK_COST_LIMIT, horizon: int = EPISODE_LENGTH) -> tuple[float, float]:
    """Mean and standard deviation of the uniform target U[0, v_max] of an earlier design (not reported in the paper).

    The lower edge is at rest and v_max is the largest upper edge whose expected episodic cost
    H * P(v > v_limit) = H * (v_max - v_limit) / v_max equals the benchmark limit c_bar_0, i.e., v_max = v_limit / (1 - c_bar_0 / H).
    """
    v_max = v_limit / (1.0 - cost_limit / horizon)
    return v_max / 2.0, v_max / 12 ** 0.5


def band_target(v_limit: float, band: tuple[float, float] = TARGET_BAND) -> tuple[float, float]:
    """Mean and standard deviation of the velocity target rho_star = U[lo * v_limit, hi * v_limit] (Appendix E).

    The target is a high-reward, safe distribution: the velocity stays in a band below the threshold v_limit, so the
    expected cost under rho_star is zero, and the band is as fast as the margin (1 - hi) allows.
    """
    lo, hi = band
    return (lo + hi) / 2.0 * v_limit, (hi - lo) * v_limit / 12 ** 0.5


# ---------------------------------------------------------------------------
# Environment (Safety-Gymnasium velocity task with the v1 threshold of the benchmark)
# ---------------------------------------------------------------------------
_UNIFORM_MU, _UNIFORM_SD = band_target(V_LIMIT)        # velocity target U[2.2467, 3.0491]: mean 2.6479, standard deviation 0.2316
ENVIRONMENTS: dict[str, dict[str, Any]] = {
    "HalfCheetah": dict(
        env_id="SafetyHalfCheetahVelocityGymnasium-v1",
        vel_index=8,             # index of the forward velocity of the torso in the observation vector
        pitch_index=1,           # index of the pitch angle of the torso in the observation vector (qpos[pitch_index + 1])
        v_limit=V_LIMIT,         # velocity threshold; cost c_t = 1[v_t > v_limit]
        target_family="uniform", # family of the velocity target (targets.TARGET_FAMILIES)
        mu_star=_UNIFORM_MU,     # mean of the velocity target U[0.70 v_limit, 0.95 v_limit]
        sd_star=_UNIFORM_SD,     # standard deviation of the velocity target
        dissimilarity="empirical_w2",         # training dissimilarity: exact W_2^2 of the window to rho_star
        total_steps=3_000_000,
    ),
}

SEEDS = (0, 1, 2, 3, 4)
TRAIN_NOISE = 0.05                   # fixed noise level nu, used only when train_noise_max = 0
TRAIN_NOISE_MAX = 0.3                # nu ~ U[0, TRAIN_NOISE_MAX] redrawn at every training episode
EVAL_NOISES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
EVAL_EPISODES = 64                   # M, evaluation episodes per (policy, noise level)


# ---------------------------------------------------------------------------
# Methods (Table "Compared methods" of Section 5; results in Section 5.1 and Appendix F). Every method is trained on the benchmark reward r^b_t, to which the DMDP methods add
# r_tD and -beta * D; PPO, PPO-Lag, Dist-PPO, Dist-PPO-Lag, Implicit-PPO and Implicit-PPO-Lag use the PPO learner, SAC, SAC-Lag,
# Dist-SAC, Dist-SAC-Lag, Implicit-SAC and Implicit-SAC-Lag the SAC learner, and DMDP and DMDP-Lag either one (TrainConfig.learner, run_all.py --learner). With a given learner, the methods differ only in:
#   policy_input = "state" : a_t = pi(o_t)                                  (state feedback)
#   policy_input = "hybrid": a_t = phi(omega_hat_t, o_t)                    (DMDP policy, Appendix E)
#   d_bonus > 0            : threshold reward r_tD = +d if D(rho_bar_t || rho_star) <= D_thr, else 0, added to r^b_t (Appendix E)
#   beta > 0               : reward r^b_t + r_tD - beta * D(rho_bar_t || rho_star)                   (Appendix E)
#   lagrangian             : episodic cost constraint J_c <= c_bar (PPO-Lagrangian, or SAC-Lagrangian with the SAC learner)
#   n_quantiles > 0        : distributional critic: the reward critic (both reward critics with SAC) outputs n_quantiles quantiles
#                            of the return, learned by quantile regression; the policy uses their mean (Distributional and
#                            Implicit PPO and SAC, with and without the cost constraint)
# With policy_input = "hybrid" (DMDP, DMDP-Lag and the Implicit methods), omega_hat_t also holds the window statistics of the pitch reading
# (pitch_weight > 0).
# ---------------------------------------------------------------------------
BETA = 0.5                    # weight of D for DMDP and DMDP-Lag (chosen in the design pilot on seeds 10-12)
# PPO and SAC are trained on the benchmark reward r^b_t = v_t - 0.1 * ||a_t||^2 (cost_penalty = 0 for every method), and PPO-Lag
# and SAC-Lag on r^b_t - lambda * c_t. DMDP is trained on r^b_t + r_tD - beta * D, without a cost constraint. DMDP-Lag
# (DMDP with the cost constraint) is trained on r^b_t + r_tD - beta * D - lambda * c_t with the cost limit COST_LIMIT.
# Distributional PPO (Dist-PPO) and Distributional SAC (Dist-SAC) are PPO and SAC with a distributional critic: the same input o_t,
# the benchmark reward and no cost constraint, but the critic learns the distribution of the return (n_quantiles = N_QUANTILES).
# Implicit PPO (Implicit-PPO) and Implicit SAC (Implicit-SAC) are Dist-PPO and Dist-SAC with the policy input of
# DMDP, phi(omega_hat_t, o_t): distribution feedback with the benchmark reward, without r_tD, D, a target or a cost constraint.
# Dist-PPO-Lag, Dist-SAC-Lag, Implicit-PPO-Lag and Implicit-SAC-Lag are these four baselines trained on r^b_t - lambda * c_t with the
# cost constraint of PPO-Lag and SAC-Lag (cost limit COST_LIMIT); the cost critic learns the expected cost.
METHODS: dict[str, dict[str, Any]] = {
    "PPO":      dict(policy_input="state",  lagrangian=False, beta=0.0,  d_bonus=0.0,     cost_penalty=0.0),
    "PPO-Lag":  dict(policy_input="state",  lagrangian=True,  beta=0.0,  d_bonus=0.0,     cost_penalty=0.0),
    "DMDP":     dict(policy_input="hybrid", lagrangian=False, beta=BETA, d_bonus=D_BONUS, cost_penalty=0.0),
    "DMDP-Lag": dict(policy_input="hybrid", lagrangian=True,  beta=BETA, d_bonus=D_BONUS, cost_penalty=0.0),
    "SAC":      dict(policy_input="state",  lagrangian=False, beta=0.0,  d_bonus=0.0,     cost_penalty=0.0, learner="sac"),
    "SAC-Lag":  dict(policy_input="state",  lagrangian=True,  beta=0.0,  d_bonus=0.0,     cost_penalty=0.0, learner="sac"),
    "Dist-PPO": dict(policy_input="state",  lagrangian=False, beta=0.0,  d_bonus=0.0,     cost_penalty=0.0, learner="ppo",
                     n_quantiles=N_QUANTILES),
    "Dist-SAC": dict(policy_input="state",  lagrangian=False, beta=0.0,  d_bonus=0.0,     cost_penalty=0.0, learner="sac",
                     n_quantiles=N_QUANTILES),
    "Implicit-PPO":     dict(policy_input="hybrid", lagrangian=False, beta=0.0, d_bonus=0.0, cost_penalty=0.0, learner="ppo",
                             n_quantiles=N_QUANTILES),
    "Implicit-SAC":     dict(policy_input="hybrid", lagrangian=False, beta=0.0, d_bonus=0.0, cost_penalty=0.0, learner="sac",
                             n_quantiles=N_QUANTILES),
    "Dist-PPO-Lag":     dict(policy_input="state",  lagrangian=True, beta=0.0,  d_bonus=0.0,  cost_penalty=0.0, learner="ppo",
                             n_quantiles=N_QUANTILES),
    "Dist-SAC-Lag":     dict(policy_input="state",  lagrangian=True, beta=0.0,  d_bonus=0.0,  cost_penalty=0.0, learner="sac",
                             n_quantiles=N_QUANTILES),
    "Implicit-PPO-Lag": dict(policy_input="hybrid", lagrangian=True, beta=0.0,  d_bonus=0.0,  cost_penalty=0.0, learner="ppo",
                             n_quantiles=N_QUANTILES),
    "Implicit-SAC-Lag": dict(policy_input="hybrid", lagrangian=True, beta=0.0,  d_bonus=0.0,  cost_penalty=0.0, learner="sac",
                             n_quantiles=N_QUANTILES),
}

METHOD_LABELS = {
    "PPO": "PPO",
    "PPO-Lag": "PPO-Lag",
    "DMDP": "DMDP",
    "DMDP-Lag": "DMDP-Lag",
    "SAC": "SAC",
    "SAC-Lag": "SAC-Lag",
    "Dist-PPO": "Distributional PPO",
    "Dist-SAC": "Distributional SAC",
    "Implicit-PPO": "Implicit DMDP (PPO)",
    "Implicit-SAC": "Implicit DMDP (SAC)",
    "Dist-PPO-Lag": "Distributional PPO-Lag",
    "Dist-SAC-Lag": "Distributional SAC-Lag",
    "Implicit-PPO-Lag": "Implicit DMDP-Lag (PPO)",
    "Implicit-SAC-Lag": "Implicit DMDP-Lag (SAC)",
}


def method_label(method: str, learner: str = "ppo") -> str:
    """Name of a method in the paper; DMDP and DMDP-Lag, trained with either learner, carry it in parentheses, e.g. "DMDP (PPO)"."""
    if method in ("DMDP", "DMDP-Lag"):
        return f"{METHOD_LABELS[method]} ({learner.upper()})"
    return METHOD_LABELS[method]

_HALF_CHEETAH = ENVIRONMENTS["HalfCheetah"]


@dataclass
class TrainConfig:
    """All settings of one training run (hyperparameters in the Hyperparameters table of Appendix E).

    The field names are stored in ``config.json`` and ``policy.pt`` of every run, so they must
    not be renamed. The defaults describe DMDP on HalfCheetah; use ``make_config`` to build
    the configuration of a given method.
    """

    env: str = "HalfCheetah"
    method: str = "DMDP"
    seed: int = 0
    total_steps: int = _HALF_CHEETAH["total_steps"]
    train_noise: float = TRAIN_NOISE          # fixed nu during training (used only if train_noise_max = 0)
    train_noise_max: float = TRAIN_NOISE_MAX  # if > 0, nu ~ U[0, train_noise_max] per episode
    noise_offset: float = NOISE_OFFSET        # m of the noise mixture 0.5 N(-m, s^2) + 0.5 N(m, s^2) (0: Gaussian noise N(0, s^2))
    noise_sd: float = NOISE_SD                # s of the noise mixture
    window: int = 10                          # n_w, readings in the sliding-window estimator
    num_envs: int = 8                         # parallel environments
    num_steps: int = 256                      # rollout length per environment (8 x 256 = 2048 per update)
    update_epochs: int = 10
    num_minibatches: int = 32
    learning_rate: float = 3e-4               # Adam, annealed linearly to zero
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 64                          # two hidden tanh layers of this width
    init_log_std: float = INIT_LOG_STD        # initial log standard deviation of the actions (std 0.37; the PPO default is 0); PPO only
    cost_limit: float = COST_LIMIT            # c_bar
    cost_penalty: float = 0.0                 # kappa of the cost penalty in the reward r^b_t - (kappa * c_t)^2 (COST_PENALTY: pilot only)
    lagrangian_lr: float = 0.035              # Adam learning rate of the Lagrange multiplier
    reward_normalization: bool = True         # divide the training reward by the running std of its discounted return
    policy_input: str = "hybrid"              # "state" or "hybrid"
    lagrangian: bool = False                  # DMDP has no cost constraint (METHODS)
    beta: float = BETA                        # weight of D (DMDP and DMDP-Lag)
    dissimilarity: str = _HALF_CHEETAH["dissimilarity"]    # "empirical_w2", or "gaussian_w2" (closed form, Gaussian rho_star)
    d_bonus: float = D_BONUS                  # d of the threshold reward r_tD = +d if D <= D_thr, else 0
    d_threshold: float = D_THRESHOLD          # D_thr of the threshold reward
    target_family: str = _HALF_CHEETAH["target_family"]    # family of the velocity target; mu_star and sd_star are its moments
    v_limit: float = _HALF_CHEETAH["v_limit"]              # velocity threshold of the simulator (V_LIMIT)
    mu_star: float = _HALF_CHEETAH["mu_star"]              # mean of the velocity target
    sd_star: float = _HALF_CHEETAH["sd_star"]              # standard deviation of the velocity target
    env_id: str = _HALF_CHEETAH["env_id"]
    vel_index: int = _HALF_CHEETAH["vel_index"]
    pitch_index: int = _HALF_CHEETAH["pitch_index"]         # pitch reading o_t[pitch_index]; true pitch qpos[pitch_index + 1]
    pitch_mu_star: float = 0.0                              # mean of the pitch target
    pitch_sd_star: float = PITCH_HALF_WIDTH / 3 ** 0.5      # standard deviation of the pitch target U[-0.2, 0.2]
    velocity_weight: float = 1.0                            # weight of the velocity term of D
    pitch_weight: float = 1.0                               # weight of the pitch term of D; 0: velocity only (earlier designs)
    learner: str = "ppo"                                    # "ppo" (train.py) or "sac" (sac.py)
    sac_hidden: int = 256                                   # SAC: two hidden ReLU layers of this width (actor and critics)
    sac_learning_rate: float = 3e-4                         # SAC: Adam learning rate of the actor, critics and temperature
    sac_batch_size: int = 256                               # SAC: minibatch size
    sac_buffer_size: int = 1_000_000                        # SAC: replay buffer capacity
    sac_tau: float = 0.005                                  # SAC: Polyak averaging coefficient of the target critics
    sac_learning_starts: int = 5000                         # SAC: steps with uniformly random actions before the first update
    sac_lambda_interval: int = 2048                         # SAC: steps between two updates of the Lagrange multiplier
    n_quantiles: int = 0                                    # quantiles of the return learned by the reward critics (Distributional
                                                            # and Implicit PPO and SAC); 0: the expected return

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def config_from_dict(stored: dict[str, Any]) -> TrainConfig:
    """``TrainConfig`` of a stored run (``config.json`` or the ``cfg`` entry of ``policy.pt``).

    Runs stored before the cost penalty was introduced have no ``cost_penalty`` entry; they were trained with the
    benchmark reward, i.e., kappa = 0, so a missing entry is read as 0 (the current default; it was COST_PENALTY = 2 while
    the runs of the design pilot were stored). Likewise, runs
    stored before the threshold reward r_tD was introduced have no ``d_bonus`` entry and are read with d = 0. Runs of the
    earlier designs with r_tD = +-beta (``d_reward="threshold"``) or r_tD = +-d with a quantile threshold (``d_quantile``)
    cannot be represented and raise an error unless they did not receive r_tD. Runs stored before the noise mixture was
    introduced have no ``noise_offset`` and ``noise_sd`` entries; they were trained and evaluated with Gaussian noise
    N(0, I), i.e., m = 0 and s = 1. D_thr acts on training only through r_tD, so runs without it (d = 0), such as
    the baselines, which were stored with the earlier D_thr = 0.14, are read with the current D_THRESHOLD.
    """
    stored = dict(stored)
    # SAC runs started while a separate learning rate of the multiplier was being piloted store it; it has no effect
    # without the constraint, and constrained runs with a rate other than lagrangian_lr cannot be represented
    if "sac_lagrangian_lr" in stored:
        if stored.pop("sac_lagrangian_lr") != stored.get("lagrangian_lr") and stored.get("lagrangian"):
            raise ValueError("SAC pilot run with a separate learning rate of the Lagrange multiplier")
    if stored.pop("d_reward", "penalty") == "threshold":
        raise ValueError("run of the earlier design in which r_tD = +-beta replaced the penalty -beta * D of DMDP")
    if "d_quantile" in stored:
        stored.pop("d_quantile")
        if stored.get("d_bonus", 0.0) > 0.0:
            raise ValueError("run of the earlier design with r_tD = +-d and a quantile threshold")
    # earlier designs regulated the velocity only and trained on the unscaled reward; earlier runs used Gaussian noise
    legacy = {"cost_penalty": 0.0, "d_bonus": 0.0, "pitch_weight": 0.0, "reward_normalization": False, "init_log_std": 0.0,
              "noise_offset": 0.0, "noise_sd": 1.0}
    stored = {**legacy, **stored}
    if not stored["d_bonus"]:
        stored["d_threshold"] = D_THRESHOLD
    return TrainConfig(**stored)


def make_config(env: str, method: str, seed: int, **overrides: Any) -> TrainConfig:
    """Configuration of one run of ``method`` on ``env`` with ``seed``.

    Environment settings come from ``ENVIRONMENTS``, method settings from ``METHODS``;
    keyword arguments override any ``TrainConfig`` field (e.g. ``total_steps=200_000``).
    """
    spec = ENVIRONMENTS[env]
    meth = METHODS[method]
    learner = overrides.pop("learner", meth.get("learner", "ppo"))
    if learner not in LEARNERS:
        raise ValueError(f"learner must be one of {LEARNERS}, got {learner!r}")
    if "learner" in meth and meth["learner"] != learner:
        raise ValueError(f"{method} uses the {meth['learner']} learner")
    if learner == "ppo" and method.startswith("SAC") or learner == "sac" and method.startswith("PPO"):
        raise ValueError(f"{method} cannot be trained with the {learner} learner")
    cfg = TrainConfig(
        env=env, method=method, seed=seed,
        total_steps=spec["total_steps"], env_id=spec["env_id"], vel_index=spec["vel_index"],
        v_limit=spec["v_limit"], mu_star=spec["mu_star"], sd_star=spec["sd_star"],
        target_family=spec["target_family"], dissimilarity=spec["dissimilarity"],
        policy_input=meth["policy_input"], lagrangian=meth["lagrangian"], beta=meth["beta"], d_bonus=meth["d_bonus"],
        cost_penalty=meth["cost_penalty"], n_quantiles=meth.get("n_quantiles", 0),
        pitch_index=spec["pitch_index"], learner=learner,
    )
    if learner == "sac":
        cfg.total_steps, cfg.num_envs = SAC_TOTAL_STEPS, SAC_NUM_ENVS
    valid = {f.name for f in fields(TrainConfig)}
    for key, value in overrides.items():
        if key not in valid:
            raise ValueError(f"unknown TrainConfig field: {key!r}")
        setattr(cfg, key, value)
    return cfg
