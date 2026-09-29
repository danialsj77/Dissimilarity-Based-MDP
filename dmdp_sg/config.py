"""Experiment configuration (Section 7.1 and Appendix D of the paper).

Notation, as in the paper:

    s_t          observation vector of the benchmark
    o_t          noisy sensor reading, o_t = s_t + nu * xi_t with xi_t ~ N(0, I)   (Section 6.2)
    v_t          true forward velocity, the regulated scalar feature              (Section 6.1)
    o^v_t        noisy reading of the forward velocity, o_t[vel_index]
    rho_k        distribution of v_k at step k across evaluation episodes
    rho_omega    Gaussian family N(mu, sd^2) with parameter omega = (mu, sd)
    rho_star     target distribution N(mu_star, sd_star^2)                         (Section 6.1)
    omega_hat_t  window mean and std of the last n_w readings o^v_t (policy input, Section 6.2)
    D            training dissimilarity, squared 2-Wasserstein distance between Gaussians
    beta         weight of D in the shaped reward r_t - beta * D                  (Section 6.3)
    c_bar        episodic cost limit (COST_LIMIT)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

from scipy.stats import norm

EPISODE_LENGTH = 1000   # H, steps per episode (HalfCheetah never terminates early)
COST_LIMIT = 25.0       # c_bar, standard episodic cost limit of Safety-Gymnasium / OmniSafe / SafePO


def target_mean(v_limit: float, sd_star: float, cost_limit: float = COST_LIMIT,
                horizon: int = EPISODE_LENGTH) -> float:
    """Mean of the task-derived target distribution rho_star (Section 6.1).

    rho_star = N(mu_star, sd_star^2) is the fastest Gaussian velocity distribution with spread
    sd_star whose expected episodic cost H * P(v > v_limit) equals the cost limit c_bar:

        mu_star = v_limit - z_{1 - c_bar / H} * sd_star,

    where z_q is the q-quantile of the standard normal distribution.
    """
    return v_limit - norm.ppf(1.0 - cost_limit / horizon) * sd_star


# ---------------------------------------------------------------------------
# Environment (Safety-Gymnasium velocity task with the v1 threshold of the benchmark)
# ---------------------------------------------------------------------------
ENVIRONMENTS: dict[str, dict[str, Any]] = {
    "HalfCheetah": dict(
        env_id="SafetyHalfCheetahVelocityGymnasium-v1",
        vel_index=8,             # index of the forward velocity of the torso in the observation vector
        v_limit=3.2096,          # velocity threshold; cost c_t = 1[v_t > v_limit]
        sd_star=0.50,            # standard deviation of rho_star
        mu_star=target_mean(3.2096, 0.50),   # = 2.2296, mean of rho_star
        mu_star_slow=1.00,       # slow target: stability alone does not imply reward (Section 7.2)
        total_steps=3_000_000,
    ),
}

SEEDS = (0, 1, 2, 3, 4)
TRAIN_NOISE = 0.05                   # fixed noise level nu, used only when train_noise_max = 0
TRAIN_NOISE_MAX = 0.3                # nu ~ U[0, TRAIN_NOISE_MAX] redrawn at every training episode
EVAL_NOISES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
EVAL_EPISODES = 64                   # M, evaluation episodes per (policy, noise level)


# ---------------------------------------------------------------------------
# Methods (Section 7.1). All methods share the PPO learner and differ only in:
#   policy_input = "state" : a_t = pi(o_t)                                  (state feedback)
#   policy_input = "window": a_t = pi(o_t, o^v_{t-n_w+1}, ..., o^v_t)       (raw window, same information)
#   policy_input = "hybrid": a_t = phi(omega_hat_t, o_t)                    (DMDP policy, Section 6.2)
#   beta > 0               : shaped reward r_t - beta * D(rho_{omega_bar_t} || rho_star)  (Section 6.3)
#   lagrangian             : episodic cost constraint J_c <= c_bar (PPO-Lagrangian)
#   target = "slow"        : rho_star = N(mu_star_slow, sd_star^2) instead of the task-derived target
# ---------------------------------------------------------------------------
BETA = 1.0
METHODS: dict[str, dict[str, Any]] = {
    "PPO":        dict(policy_input="state",  lagrangian=False, beta=0.0),
    "PPO-Lag":    dict(policy_input="state",  lagrangian=True,  beta=0.0),
    "Window-Lag": dict(policy_input="window", lagrangian=True,  beta=0.0),
    "DMDP":       dict(policy_input="hybrid", lagrangian=True,  beta=BETA),
    "DMDP-noD":   dict(policy_input="hybrid", lagrangian=True,  beta=0.0),
    "State-D":    dict(policy_input="state",  lagrangian=True,  beta=BETA),
    "DMDP-slow":  dict(policy_input="hybrid", lagrangian=True,  beta=BETA, target="slow"),
}

METHOD_LABELS = {
    "PPO": "PPO",
    "PPO-Lag": "PPO-Lag",
    "Window-Lag": "PPO-Lag (window input)",
    "DMDP": "DMDP",
    "DMDP-noD": r"DMDP w/o $\mathbb{D}$",
    "State-D": r"DMDP w/o $\hat{\omega}_t$",
    "DMDP-slow": r"DMDP, slow $\rho_\star$",
}

_HALF_CHEETAH = ENVIRONMENTS["HalfCheetah"]


@dataclass
class TrainConfig:
    """All settings of one training run (hyperparameters in Appendix D).

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
    cost_limit: float = COST_LIMIT            # c_bar
    lagrangian_lr: float = 0.035              # Adam learning rate of the Lagrange multiplier
    policy_input: str = "hybrid"              # "state", "window" or "hybrid"
    lagrangian: bool = True
    beta: float = BETA
    v_limit: float = _HALF_CHEETAH["v_limit"]
    mu_star: float = _HALF_CHEETAH["mu_star"]
    sd_star: float = _HALF_CHEETAH["sd_star"]
    env_id: str = _HALF_CHEETAH["env_id"]
    vel_index: int = _HALF_CHEETAH["vel_index"]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_config(env: str, method: str, seed: int, **overrides: Any) -> TrainConfig:
    """Configuration of one run of ``method`` on ``env`` with ``seed``.

    Environment settings come from ``ENVIRONMENTS``, method settings from ``METHODS``;
    keyword arguments override any ``TrainConfig`` field (e.g. ``total_steps=200_000``).
    """
    spec = ENVIRONMENTS[env]
    meth = METHODS[method]
    cfg = TrainConfig(
        env=env, method=method, seed=seed,
        total_steps=spec["total_steps"], env_id=spec["env_id"], vel_index=spec["vel_index"],
        v_limit=spec["v_limit"], mu_star=spec["mu_star"], sd_star=spec["sd_star"],
        policy_input=meth["policy_input"], lagrangian=meth["lagrangian"], beta=meth["beta"],
    )
    if meth.get("target") == "slow":
        cfg.mu_star = spec["mu_star_slow"]
    valid = {f.name for f in fields(TrainConfig)}
    for key, value in overrides.items():
        if key not in valid:
            raise ValueError(f"unknown TrainConfig field: {key!r}")
        setattr(cfg, key, value)
    return cfg
