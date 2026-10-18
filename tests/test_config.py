"""Configuration: target distribution (Section 6.1) and method settings (Section 7.1)."""

import pytest
from scipy.stats import norm

from dmdp_sg.config import (BENCHMARK_COST_LIMIT, BETA, COST_LIMIT, COST_PENALTY, D_THRESHOLD, ENVIRONMENTS, EPISODE_LENGTH, EVAL_NOISES,
                            METHOD_LABELS, METHODS, N_QUANTILES, PITCH_HALF_WIDTH, TARGET_BAND, V_LIMIT, TrainConfig, band_target,
                            config_from_dict, make_config, method_label, target_mean, uniform_target)
from dmdp_sg.targets import Target


def test_velocity_and_pitch_targets_half_cheetah():
    spec = ENVIRONMENTS["HalfCheetah"]
    # velocity target U[0.70 v_lim, 0.95 v_lim]: fast, and safe with a margin of 5% below the threshold
    target = Target(spec["target_family"], spec["mu_star"], spec["sd_star"])
    assert TARGET_BAND == (0.70, 0.95) and target.label() == "U[2.25, 3.05]"
    assert spec["v_limit"] == V_LIMIT == 3.2096 and make_config("HalfCheetah", "DMDP", 0).v_limit == V_LIMIT
    assert (spec["mu_star"], spec["sd_star"]) == band_target(V_LIMIT) == band_target(spec["v_limit"])
    assert target.quantile(1.0) == pytest.approx(0.95 * spec["v_limit"]) and target.quantile(0.0) == pytest.approx(0.70 * spec["v_limit"])
    assert EPISODE_LENGTH * target.tail(spec["v_limit"]) == 0.0           # no cost under the target
    # pitch target U[-0.2, 0.2] rad
    cfg = make_config("HalfCheetah", "DMDP", 0)
    pitch = Target("uniform", cfg.pitch_mu_star, cfg.pitch_sd_star)
    assert PITCH_HALF_WIDTH == 0.2 and pitch.quantile(1.0) == pytest.approx(0.2) and pitch.quantile(0.0) == pytest.approx(-0.2)
    assert (cfg.velocity_weight, cfg.pitch_weight, cfg.pitch_index) == (1.0, 1.0, 1)


def test_earlier_cost_derived_targets():
    spec = ENVIRONMENTS["HalfCheetah"]
    # the target of the earlier designs: U[0, v_max] with v_max = v_lim / (1 - c_bar_0 / H) = 3.2096 / 0.975
    mu, sd = uniform_target(spec["v_limit"], BENCHMARK_COST_LIMIT, EPISODE_LENGTH)
    target = Target("uniform", mu, sd)
    assert target.label() == "U[0.00, 3.29]" and mu == pytest.approx(target_mean(spec["v_limit"], sd, family="uniform"))
    assert EPISODE_LENGTH * target.tail(spec["v_limit"]) == pytest.approx(BENCHMARK_COST_LIMIT)
    assert COST_LIMIT == 10.0       # c_bar of the constrained methods
    # the Gaussian rule of the earlier design: mu_star = v_lim - z_{1 - c_bar/H} sd_star
    assert EPISODE_LENGTH * norm.sf(spec["v_limit"], target_mean(spec["v_limit"], 0.5), 0.5) == pytest.approx(BENCHMARK_COST_LIMIT)


@pytest.mark.parametrize("method", list(METHODS))
def test_make_config_applies_method_settings(method):
    cfg = make_config("HalfCheetah", method, 3)
    meth = METHODS[method]
    assert (cfg.policy_input, cfg.lagrangian, cfg.beta) == (meth["policy_input"], meth["lagrangian"], meth["beta"])
    learner = meth.get("learner", "ppo")
    assert cfg.seed == 3 and cfg.total_steps == (1_000_000 if learner == "sac" else 3_000_000)
    assert cfg.learner == learner and learner == ("sac" if "SAC" in method else "ppo")
    assert cfg.n_quantiles == (N_QUANTILES if method.startswith(("Dist-", "Implicit-")) else 0)
    assert cfg.mu_star == ENVIRONMENTS["HalfCheetah"]["mu_star"]
    assert (cfg.target_family, cfg.dissimilarity) == ("uniform", "empirical_w2")
    # every method is trained on the benchmark reward v_t - 0.1 ||a_t||^2 (the DMDP methods add r_tD and -beta * D);
    # the cost penalty (2 c_t)^2 was used only in the design pilot
    assert cfg.cost_penalty == 0.0 and COST_PENALTY == 2.0


def test_make_config_overrides():
    cfg = make_config("HalfCheetah", "DMDP", 0, total_steps=1000, window=5)
    assert cfg.total_steps == 1000 and cfg.window == 5
    with pytest.raises(ValueError):
        make_config("HalfCheetah", "DMDP", 0, windw=5)


def test_default_config_is_dmdp_on_half_cheetah():
    assert TrainConfig() == make_config("HalfCheetah", "DMDP", 0)


def test_methods_of_the_study():
    # DMDP methods with either learner; Distributional PPO and SAC: PPO and SAC with a distributional critic; Implicit
    # PPO and SAC: the distributional baselines with the policy input of DMDP
    assert list(METHODS) == ["PPO", "PPO-Lag", "DMDP", "DMDP-Lag", "SAC", "SAC-Lag", "Dist-PPO", "Dist-SAC", "Implicit-PPO", "Implicit-SAC",
                             "Dist-PPO-Lag", "Dist-SAC-Lag", "Implicit-PPO-Lag", "Implicit-SAC-Lag"]
    assert list(METHOD_LABELS) == list(METHODS) and METHOD_LABELS["Dist-SAC"] == "Distributional SAC" and N_QUANTILES == 32
    assert METHOD_LABELS["Implicit-PPO"] == "Implicit DMDP (PPO)" and METHOD_LABELS["Implicit-SAC"] == "Implicit DMDP (SAC)"
    assert METHOD_LABELS["DMDP-Lag"] == "DMDP-Lag" and METHOD_LABELS["Dist-PPO-Lag"] == "Distributional PPO-Lag"
    assert METHOD_LABELS["Implicit-SAC-Lag"] == "Implicit DMDP-Lag (SAC)"
    assert method_label("DMDP", "ppo") == "DMDP (PPO)" and method_label("DMDP-Lag", "sac") == "DMDP-Lag (SAC)"
    assert method_label("Dist-SAC", "sac") == "Distributional SAC"
    assert BETA == 0.5 and COST_LIMIT == 10.0 and TrainConfig().init_log_std == -1.0    # design pilot; initial action std 0.37
    # every method is trained on the benchmark reward (no cost penalty)
    for method, spec in METHODS.items():
        assert make_config("HalfCheetah", method, 0).cost_penalty == spec["cost_penalty"] == 0.0


@pytest.mark.parametrize("method, base", [("Dist-PPO", "PPO"), ("Dist-SAC", "SAC")])
def test_distributional_baselines_differ_from_ppo_and_sac_only_in_the_critic(method, base):
    # benchmark reward (no r_tD, no D, no cost penalty), input o_t, no constraint; the critic learns quantiles of the return
    cfg, ref = make_config("HalfCheetah", method, 1), make_config("HalfCheetah", base, 1)
    assert {k for k, v in cfg.to_dict().items() if v != ref.to_dict()[k]} == {"method", "n_quantiles"}
    assert (cfg.policy_input, cfg.d_bonus, cfg.beta, cfg.cost_penalty, cfg.lagrangian) == ("state", 0.0, 0.0, 0.0, False)
    with pytest.raises(ValueError):
        make_config("HalfCheetah", method, 0, learner="sac" if base == "PPO" else "ppo")
    stored = {k: v for k, v in ref.to_dict().items() if k != "n_quantiles"}     # runs stored before the field existed
    assert config_from_dict(stored) == ref


@pytest.mark.parametrize("method, base", [("Dist-PPO-Lag", "Dist-PPO"), ("Dist-SAC-Lag", "Dist-SAC"),
                                          ("Implicit-PPO-Lag", "Implicit-PPO"), ("Implicit-SAC-Lag", "Implicit-SAC")])
def test_lagrangian_distributional_baselines_add_only_the_cost_constraint(method, base):
    # the distributional and implicit baselines with the cost constraint of PPO-Lag and SAC-Lag (reward r^b_t - lambda c_t)
    cfg, ref = make_config("HalfCheetah", method, 1), make_config("HalfCheetah", base, 1)
    assert {k for k, v in cfg.to_dict().items() if v != ref.to_dict()[k]} == {"method", "lagrangian"}
    assert cfg.lagrangian and cfg.cost_limit == COST_LIMIT and cfg.n_quantiles == N_QUANTILES
    lag = make_config("HalfCheetah", "PPO-Lag" if cfg.learner == "ppo" else "SAC-Lag", 1)
    assert {k for k, v in cfg.to_dict().items() if v != lag.to_dict()[k]} <= {"method", "n_quantiles", "policy_input"}


@pytest.mark.parametrize("method, dist, dmdp", [("Implicit-PPO", "Dist-PPO", "DMDP"), ("Implicit-SAC", "Dist-SAC", "DMDP")])
def test_distribution_feedback_baselines(method, dist, dmdp):
    # the policy input of DMDP, phi(omega_hat_t, o_t), with the reward and the critic of the distributional baselines: no r_tD, no D,
    # no cost penalty and no constraint
    learner = "sac" if method.endswith("SAC") else "ppo"
    cfg, ref = make_config("HalfCheetah", method, 2), make_config("HalfCheetah", dist, 2)
    assert {k for k, v in cfg.to_dict().items() if v != ref.to_dict()[k]} == {"method", "policy_input"}
    assert (cfg.policy_input, cfg.d_bonus, cfg.beta, cfg.cost_penalty, cfg.lagrangian) == ("hybrid", 0.0, 0.0, 0.0, False)
    assert cfg.n_quantiles == N_QUANTILES and cfg.learner == learner
    # with DMDP, it shares the policy input (window statistics of the velocity and the pitch) and differs in the reward and critic
    other = make_config("HalfCheetah", dmdp, 2, learner=learner)
    assert {k for k, v in cfg.to_dict().items() if v != other.to_dict()[k]} == {"method", "beta", "d_bonus", "n_quantiles"}
    with pytest.raises(ValueError):
        make_config("HalfCheetah", method, 0, learner="ppo" if learner == "sac" else "sac")


def test_noise_model_and_evaluation_levels():
    # o_t = y_t + nu * xi_t with xi_t from the bimodal mixture 0.5 N(-0.8, 0.36) + 0.5 N(0.8, 0.36); evaluation at nu <= 0.5
    assert EVAL_NOISES == (0.0, 0.1, 0.2, 0.3, 0.4, 0.5) and TrainConfig().train_noise_max == 0.3
    for method in METHODS:
        cfg = make_config("HalfCheetah", method, 0)
        assert (cfg.noise_offset, cfg.noise_sd) == (0.8, 0.6)
        stored = {k: v for k, v in cfg.to_dict().items() if k not in ("noise_offset", "noise_sd")}
        legacy = config_from_dict(stored)          # runs stored before the mixture used Gaussian noise N(0, I)
        assert (legacy.noise_offset, legacy.noise_sd) == (0.0, 1.0) and legacy != cfg


def test_legacy_configs_and_threshold_reward_settings():
    for method in METHODS:
        cfg = make_config("HalfCheetah", method, 0)
        # r_tD = +10 (or 0) and the dense term -beta * D only for DMDP and DMDP-Lag
        assert cfg.d_bonus == (10.0 if method.startswith("DMDP") else 0.0) and cfg.d_threshold == D_THRESHOLD == 0.1
        # D_thr has no effect without r_tD: a baseline stored with the earlier D_thr = 0.14 is the same configuration
        if not cfg.d_bonus:
            assert config_from_dict({**cfg.to_dict(), "d_threshold": 0.14}) == cfg
        else:
            assert config_from_dict({**cfg.to_dict(), "d_threshold": 0.14}) != cfg
        assert cfg.beta == (BETA if method.startswith("DMDP") else 0.0)
        earlier = {k: v for k, v in cfg.to_dict().items()
                   if not k.startswith("pitch") and k not in ("velocity_weight", "reward_normalization", "init_log_std")}
        loaded = config_from_dict(earlier)        # runs stored before the pitch became a regulated feature
        assert loaded.pitch_weight == 0.0 and not loaded.reward_normalization and cfg.reward_normalization
        assert loaded.init_log_std == 0.0 and cfg.init_log_std == -1.0     # earlier runs used the PPO default std 1
        assert config_from_dict(cfg.to_dict()) == cfg
        stored = cfg.to_dict()
        del stored["d_bonus"], stored["d_threshold"]
        legacy = config_from_dict(stored)                          # runs stored before the threshold reward
        assert legacy.d_bonus == 0.0 and (legacy != cfg) == (cfg.d_bonus > 0)
        earlier = {**cfg.to_dict(), "d_quantile": 0.5}              # earlier design: r_tD = +-d with a quantile threshold
        if cfg.d_bonus > 0:
            with pytest.raises(ValueError):
                config_from_dict(earlier)
        else:
            assert config_from_dict(earlier) == cfg
        stored["d_reward"] = "threshold"                           # intermediate design: r_tD replaced -beta * D
        with pytest.raises(ValueError):
            config_from_dict(stored)
