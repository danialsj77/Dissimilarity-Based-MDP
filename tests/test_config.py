"""Configuration: target distribution (Section 6.1) and method settings (Section 7.1)."""

import pytest
from scipy.stats import norm

from dmdp_sg.config import (COST_LIMIT, ENVIRONMENTS, EPISODE_LENGTH, METHODS, TrainConfig, make_config,
                            target_mean)


def test_target_mean_half_cheetah():
    spec = ENVIRONMENTS["HalfCheetah"]
    # mu_star = v_lim - z_{1 - c_bar/H} * sd_star = 3.2096 - 1.95996 * 0.5
    assert round(spec["mu_star"], 4) == 2.2296
    assert spec["mu_star"] == target_mean(spec["v_limit"], spec["sd_star"], COST_LIMIT, EPISODE_LENGTH)


def test_expected_cost_under_target_equals_cost_limit():
    spec = ENVIRONMENTS["HalfCheetah"]
    expected_cost = EPISODE_LENGTH * norm.sf(spec["v_limit"], spec["mu_star"], spec["sd_star"])
    assert expected_cost == pytest.approx(COST_LIMIT)


@pytest.mark.parametrize("method", list(METHODS))
def test_make_config_applies_method_settings(method):
    cfg = make_config("HalfCheetah", method, 3)
    meth = METHODS[method]
    assert (cfg.policy_input, cfg.lagrangian, cfg.beta) == (meth["policy_input"], meth["lagrangian"], meth["beta"])
    assert cfg.seed == 3 and cfg.total_steps == 3_000_000
    expected_mu = ENVIRONMENTS["HalfCheetah"]["mu_star_slow" if method == "DMDP-slow" else "mu_star"]
    assert cfg.mu_star == expected_mu


def test_make_config_overrides():
    cfg = make_config("HalfCheetah", "DMDP", 0, total_steps=1000, window=5)
    assert cfg.total_steps == 1000 and cfg.window == 5
    with pytest.raises(ValueError):
        make_config("HalfCheetah", "DMDP", 0, windw=5)


def test_default_config_is_dmdp_on_half_cheetah():
    assert TrainConfig() == make_config("HalfCheetah", "DMDP", 0)
