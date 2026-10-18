"""Calibration and test sets of the scalar example: order statistic, infinite scores, residual radius, capped decrease."""

import numpy as np
import pytest

from dmdp_toy import certify as C
from dmdp_toy import system as S
from dmdp_toy.stages import calibrate


def test_order_statistic_confidence():
    bar, conf = C.order_statistic(np.linspace(0.1, 0.2, 49))
    assert bar == 0.2 and abs(conf - 0.99427) < 1e-5


def test_infinite_score_propagates():
    bar, _ = C.order_statistic([0.1, np.inf, 0.2])
    assert np.isinf(bar) and np.isinf(C.residual_radius(bar, 0.75))
    assert C.fmt(bar) == r"$\infty$"


def test_residual_radius():
    assert np.isclose(C.residual_radius(0.12, 0.75), 0.4)
    assert np.isclose(C.residual_radius(0.03, 0.25), np.sqrt(0.12))


def test_capped_decrease_error_bound():
    rng = np.random.default_rng(0)
    d = rng.uniform(0, 4, 5000)
    true_dec = -0.4 * d + 0.05 * rng.standard_normal(5000)
    d_hat = true_dec + 0.1 * rng.standard_normal(5000)
    capped = C.capped_decrease(d_hat, d, 0.25)
    e_n = max(0.0, np.max(d_hat + 0.25 * d))
    assert np.all(capped <= -0.25 * d)
    assert np.max(np.abs(capped - true_dec)) <= np.max(np.abs(d_hat - true_dec)) + e_n + 1e-12


def test_test_sets():
    mu, vs = C.test_points()
    assert mu.size == 11_822 and mu[-1] == S.MU_STAR and vs[-1] == S.VS_STAR
    assert mu.min() >= S.MU_LO and vs.min() >= S.VS_LO and vs.max() <= S.VS_HI
    cols = C.model_test_points()
    assert len(cols) == 4 and cols[0].size == 20_016
    assert all(np.isin([lo, hi], c).all() for c, lo, hi in zip(cols, S.LO, S.HI))


def test_calibration_is_the_plain_maximum():
    """Delta_bar = max of the scores as specified: no numerical margin, monotone in the scores."""
    scores = [0.05] * 48
    for top in (0.05, 0.058, 0.081):
        cal = calibrate(scores + [top])
        assert cal["Delta_bar"] == top and cal["n_calib"] == 49 and abs(cal["confidence"] - 0.99427) < 1e-5
    assert set(calibrate(scores)) == {"Delta_bar", "confidence", "n_calib"}
    assert np.isinf(calibrate(scores + [np.inf])["Delta_bar"])


def test_independent_check_set():
    mu, vs = C.test_points()
    mi, vi_ = C.independent_points()
    assert mi.size == 80_000 and mi.min() >= S.MU_LO and mi.max() <= S.MU_HI
    assert vi_.min() >= S.VS_LO and vi_.max() <= S.VS_HI
    r = np.hypot(mi - S.MU_STAR, vi_ - S.VS_STAR)
    assert r.min() < 2e-5 and r[r > 0].min() < 1e-3 < np.hypot(mu - S.MU_STAR, vs - S.VS_STAR)[:-1].min()
    coarse = set(zip(mu.round(12), vs.round(12)))
    assert not any(p in coarse for p in zip(mi.round(12), vi_.round(12)))       # not nested in T_omega
    assert C.relative_change(1.04, 1.0) == pytest.approx(0.04) and C.relative_change(0.9, 1.0) == pytest.approx(-0.1)
    assert C.relative_change(0.0, 0.0) == 0.0 and np.isinf(C.relative_change(1.0, 0.0))


def test_number_format():
    assert C.fmt(0.1234) == "$0.12$" and C.fmt(1.234) == "$1.2$" and C.fmt(0.0012) == r"$1.2\times10^{-3}$"
    assert C.num(0.0996) == "0.10" and C.num(9.96) == "10" and C.num(99.6) == "100" and C.num(0.00996) == "0.010"
    assert C.num(0.0994) == "0.099" and C.num(0.012345) == "0.012" and C.num(123.4) == "120"
