"""Evaluation metrics: stability curve D(rho_k || rho_hat) and its sampling floor (Section 7.1, Appendix D)."""

import numpy as np
import pytest

from dmdp_sg.metrics import distance_to_target, estimate_rho_hat, sampling_floor, split_halves, stability_curve

EPISODES, STEPS, K0, BLOCK = 64, 1000, 500, 10


def _late_mean(values):
    centers, D, _ = stability_curve(values, k0=K0, block=BLOCK)
    return float(np.nanmean(D[centers >= K0]))


def test_split_halves_is_a_partition():
    a, b = split_halves(EPISODES)
    assert len(a) == len(b) == EPISODES // 2
    assert sorted(np.concatenate([a, b])) == list(range(EPISODES))


def test_stability_curve_shapes_and_survival():
    values = np.random.default_rng(0).normal(size=(EPISODES, STEPS))
    values[:8, 900:] = np.nan                                  # eight episodes end at step 900
    centers, D, survival = stability_curve(values, k0=K0, block=BLOCK)
    assert centers.shape == D.shape == (STEPS // BLOCK,)
    assert centers[0] == BLOCK / 2
    assert survival[0] == 1.0 and survival[-1] == pytest.approx(56 / 64)
    assert np.all(np.isfinite(D))


def test_stationary_iid_process_is_at_its_sampling_floor():
    values = np.random.default_rng(1).normal(size=(EPISODES, STEPS))
    floor = sampling_floor(values, K0, BLOCK, np.random.default_rng(2))
    assert 0.85 < _late_mean(values) / floor < 1.15


def test_floor_keeps_differences_between_episodes():
    rng = np.random.default_rng(3)
    values = rng.normal(0.0, 0.5, (EPISODES, 1)) + rng.normal(size=(EPISODES, STEPS))   # episode-specific means
    floor = sampling_floor(values, K0, BLOCK, np.random.default_rng(4))
    assert 0.85 < _late_mean(values) / floor < 1.15


def test_transient_is_detected():
    rng = np.random.default_rng(5)
    values = rng.normal(size=(EPISODES, STEPS)) + 3.0 * np.exp(-np.arange(STEPS) / 50.0)
    centers, D, _ = stability_curve(values, k0=K0, block=BLOCK)
    floor = sampling_floor(values, K0, BLOCK, np.random.default_rng(6))
    assert np.nanmean(D[centers < 200]) > 5 * floor
    assert _late_mean(values) / floor < 1.15


def test_sampling_floor_is_reproducible():
    values = np.random.default_rng(7).normal(size=(EPISODES, STEPS))
    assert sampling_floor(values, K0, BLOCK, np.random.default_rng(0)) == \
        sampling_floor(values, K0, BLOCK, np.random.default_rng(0))


def test_estimate_rho_hat_and_distance_to_target():
    values = np.random.default_rng(8).normal(2.23, 0.5, (EPISODES, STEPS))
    values[0, 600:] = np.nan
    late = estimate_rho_hat(values, K0)
    assert late.size == EPISODES * (STEPS - K0) - 400 and np.all(np.isfinite(late))
    assert distance_to_target(values, 2.23, 0.5, k0=K0) < 0.02
    assert distance_to_target(values, 3.23, 0.5, k0=K0) == pytest.approx(1.0, abs=0.02)


def test_wasserstein_1_matches_scipy():
    from scipy.stats import wasserstein_distance
    from dmdp_sg.metrics import Reference, wasserstein_1
    rng = np.random.default_rng(9)
    for n, m in [(1, 1), (7, 3), (320, 16000), (50, 50)]:
        a, b = rng.standard_t(3, n), np.round(rng.normal(0.5, 2.0, m), 1)     # rounding creates ties
        assert wasserstein_1(a, Reference(b)) == pytest.approx(wasserstein_distance(a, b), rel=1e-10, abs=1e-12)


def _gait(phases, rng):
    k = np.arange(STEPS)
    return np.sin(2 * np.pi * k / 4.3 + phases[:, None]) + 0.3 * rng.normal(size=(EPISODES, STEPS))


def test_oscillation_with_independent_phases_is_at_its_floor():
    rng = np.random.default_rng(10)
    values = _gait(rng.uniform(0, 2 * np.pi, EPISODES), rng)
    assert 0.9 < _late_mean(values) / sampling_floor(values, K0, BLOCK, np.random.default_rng(11)) < 1.1


def test_phase_locked_oscillation_is_above_its_floor():
    rng = np.random.default_rng(12)
    values = _gait(np.zeros(EPISODES), rng)                    # all episodes oscillate in phase: rho_k is periodic
    assert _late_mean(values) / sampling_floor(values, K0, BLOCK, np.random.default_rng(13)) > 1.5
