"""Evaluation metrics: stability curve D(rho_k || rho_hat), its stationary reference value (``sampling_floor``) and the
distance to the target (Section 7.2)."""

import numpy as np
import pytest

from dmdp_sg.metrics import distance_to_target, estimate_rho_hat, sampling_floor, split_halves, stability_curve
from dmdp_sg.targets import Target

EPISODES, STEPS, K0, BLOCK = 64, 1000, 500, 10


def _late_mean(values, block=BLOCK):
    centers, D, _ = stability_curve(values, k0=K0, block=block)
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
    uniform = np.random.default_rng(9).uniform(0.0, 3.2919, (EPISODES, STEPS))
    assert distance_to_target(uniform, 1.64595, 0.95029, k0=K0, family="uniform") < 0.02
    assert distance_to_target(uniform, 1.64595, 0.95029, k0=K0) > 0.1               # not close to the Gaussian


def _w1_to_uniform(sample, lo, hi):
    """Exact W1 between the empirical distribution of ``sample`` and U[lo, hi], in closed form.

    With the order statistics x_(i), W1 is the sum over i of int_{(i-1)/n}^{i/n} |x_(i) - (lo + (hi - lo) u)| du. The
    integrand is |g| for a function g that is linear in u with slope hi - lo: a piece without a sign change is a
    trapezoid, and a piece with one is two triangles, whose areas are g^2 / (2 (hi - lo)) at the two ends of the piece.
    """
    x = np.sort(sample)
    n, width = x.size, hi - lo
    g0, g1 = lo + width * np.arange(n) / n - x, lo + width * np.arange(1, n + 1) / n - x
    return float(np.sum(np.where(g0 * g1 >= 0.0, np.abs(g0 + g1) / (2 * n), (g0 ** 2 + g1 ** 2) / (2 * width))))


def test_closed_form_w1_to_uniform_is_exact():
    # the closed form against a fine numerical integral of |F_sample - F_uniform|
    x = np.array([-0.3, 0.2, 0.2, 1.1, 4.0])
    grid = np.linspace(-2.0, 6.0, 4_000_001)
    cdf_sample = np.searchsorted(np.sort(x), grid, side="right") / x.size
    cdf_uniform = np.clip((grid - 0.0) / 3.0, 0.0, 1.0)
    numerical = np.sum(np.abs(cdf_sample - cdf_uniform)) * (grid[1] - grid[0])
    assert _w1_to_uniform(x, 0.0, 3.0) == pytest.approx(numerical, abs=1e-5)


def test_distance_to_a_uniform_target_matches_the_closed_form():
    mu, sd = 1.6, 0.95
    lo, hi = mu - 3 ** 0.5 * sd, mu + 3 ** 0.5 * sd
    rng = np.random.default_rng(17)
    samples = [rng.uniform(lo, hi, 500),                       # from the target itself
               rng.normal(0.5, 1.5, 300),                      # extends beyond both ends of the support
               rng.uniform(lo + 0.3, hi - 0.8, 200),           # a narrower sample inside the support
               np.round(rng.normal(mu, 0.3, 400), 1),          # ties
               rng.uniform(hi, hi + 2.0, 50),                  # disjoint from the support
               np.array([mu, mu + 0.2, mu - 0.7])]
    for sample in samples:
        d = distance_to_target(sample[None, :], mu, sd, k0=0, family="uniform")
        assert d == pytest.approx(_w1_to_uniform(sample, lo, hi), abs=1e-4)


@pytest.mark.parametrize("family", ["gaussian", "uniform"])
def test_distance_to_target_has_no_bias_from_the_representation_of_the_target(family):
    # a pure shift of rho_star by s has the distance s exactly; with a random sample as the reference, the sample mean
    # of the reference would add an error of the order of sd / sqrt(n_ref)
    mu, sd, n_ref = 2.2, 0.5, 20000
    exact = np.asarray(Target(family, mu, sd).quantile((np.arange(n_ref) + 0.5) / n_ref))
    for shift in (0.0, 0.3, -0.7):
        d = distance_to_target((exact + shift).reshape(100, -1), mu, sd, k0=0, family=family)
        assert d == pytest.approx(abs(shift), abs=1e-9)


def test_distance_to_target_is_deterministic():
    values = np.random.default_rng(18).normal(2.0, 0.6, (EPISODES, STEPS))
    results = []
    for global_seed in (1, 2):
        np.random.seed(global_seed)                            # the reference must not depend on any random state
        results.append([distance_to_target(values, 2.2, 0.5, k0=K0, family=f) for f in ("gaussian", "uniform")])
    assert results[0] == results[1]


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
    values = _gait(np.zeros(EPISODES), rng)                    # synchronized oscillations: rho_k is periodic
    assert _late_mean(values) / sampling_floor(values, K0, BLOCK, np.random.default_rng(13)) > 1.5


def test_single_steps_detect_an_oscillation_that_blocks_average_out():
    # the oscillations of all episodes are synchronized with a period of 5 steps, so rho_k is periodic, but a block
    # of 10 steps always pools two full periods and the pooled distribution does not depend on the block
    rng = np.random.default_rng(14)
    values = np.sin(2 * np.pi * np.arange(STEPS) / 5) + 0.3 * rng.normal(size=(EPISODES, STEPS))
    ratio = {b: _late_mean(values, b) / sampling_floor(values, K0, b, np.random.default_rng(15)) for b in (1, 10)}
    assert ratio[1] > 1.5
    assert 0.85 < ratio[10] < 1.15


def test_default_block_is_a_single_step():
    values = np.random.default_rng(16).normal(size=(EPISODES, STEPS))
    centers, D, _ = stability_curve(values, k0=K0)
    assert centers.shape == D.shape == (STEPS,)


def test_distance_between_equal_samples_is_zero():
    # without the clamp in wasserstein_1, rounding makes the distance slightly negative for about half of these seeds
    from dmdp_sg.metrics import Reference, wasserstein_1
    for seed in range(20):
        x = np.random.default_rng(seed).normal(size=320)
        d = wasserstein_1(x, Reference(x))
        assert 0.0 <= d < 1e-12
    assert np.isnan(wasserstein_1(np.array([0.3, np.nan, 1.0]), Reference(np.arange(5.0))))


def test_exceedance_probability():
    from dmdp_sg.metrics import exceedance_probability
    returns = np.array([900.0, 1000.0, 1100.0, np.nan, 1300.0])
    np.testing.assert_allclose(exceedance_probability(returns, [800.0, 1000.0, 1000.1, 1300.0, 1400.0]),
                               [1.0, 0.75, 0.5, 0.25, 0.0])
    assert exceedance_probability(returns, 1100.0).shape == (1,)
    assert np.isnan(exceedance_probability(np.array([np.nan]), 0.0)[0])
