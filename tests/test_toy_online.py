"""Online estimator of the scalar example: exact error tail, sample-size schedule, projection, coupling."""

import numpy as np

from dmdp_toy import online, system as S


def test_tail_matches_monte_carlo():
    rng = np.random.default_rng(0)
    n = 200_000
    for M, x in [(200, 0.15), (2000, 0.05)]:
        G = np.sqrt(rng.standard_normal(n) ** 2 / M + (np.sqrt(rng.chisquare(M - 1, n) / M) - 1) ** 2)
        p = online.mle_error_tail(M, x)
        assert abs(p - np.mean(G > x)) < 5 * np.sqrt(p * (1 - p) / n)


def test_sample_size_schedule():
    sched, _ = online.sample_size_schedule(2)
    assert sched.tolist() == [5950, 7916]
    x = online.DELTA_ON / S.VS_HI
    for t, M in enumerate(sched):
        p = online.ALPHA_ON / ((t + 1) * (t + 2))
        assert online.mle_error_tail(M, x) <= p < online.mle_error_tail(M - 1, x)
    longer, _ = online.sample_size_schedule(6)
    assert np.all(np.diff(longer) >= 0)
    t = np.arange(100_000)
    assert np.sum(online.ALPHA_ON / ((t + 1) * (t + 2))) <= online.ALPHA_ON


def test_projection_is_non_expansive():
    rng = np.random.default_rng(1)
    w = np.c_[rng.uniform(S.MU_LO, S.MU_HI, 1000), rng.uniform(S.VS_LO, S.VS_HI, 1000)]
    w_hat = w + rng.normal(0, 1.0, w.shape)
    p = np.c_[S.project(w_hat[:, 0], w_hat[:, 1])]
    assert np.all(np.linalg.norm(p - w, axis=1) <= np.linalg.norm(w_hat - w, axis=1) + 1e-15)


def test_coupling_error_equals_vs_times_G():
    M, mu, vs = 500, np.full(100, 0.3), np.linspace(S.VS_LO, S.VS_HI, 100)
    xbar, vs_mle, s2 = S.sample_mle(mu, vs, M, np.random.default_rng(5))
    rng = np.random.default_rng(5)
    z, v = rng.standard_normal(100), rng.chisquare(M - 1, 100)
    G = np.sqrt(z ** 2 / M + (np.sqrt(v / M) - 1) ** 2)
    assert np.allclose(np.hypot(xbar - mu, vs_mle - vs), vs * G)
    assert np.allclose(s2, vs_mle ** 2 * M / (M - 1))


def test_unbiased_target():
    rng = np.random.default_rng(3)
    M, mu, vs = 50, 1.7, 1.1
    xbar, _, s2 = S.sample_mle(np.full(400_000, mu), np.full(400_000, vs), M, rng)
    w = S.unbiased_W(xbar, s2, M)
    assert abs(w.mean() - S.dist2(mu, vs)) < 5 * w.std() / np.sqrt(w.size)
