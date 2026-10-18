"""Exact facts of the scalar example (Appendix D, Lemma lemma:toy_facts): closure, invariance, W_2^2, contraction,
kappa_2 and the one-step region R."""

import numpy as np
from scipy.stats import norm

from dmdp_toy import system as S


def _box(n_mu=121, n_vs=61, mu=(S.MU_LO, S.MU_HI), vs=(S.VS_LO, S.VS_HI)):
    m, s = np.meshgrid(np.linspace(*mu, n_mu), np.linspace(*vs, n_vs), indexing="ij")
    return m.ravel(), s.ravel()


def test_monte_carlo_closure():
    rng = np.random.default_rng(0)
    n = 100_000
    for (mu, vs, a1, a2) in [(-1.0, 1.6, 0.7, 1.5), (3.0, 0.7, 0.95, 0.5), (1.0, 0.8, S.A_STAR[0], 1.0)]:
        x = S.step_particles(mu + vs * rng.standard_normal(n), a1, a2, rng)
        m, s = S.f(mu, vs, a1, a2)
        assert abs(x.mean() - m) < 5 * s / np.sqrt(n)
        assert abs(x.std() - s) < 5 * s / np.sqrt(2 * n)


def test_invariance_under_every_action():
    m, s = _box()
    for a1 in (S.A1_LO, S.A1_HI, 0.8):
        for a2 in (S.A2_LO, S.A2_HI, 1.0):
            fm, fs = S.f(m, s, a1, a2)
            assert fm.min() >= S.MU_LO - 1e-12 and fm.max() <= S.MU_HI + 1e-12
            assert fs.min() >= S.VS_LO - 1e-12 and fs.max() <= S.VS_HI + 1e-12
    # the varsigma interval ends are the fixed points at the extreme persistences
    assert np.isclose(S.f(0, S.VS_LO, S.A1_LO, 1)[1], S.VS_LO) and np.isclose(S.f(0, S.VS_HI, S.A1_HI, 1)[1], S.VS_HI)


def test_dist2_is_quantile_w2():
    u = (np.arange(200_000) + 0.5) / 200_000
    for mu, vs in [(-1.0, 1.6), (2.2, 0.75), (1.0, 0.8)]:
        w2 = np.mean((norm.ppf(u, mu, vs) - norm.ppf(u, S.MU_STAR, S.VS_STAR)) ** 2)
        assert abs(w2 - S.dist2(mu, vs)) < 1e-3 * max(S.dist2(mu, vs), 1e-3)


def test_contraction_under_a_star_on_all_of_omega():
    m, s = _box(201, 201, (-6.0, 8.0), (0.01, 5.0))           # includes points outside Omega_com
    d = S.dist2(m, s)
    assert np.all(S.dist2(*S.f(m, s, *S.A_STAR)) <= S.CONTRACTION * d + 1e-12)
    assert np.isclose(S.CONTRACTION, S.A_STAR[0] ** 2)


def test_kappa2_bound_of_constant_a_star():
    m, s = _box()
    d = S.dist2(m, s)
    J, mm, ss = np.zeros_like(m), m.copy(), s.copy()
    for _ in range(400):
        J += S.dist2(mm, ss)
        mm, ss = S.f(mm, ss, *S.A_STAR)
    pos = d > 0
    assert np.max(J[pos] / d[pos]) <= S.KAPPA2 + 1e-9
    assert np.max(J[pos] / d[pos]) > 2.55                     # tight in the mu direction


def test_deadbeat_region():
    m, s = _box()
    a1, a2, inside = S.deadbeat(m, s)
    fm, fs = S.f(m[inside], s[inside], a1[inside], a2[inside])
    assert inside.any() and np.allclose(fm, S.MU_STAR, atol=1e-12) and np.allclose(fs, S.VS_STAR, atol=1e-12)
    th = np.linspace(0, 2 * np.pi, 720)
    for r in (0.01, 0.05, 0.09):
        assert S.deadbeat(S.MU_STAR + r * np.cos(th), S.VS_STAR + r * np.sin(th))[2].all()
    m, s = _box(1601, 801)
    assert abs(S.deadbeat(m, s)[2].mean() - 0.0147) < 3e-4      # R covers about 1.47% of Omega_com
