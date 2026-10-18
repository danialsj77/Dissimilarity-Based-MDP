"""Online plug-in deployment, Theorem 4 (theo:online_plugin_stability), and the plug-in runs of Theorem 5 (Appendix D).

Estimator: at step t, M_t fresh samples of rho_t give omega_hat_t = projection onto Omega_com of (xbar, vs_mle)
(``system.sample_mle``). Since omega_t lies in the convex Omega_com, the projection is non-expansive and
||omega_hat_t - omega_t|| <= vs_t G_M <= VS_HI G_M, where G_M = ||(Z / sqrt(M), sqrt(V / M) - 1)|| does not depend
on omega_t or on the policy. With p_t = alpha_on / ((t + 1)(t + 2)) and M_t the smallest M with
P(G_M > delta_on / VS_HI) <= p_t, the union bound gives eq. (online_estimator_uniform_error) for all t >= 0, every
omega_0 and every policy, with probability at least 1 - sum_t p_t = 1 - alpha_on. A fixed M fails this event with
probability one over an infinite horizon.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, stats

from . import system as S

DELTA_ON = 0.05
ALPHA_ON = 0.05
DELTAS = (0.01, 0.02, 0.05, 0.1)
RADII = (0.25, 0.5, 0.75, 1.0)
N_ANGLES = 24


def mle_error_tail(M, x):
    """P(G_M > x) by quadrature over V ~ chi2(M - 1):
    1 - int_{M(1-x)_+^2}^{M(1+x)^2} F_chi2(1)(M[x^2 - (sqrt(v/M) - 1)^2]) f_chi2(M-1)(v) dv."""
    lo, hi = M * max(1.0 - x, 0.0) ** 2, M * (1.0 + x) ** 2
    pdf = stats.chi2(M - 1).pdf

    def integrand(v):
        return stats.chi2.cdf(max(M * (x * x - (np.sqrt(v / M) - 1.0) ** 2), 0.0), 1) * pdf(v)

    val, _ = integrate.quad(integrand, lo, hi, limit=400, epsabs=1e-14, epsrel=1e-10)
    return 1.0 - val


def smallest_sample_size(x, p, start=10, growth=1.05):
    """Smallest M >= start with P(G_M > x) <= p (geometric search, then bisection)."""
    if mle_error_tail(start, x) <= p:
        return start
    lo, hi = start, int(start * growth) + 1
    while mle_error_tail(hi, x) > p:
        lo, hi = hi, int(hi * growth) + 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if mle_error_tail(mid, x) > p:
            lo = mid
        else:
            hi = mid
    return hi


def sample_size_schedule(t_max, delta=DELTA_ON, alpha=ALPHA_ON, vs_hi=S.VS_HI, extra=()):
    """M_t for t = 0, ..., t_max - 1 (and for the steps in ``extra``); M_t is nondecreasing in t."""
    x = delta / vs_hi
    out, M = {}, 10
    for t in list(range(t_max)) + list(extra):
        M = smallest_sample_size(x, alpha / ((t + 1) * (t + 2)), M, growth=2.0 if t == 0 else 1.05)
        out[t] = M
    return np.array([out[t] for t in range(t_max)]), {t: out[t] for t in extra}


def plugin_perturbation(W, policy, delta, grid_shape, radii=RADII, n_angles=N_ANGLES):
    """Delta_on(delta) of eq. (online_plugin_perturbation_bound) (W = U~) or Delta_omega,n(delta) of eq.
    (model_free_parameter_estimation_perturbation) (W = D): the maximum over omega on a uniform grid of Omega_com
    and omega~ = project(omega + rho delta (cos th, sin th)) of |W(f(omega, phi(omega~))) - W(f(omega, phi(omega)))|."""
    mg, sg = np.meshgrid(np.linspace(S.MU_LO, S.MU_HI, grid_shape[0]), np.linspace(S.VS_LO, S.VS_HI, grid_shape[1]),
                         indexing="ij")
    mu, vs = mg.ravel(), sg.ravel()
    a = policy(mu, vs)
    w0 = W(*S.f(mu, vs, a[:, 0], a[:, 1]))
    worst = 0.0
    for rho in radii:
        for th in np.linspace(0.0, 2 * np.pi, n_angles, endpoint=False):
            mt, st = S.project(mu + rho * delta * np.cos(th), vs + rho * delta * np.sin(th))
            at = policy(mt, st)
            worst = max(worst, float(np.max(np.abs(W(*S.f(mu, vs, at[:, 0], at[:, 1])) - w0))))
    return worst


def simulate_plugin(policy, W, starts, seeds, schedule, c_sigma, delta=DELTA_ON):
    """Plug-in closed loops omega_{t+1} = f(omega_t, phi(omega_hat_t)) with the estimator of the module docstring.

    One run per entry of ``starts`` (with its own seed). Returns, per run and step t < len(schedule), the decrease
    W(omega_{t+1}) - W(omega_t) + c_sigma D(omega_t), the estimation error, and the distances ||omega_t - omega_star||
    for t = 0, ..., len(schedule).
    """
    T = len(schedule)
    dec, err, dist = np.empty((len(starts), T)), np.empty((len(starts), T)), np.empty((len(starts), T + 1))
    for j, (w0, seed) in enumerate(zip(starts, seeds)):
        rng = np.random.default_rng(seed)
        mu, vs = float(w0[0]), float(w0[1])
        for t, M in enumerate(schedule):
            xbar, vs_mle, _ = S.sample_mle(mu, vs, int(M), rng)
            mh, sh = S.project(xbar, vs_mle)
            err[j, t] = np.hypot(mh - mu, sh - vs)
            a = policy(np.array([mh]), np.array([sh]))[0]
            mn, sn = S.f(mu, vs, a[0], a[1])
            dec[j, t] = float(W(np.array([mn]), np.array([sn]))[0] - W(np.array([mu]), np.array([vs]))[0]
                              + c_sigma * S.dist2(mu, vs))
            dist[j, t] = np.hypot(mu - S.MU_STAR, vs - S.VS_STAR)
            mu, vs = mn, sn
        dist[j, T] = np.hypot(mu - S.MU_STAR, vs - S.VS_STAR)
    return dict(dec=dec, err=err, dist=dist, event=np.all(err <= delta, axis=1))
