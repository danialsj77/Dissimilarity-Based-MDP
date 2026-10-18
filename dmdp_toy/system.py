"""Scalar linear system with Gaussian noise whose distribution dynamics close exactly (Appendix D).

    s_{k+1} = a_{k,1} s_k + (1 - a_{k,1}) a_{k,2} + epsilon_k,    epsilon_k ~ N(0, 0.5^2) i.i.d.,

with the action a_k = (a_{k,1}, a_{k,2}) in A = [0.70, 0.95] x [0.5, 1.5] common to all realizations. For
rho_omega = N(mu, varsigma^2) with omega = (mu, varsigma), the transition is closed in the Gaussian family,
T_{tau,a} rho_omega = rho_{f(omega, a)}, with the parameter dynamics of eq. (parameter_dynamics)

    f(omega, a) = (a1 mu + (1 - a1) a2, sqrt(a1^2 varsigma^2 + 0.25)).

Target rho_star = N(1, 0.8^2), omega_star = (1, 0.8), unique steady-state action a_star = (sqrt(39/64), 1).
The dissimilarity is D = W_2^2(rho_omega, rho_star) = (mu - 1)^2 + (varsigma - 0.8)^2 = r^2 with
r = ||omega - omega_star||, the stage cost is l = D, l_0 = 0 (eq. parameter_based_optimal_steady), the storage is
lambda = 0, so l^R = l (eq. rotated_stage_cost_parameter), and gamma = 1.

Omega_com = [-1, 3] x [VS_LO, VS_HI] is positively invariant under f(., a) for every a in A: mu+ is a convex
combination of mu and a2, and VS_LO, VS_HI are the fixed points of varsigma^2 -> a1^2 varsigma^2 + 0.25 at
a1 = 0.70 and a1 = 0.95. Lemma D.1 (lemma:toy_facts):
  * D(f(omega, a_star)) <= CONTRACTION * D(omega) for every omega in Omega, CONTRACTION = a_star1^2 = 39/64;
  * hence r^2 <= U_star^R <= KAPPA2 r^2 with KAPPA2 = 1 / (1 - 39/64) = 64/25;
  * on R = {omega in Omega_com : deadbeat(omega) in A}, f(omega, deadbeat(omega)) = omega_star and U_star^R = D.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

NOISE_SD = 0.5
A1_LO, A1_HI = 0.70, 0.95
A2_LO, A2_HI = 0.5, 1.5
MU_LO, MU_HI = -1.0, 3.0
VS_LO = NOISE_SD / np.sqrt(1.0 - A1_LO ** 2)        # 0.700140
VS_HI = NOISE_SD / np.sqrt(1.0 - A1_HI ** 2)        # 1.601282
MU_STAR, VS_STAR = 1.0, 0.8
OMEGA_STAR = np.array([MU_STAR, VS_STAR])
A_STAR = np.array([np.sqrt(1.0 - NOISE_SD ** 2 / VS_STAR ** 2), MU_STAR])   # (0.780625, 1)
CONTRACTION = 39.0 / 64.0
KAPPA2 = 64.0 / 25.0
LO = np.array([MU_LO, VS_LO, A1_LO, A2_LO])          # Omega_com x A
HI = np.array([MU_HI, VS_HI, A1_HI, A2_HI])


def f(mu, vs, a1, a2):
    """Exact parameter dynamics, eq. (parameter_dynamics)."""
    return a1 * mu + (1.0 - a1) * a2, np.sqrt(a1 ** 2 * vs ** 2 + NOISE_SD ** 2)


def dist2(mu, vs):
    """D(rho_omega || rho_star) = W_2^2 between N(mu, vs^2) and N(1, 0.8^2) = ||omega - omega_star||^2."""
    return (mu - MU_STAR) ** 2 + (vs - VS_STAR) ** 2


def project(mu, vs):
    """Componentwise projection onto Omega_com (the Euclidean projection onto the box, non-expansive)."""
    return np.clip(mu, MU_LO, MU_HI), np.clip(vs, VS_LO, VS_HI)


def deadbeat(mu, vs):
    """The action that maps omega to omega_star in one step, and the flag omega in R (the action lies in A)."""
    a1 = np.sqrt(VS_STAR ** 2 - NOISE_SD ** 2) / vs
    a2 = (MU_STAR - a1 * mu) / (1.0 - a1)
    inside = (a1 >= A1_LO) & (a1 <= A1_HI) & (a2 >= A2_LO) & (a2 <= A2_HI)
    return a1, a2, inside


def step_particles(x, a1, a2, rng):
    """One step of the sample-level dynamics for the particles x under the common action (a1, a2)."""
    return a1 * x + (1.0 - a1) * a2 + NOISE_SD * rng.standard_normal(np.shape(x))


def sample_mle(mu, vs, M, rng):
    """Sample statistics of M i.i.d. draws of N(mu, vs^2) through their exact distribution (location-scale coupling).

    xbar = mu + vs Z / sqrt(M) and vs_mle = vs sqrt(V / M) with Z ~ N(0, 1), V ~ chi2(M - 1) independent, so that
    ||(xbar, vs_mle) - omega|| = vs G_M with G_M = ||(Z / sqrt(M), sqrt(V / M) - 1)||, whose law does not depend on
    omega. Returns (xbar, vs_mle, s2) with the unbiased sample variance s2 = vs^2 V / (M - 1).
    """
    z = rng.standard_normal(np.shape(mu))
    v = rng.chisquare(M - 1, np.shape(mu))
    return mu + vs * z / np.sqrt(M), vs * np.sqrt(v / M), vs ** 2 * v / (M - 1)


def c4(M):
    """E[s] / varsigma for the unbiased sample variance s^2 of M Gaussian draws."""
    return np.exp(0.5 * np.log(2.0 / (M - 1)) + gammaln(M / 2.0) - gammaln((M - 1) / 2.0))


def unbiased_W(xbar, s2, M):
    """Unbiased estimate of D(rho_omega || rho_star) from xbar and the unbiased sample variance s2 (Theorem 5 targets)."""
    return (xbar - MU_STAR) ** 2 - s2 / M + s2 - 2.0 * VS_STAR * np.sqrt(s2) / c4(M) + VS_STAR ** 2


def unbiased_W_var(xbar, s2, M):
    """Delta-method variance of ``unbiased_W``, the per-point noise variance of the heteroscedastic GP."""
    return (4.0 * (xbar - MU_STAR) ** 2 * s2 + 2.0 * s2 * (np.sqrt(s2) - VS_STAR) ** 2) / M + 2.0 * s2 ** 2 / M ** 2
