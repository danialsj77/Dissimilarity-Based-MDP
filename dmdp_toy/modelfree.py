"""Model-free online learner and the quantities of Theorem 5 (theo:model_free_online_policy_learning) in Appendix D.

Lyapunov surrogate W_n = D for every n, so eq. (model_free_Wn_bounds) holds with alpha_low = alpha_up = r^2. The
learner runs episodes of 5 steps from omega_0 ~ U(Omega_com); at each step it sees the projected maximum-likelihood
estimate omega_hat_t of M_LEARN fresh samples (``system.sample_mle``). The dataset of eq. (model_free_online_dataset)
after n episodes has N_n = 5n tuples; the regression input is (omega_hat_t, a_t) and the target the unbiased estimate
``unbiased_W`` of D(rho_{t+1} || rho_star), with the known noise variance ``unbiased_W_var``. A heteroscedastic GP
gives W^+_n(omega, a), the learned next value, and the policy phi_n interpolates the node argmin actions of W^+_n
(a 26 x 26 grid of A refined by the zoom of ``vi.greedy``, so they are not tied to the action grid).

The empirical decrease is D_hat_n(omega) = W^+_n(omega, phi_n(omega)) - D(omega). Its capped version
min{D_hat_n, -sigma_n} satisfies eq. (model_free_empirical_decrease_certificate) by construction, and
Delta_dec(capped) <= Delta_dec(raw) + e_n with the raw violation e_n = max(0, max[D_hat_n + sigma_n]). The true
decrease D(F_n) - D of eq. (true_exact_decrease_model_free) needs the true f (oracle evaluation only).
"""

from __future__ import annotations

import numpy as np

from . import system as S
from .gp import GP
from .grids import Grid, uniform_actions
from .vi import ZOOM_PASSES, NodePolicy, trajectory, zoom_argmin

EPISODE_STEPS = 5
M_LEARN = 20_000
CHECKPOINTS = (20, 40, 80, 160)
UNIFORM_EPISODES = 20
P_UNIFORM = 0.5
EXPLORATION_SD = (0.05, 0.2)
POLICY_GRID = (0.02, 0.35, 0.2, 0.1)          # 37 x 20 nodes
POLICY_ACTIONS = (26, 26)
C_SIGMA = 0.25                                # sigma_n(r) = 0.25 r^2, fixed a priori
FINE_KEYS = ("Delta_dec", "Delta_min")         # rechecked on the independent set for the deployed run


def greedy_policy(gp, grid=None, n_actions=POLICY_ACTIONS, zoom=ZOOM_PASSES, block=100):
    """phi_n: node action argmin_a W^+_n(node, a), bilinearly interpolated between the nodes.

    Uniform 26 x 26 grid of A, then the zoom of ``vi.greedy`` (two 9 x 9 passes of plus or minus one step, the step
    divided by 4), so the node actions are not tied to the action grid (final spacing 1/16 of its step). Only the GP
    W^+_n is evaluated; nodes are processed in blocks of ``block``.
    """
    grid = grid or Grid.graded(*POLICY_GRID)
    n1, n2 = n_actions
    actions = uniform_actions(n1, n2)
    h = ((S.A1_HI - S.A1_LO) / (n1 - 1), (S.A2_HI - S.A2_LO) / (n2 - 1))
    nodes = np.stack([grid.M.ravel(), grid.V.ravel()], 1)
    theta = np.empty((len(nodes), 2))
    for b in range(0, len(nodes), block):
        nb = nodes[b:b + block]

        def W_plus(c1, c2, nb=nb):
            X = np.column_stack([np.repeat(nb, c1.shape[1], 0), c1.ravel(), c2.ravel()])
            return gp.mean(X).reshape(c1.shape)

        Q = W_plus(*(np.broadcast_to(actions[:, i], (len(nb), len(actions))) for i in range(2)))
        k = Q.argmin(1)
        theta[b:b + block], _ = zoom_argmin(W_plus, actions[k], Q[np.arange(len(k)), k], h, zoom)
    return NodePolicy(grid, theta.reshape(grid.shape + (2,)))


def run_learner(rng, checkpoints=CHECKPOINTS, M=M_LEARN):
    """Runs the learner for max(checkpoints) episodes; returns {n: (W^+_n GP, phi_n)} at the checkpoints."""
    X, y, v = [], [], []
    policy, out = None, {}
    for ep in range(1, max(checkpoints) + 1):
        mu, vs = rng.uniform(S.MU_LO, S.MU_HI), rng.uniform(S.VS_LO, S.VS_HI)
        xbar, vs_mle, _ = S.sample_mle(mu, vs, M, rng)
        w_hat = S.project(xbar, vs_mle)
        for _ in range(EPISODE_STEPS):
            if ep <= UNIFORM_EPISODES or policy is None or rng.uniform() < P_UNIFORM:
                a = np.array([rng.uniform(S.A1_LO, S.A1_HI), rng.uniform(S.A2_LO, S.A2_HI)])
            else:
                a = policy(np.array([w_hat[0]]), np.array([w_hat[1]]))[0] + rng.normal(0.0, EXPLORATION_SD)
                a = np.clip(a, [S.A1_LO, S.A2_LO], [S.A1_HI, S.A2_HI])
            mu, vs = S.f(mu, vs, a[0], a[1])
            xbar, vs_mle, s2 = S.sample_mle(mu, vs, M, rng)
            X.append([w_hat[0], w_hat[1], a[0], a[1]])
            y.append(S.unbiased_W(xbar, s2, M))
            v.append(S.unbiased_W_var(xbar, s2, M))
            w_hat = S.project(xbar, vs_mle)
        if ep in checkpoints:
            gp = GP(np.array(X), np.array(y), S.LO, S.HI, noise_var=np.array(v))
            policy = greedy_policy(gp)
            out[ep] = (gp, policy)
    return out


def decrease_quantities(gp, policy, mu, vs, c_sigma=C_SIGMA):
    """Raw and capped empirical decrease, Delta_dec,n (eq. model_free_decrease_estimation_error), e_n, the violation
    radius and the smallest residual that holds, Delta_min,n = max[D(F_n) - D + sigma_n], on the test points."""
    a = policy(mu, vs)
    d = S.dist2(mu, vs)
    d_hat = gp.mean(np.stack([mu, vs, a[:, 0], a[:, 1]], 1)) - d
    true_dec = S.dist2(*S.f(mu, vs, a[:, 0], a[:, 1])) - d
    capped = np.minimum(d_hat, -c_sigma * d)
    viol = d_hat + c_sigma * d > 0
    return dict(Delta_dec=float(np.max(np.abs(capped - true_dec))),
                Delta_dec_raw=float(np.max(np.abs(d_hat - true_dec))),
                e_n=float(max(0.0, np.max(d_hat + c_sigma * d))),
                viol_radius=float(np.sqrt(d[viol].max())) if viol.any() else 0.0,
                Delta_min=float(np.max(true_dec + c_sigma * d)),
                true_dec=true_dec, r=np.sqrt(d))


def offset(policy):
    """||F_n(omega_star) - omega_star|| under the true f."""
    a = policy(np.array([S.MU_STAR]), np.array([S.VS_STAR]))[0]
    m, s = S.f(S.MU_STAR, S.VS_STAR, a[0], a[1])
    return float(np.hypot(m - S.MU_STAR, s - S.VS_STAR))


def deployed_trajectory(policy, steps=40):
    return trajectory(policy, (S.MU_LO, S.VS_HI), steps)
