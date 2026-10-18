"""Offline model-based pipeline and the quantities of Theorem 3 (theo:stability_finite_samples) in Appendix D.

Data: B independent datasets of the form of eq. (data_time_series), each with M trajectories of T steps under one
action sequence a_k ~ U(A) and its own omega_0 ~ U(Omega_com), so n_tr = BT tuples (omega_hat_k, a_k,
omega_hat_{k+1}) and N = BTM samples. omega_hat_k is the projection onto Omega_com of the maximum-likelihood estimate
(sample mean, standard deviation with divisor M); Appendix C's density step is skipped because samples of rho_k
are available. Model: one GP per output, f_hat = projection onto Omega_com of the GP means (so Omega_com is invariant
under F_hat_off by construction). The learned equilibrium action a_hat solves f_hat(omega_star, a) = omega_star,
which the learned margin requires (U_hat >= 0 with U_hat(omega_star) = 0 forces F_hat_off(omega_star) = omega_star).
The learned value iteration solves eq. (parameter_based_approximation) with gamma = 1, l - l_0 = D, lambda = 0, on
an action grid centred at a_hat; the policy interpolates the node argmin actions. The learner sees only its data
and omega_star, D, A, Omega_com, l_0 = 0; the true f and U~ are used only in ``evaluate_run``.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import least_squares

from . import system as S
from .certify import relative_change
from .gp import GP
from .grids import Grid, graded_actions
from .vi import NodePolicy, ValueFunction, node_policy, trajectory, value_iteration_ratio

T_STEPS = 5
LEVELS = {1: (20, 50), 2: (40, 200), 3: (80, 800), 4: (160, 3200)}      # level: (B, M)
LEARNED_GRID = (0.01, 0.3, 0.15, 0.075)                                 # 49 x 28 nodes
LEARNED_ACTIONS = ((0.01, 0.35, 0.03), (0.01, 0.35, 0.1))               # about 460 actions around a_hat
EQ_STARTS = ((0.825, 1.0), (0.72, 0.52), (0.93, 1.48))
C_SIGMA = 0.75                                                           # sigma(r) = 0.75 r^2, fixed a priori
FINE_KEYS = ("Delta_U", "Delta_f", "Delta", "c_hat", "Delta_min")    # rechecked on the independent set
LIMIT_STARTS = ((S.MU_LO, S.VS_LO), (S.MU_LO, S.VS_HI), (S.MU_HI, S.VS_LO), (S.MU_HI, S.VS_HI),
                (1.0, S.VS_LO), (1.0, S.VS_HI), (S.MU_LO, 1.15), (S.MU_HI, 1.15))


def mle_projected(x):
    """Projection onto Omega_com of the Gaussian maximum-likelihood estimate of samples along the last axis."""
    return S.project(x.mean(-1), x.std(-1))


def generate_data(B, T, M, rng):
    """B independent datasets of eq. (data_time_series); returns inputs (BT, 4) = (omega_hat_k, a_k), outputs (BT, 2)."""
    X, Y = [], []
    for _ in range(B):
        x = rng.uniform(S.MU_LO, S.MU_HI) + rng.uniform(S.VS_LO, S.VS_HI) * rng.standard_normal(M)
        w = mle_projected(x)
        for _ in range(T):
            a1, a2 = rng.uniform(S.A1_LO, S.A1_HI), rng.uniform(S.A2_LO, S.A2_HI)
            x = S.step_particles(x, a1, a2, rng)
            w_next = mle_projected(x)
            X.append([w[0], w[1], a1, a2])
            Y.append([w_next[0], w_next[1]])
            w = w_next
    return np.array(X), np.array(Y)


class LearnedModel:
    """f_hat_off(omega, a) = projection onto Omega_com of the means of one GP per output."""

    def __init__(self, X=None, Y=None, gps=None):
        self.gps = gps if gps is not None else [GP(X, Y[:, i], S.LO, S.HI) for i in range(2)]

    def __call__(self, mu, vs, a1, a2):
        mu, vs, a1, a2 = np.broadcast_arrays(mu, vs, a1, a2)
        Xq = np.stack([mu.ravel(), vs.ravel(), a1.ravel(), a2.ravel()], 1)
        m, s = (g.mean(Xq).reshape(mu.shape) for g in self.gps)
        return S.project(m, s)

    def state(self):
        return {**self.gps[0].state("gp0_"), **self.gps[1].state("gp1_")}

    @classmethod
    def from_state(cls, st):
        return cls(gps=[GP.from_state(st, "gp0_"), GP.from_state(st, "gp1_")])


def equilibrium_action(model):
    """a_hat with f_hat(omega_star, a_hat) = omega_star (bounded least squares from 3 starts); (a_hat, residual)."""
    def residual(a):
        m, s = model(S.MU_STAR, S.VS_STAR, a[0], a[1])
        return np.array([float(m) - S.MU_STAR, float(s) - S.VS_STAR])

    best = None
    for a0 in EQ_STARTS:
        r = least_squares(residual, a0, bounds=([S.A1_LO, S.A2_LO], [S.A1_HI, S.A2_HI]),
                          xtol=1e-14, ftol=1e-14, gtol=1e-14)
        if best is None or r.cost < best.cost:
            best = r
    return best.x, float(np.sqrt(2 * best.cost))


def learned_actions(a_hat):
    """Action grid of the learned value iteration, graded around a_hat (never around a_star); a_hat is a node."""
    return graded_actions(a_hat, *LEARNED_ACTIONS)


def learned_vi(model, a_hat):
    """Value iteration of eq. (parameter_based_approximation) on the learned model; returns (grid, actions, VIResult)."""
    grid = Grid.graded(*LEARNED_GRID)
    actions = learned_actions(a_hat)
    return grid, actions, value_iteration_ratio(grid, actions, model=model, max_iter=500)


def learn(B, M, rng):
    """The learner: data, model, a_hat, learned value and policy. Uses only the data and the known quantities."""
    t0 = time.time()
    X, Y = generate_data(B, T_STEPS, M, rng)
    model = LearnedModel(X, Y)
    t_fit = time.time() - t0
    a_hat, eq_res = equilibrium_action(model)
    grid, actions, res = learned_vi(model, a_hat)
    return dict(X=X, Y=Y, model=model, a_hat=a_hat, eq_res=eq_res, grid=grid, actions=actions, vi=res,
                U_hat=ValueFunction(grid, res.Z), policy=node_policy(grid, actions, res.K), t_fit=t_fit)


def certificate_terms(U_true, U_hat, policy, model, mu, vs):
    """Delta_U, Delta_f (eqs. def_delta_U_finite_sample, def_delta_f_finite_sample) and the pointwise decreases."""
    a = policy(mu, vs)
    F = S.f(mu, vs, a[:, 0], a[:, 1])
    Fh = model(mu, vs, a[:, 0], a[:, 1])
    Ut = U_true(mu, vs)
    d = S.dist2(mu, vs)
    true_dec = U_true(*F) - Ut
    learned_dec = U_hat(*Fh) - U_hat(mu, vs)
    pos = d > 0
    return dict(Delta_U=float(np.max(np.abs(U_hat(mu, vs) - Ut))),
                Delta_f=float(np.max(np.abs(U_true(*F) - U_true(*Fh)))),
                c_hat=float(np.min(-learned_dec[pos] / d[pos])),
                Delta_min=float(np.max(true_dec + C_SIGMA * d)),
                true_dec=true_dec, r=np.sqrt(d))


def limit_distance(policy, model=S.f, steps=300):
    """max over the 8 starts of ||F^steps(omega_0) - omega_star|| and the spread of the end points."""
    ends = np.array([trajectory(policy, w0, steps, model)[-1] for w0 in LIMIT_STARTS])
    return float(np.max(np.hypot(ends[:, 0] - S.MU_STAR, ends[:, 1] - S.VS_STAR))), float(np.max(np.abs(ends - ends[0])))


def evaluate_run(learned, U_true, tests, model_tests, fine_tests=None):
    """Per-run quantities of Theorem 3 on the test sets (oracle evaluation with f and U~). With ``fine_tests`` (the
    independent check set), also the FINE_KEYS quantities there and their relative change."""
    mu, vs = tests
    t = certificate_terms(U_true, learned["U_hat"], learned["policy"], learned["model"], mu, vs)
    Delta = t["Delta_f"] + 2 * t["Delta_U"]
    mt, st, a1t, a2t = model_tests
    fm, fs = S.f(mt, st, a1t, a2t)
    hm, hs = learned["model"](mt, st, a1t, a2t)
    F_star = S.f(S.MU_STAR, S.VS_STAR, *learned["policy"](np.array([S.MU_STAR]), np.array([S.VS_STAR]))[0])
    lim, spread = limit_distance(learned["policy"])
    ok = bool(learned["eq_res"] < 1e-8 and learned["vi"].converged and t["c_hat"] >= C_SIGMA)
    out = dict(Delta_U=t["Delta_U"], Delta_f=t["Delta_f"], Delta=Delta, c_hat=t["c_hat"],
               e_f=float(np.max(np.hypot(fm - hm, fs - hs))),
               offset=float(np.hypot(F_star[0] - S.MU_STAR, F_star[1] - S.VS_STAR)),
               limit=lim, limit_spread=spread, limit_flag=bool(spread > 1e-6), Delta_min=t["Delta_min"],
               a_err=float(np.hypot(*(learned["a_hat"] - S.A_STAR))), eq_res=learned["eq_res"],
               vi_iterations=learned["vi"].iterations, vi_converged=learned["vi"].converged,
               score=Delta if ok else np.inf, theta_gp0=learned["model"].gps[0].theta,
               theta_gp1=learned["model"].gps[1].theta, a_hat=learned["a_hat"])
    if fine_tests is not None:                     # deployed run: the same maxima on the independent check set
        tf = certificate_terms(U_true, learned["U_hat"], learned["policy"], learned["model"], *fine_tests)
        tf["Delta"] = tf["Delta_f"] + 2 * tf["Delta_U"]
        out["fine"] = np.array([tf[k] for k in FINE_KEYS])
        out["fine_change"] = np.array([relative_change(tf[k], out[k]) for k in FINE_KEYS])
        out["r"], out["true_dec"] = t["r"], t["true_dec"]
    return out


def deployed_arrays(learned, steps=40):
    """Files kept for the deployed run: data, GP states, learned value and policy, and the trajectory from (-1, VS_HI)."""
    traj = trajectory(learned["policy"], (S.MU_LO, S.VS_HI), steps)
    return dict(X=learned["X"], Y=learned["Y"], **learned["model"].state(), Z_hat=learned["vi"].Z,
                grid_mu=learned["grid"].mu, grid_vs=learned["grid"].vs, theta=learned["policy"].theta,
                traj=traj, traj_D=S.dist2(traj[:, 0], traj[:, 1]))


def load_policy(data) -> NodePolicy:
    return NodePolicy(Grid(data["grid_mu"], data["grid_vs"]), data["theta"])
