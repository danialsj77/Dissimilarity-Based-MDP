"""Value iteration for the undiscounted problems (P-OCP) and (AP-OCP^R) with l - l_0 = D and lambda = 0 (Appendix D).

The value is stored as the ratio Z = U / D at the nodes of a grid that contains omega_star (Z(omega_star) := 1),
and U(omega) = D(omega) * bilinear(Z)(omega) between the nodes. The iteration

    U <- D + min_a D(f(omega, a)) Z(f(omega, a)),    starting from U = 0,

is the Bellman operator of eq. (rotated_bellman_identity_parameter) with gamma = 1 on the grid. The successor
data of every (node, action) pair is precomputed once in node blocks, with int16 cell indices and float32
weights. With the true f, the result U~ approximates U_star^R; with a learned model, it is the learned value
U_hat of eq. (parameter_based_approximation), and the node argmin actions define the interpolated policy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import system as S
from .grids import Grid, uniform_actions


@dataclass
class VIResult:
    Z: np.ndarray            # U / D at the nodes
    K: np.ndarray            # index of the argmin action at each node
    iterations: int
    change: float            # last sup-norm change of U
    converged: bool
    seconds: float


def value_iteration_ratio(grid: Grid, actions, model=None, tol=1e-10, max_iter=200, block=200) -> VIResult:
    """Ratio value iteration on ``grid`` over the finite action set ``actions`` with the transition map ``model``
    (the true f when None; looked up at call time, so a learner that drops ``model=`` is caught by the tests).

    ``converged`` requires a change below ``tol`` within ``max_iter`` iterations and omega_star to be a node
    (otherwise the interpolated policy cannot fix omega_star and Z(omega_star) is undefined).
    """
    t0 = time.time()
    model = S.f if model is None else model
    mf, sf = grid.M.ravel(), grid.V.ravel()
    n, na = mf.size, len(actions)
    I = np.empty((n, na), np.int16)
    J = np.empty((n, na), np.int16)
    TM = np.empty((n, na), np.float32)
    TS = np.empty((n, na), np.float32)
    PS = np.empty((n, na), np.float32)
    for b in range(0, n, block):
        sm, ss = model(mf[b:b + block, None], sf[b:b + block, None], actions[None, :, 0], actions[None, :, 1])
        I[b:b + block], J[b:b + block], TM[b:b + block], TS[b:b + block] = grid.locate(sm, ss)
        PS[b:b + block] = S.dist2(sm, ss)
    d_nodes = S.dist2(mf, sf)
    pos = d_nodes > 0
    Z = np.ones(grid.shape)
    U = np.zeros(n)
    K = np.zeros(n, np.int64)
    change, it = np.inf, 0
    for it in range(1, max_iter + 1):
        U_new = np.empty(n)
        for b in range(0, n, block):
            i, j, tm, ts = I[b:b + block], J[b:b + block], TM[b:b + block], TS[b:b + block]
            zi = (1 - tm) * ((1 - ts) * Z[i, j] + ts * Z[i, j + 1]) + tm * ((1 - ts) * Z[i + 1, j] + ts * Z[i + 1, j + 1])
            Q = PS[b:b + block] * zi
            K[b:b + block] = Q.argmin(1)
            U_new[b:b + block] = d_nodes[b:b + block] + Q[np.arange(len(Q)), K[b:b + block]]
        change = float(np.max(np.abs(U_new - U)))
        U = U_new
        Z = np.where(pos, U / np.where(pos, d_nodes, 1.0), 1.0).reshape(grid.shape)
        if change < tol:
            break
    converged = bool(change < tol) and grid.has_node(S.OMEGA_STAR)
    return VIResult(Z, K.reshape(grid.shape), it, change, converged, time.time() - t0)


class ValueFunction:
    """U(omega) = D(omega) * bilinear(Z)(omega) on Omega_com."""

    def __init__(self, grid: Grid, Z):
        self.grid, self.Z = grid, np.asarray(Z)

    def __call__(self, mu, vs):
        return S.dist2(mu, vs) * self.grid.interp(self.Z, mu, vs)


class NodePolicy:
    """phi(omega; theta): bilinear interpolation of the node actions theta (continuous, values in the convex A)."""

    def __init__(self, grid: Grid, theta):
        self.grid, self.theta = grid, np.asarray(theta)

    def __call__(self, mu, vs):
        return self.grid.interp(self.theta, mu, vs)


def node_policy(grid: Grid, actions, K) -> NodePolicy:
    return NodePolicy(grid, actions[K])


ZOOM_PASSES = 2
_ZOOM = np.linspace(-1.0, 1.0, 9)


def zoom_argmin(Q, best, q, h, passes=ZOOM_PASSES):
    """Refines the argmin actions ``best`` (rows, 2) with values ``q``: ``passes`` passes on a 9 x 9 grid of A spanning
    plus or minus one current step h = (h1, h2) around the current best, the step divided by 4 after each pass (final
    spacing h / 16). ``Q(c1, c2)`` returns the values of the candidate actions c1, c2 of shape (rows, 81)."""
    best, q = np.array(best, float), np.array(q, float)
    h1, h2 = h
    rows = np.arange(len(best))
    for _ in range(passes):
        o1, o2 = np.meshgrid(_ZOOM * h1, _ZOOM * h2, indexing="ij")
        c1 = np.clip(best[:, :1] + o1.ravel()[None], S.A1_LO, S.A1_HI)
        c2 = np.clip(best[:, 1:] + o2.ravel()[None], S.A2_LO, S.A2_HI)
        Qc = Q(c1, c2)
        k = Qc.argmin(1)
        better = Qc[rows, k] < q
        best = np.where(better[:, None], np.c_[c1[rows, k], c2[rows, k]], best)
        q = np.minimum(Qc[rows, k], q)
        h1, h2 = h1 / 4, h2 / 4
    return best, q


def greedy(U, mu, vs, n=41, zoom=ZOOM_PASSES, closed_form=True, block=200):
    """Greedy action argmin_a U(f(omega, a)) under the true f (evaluation only) and the minimum.

    Uniform n x n grid of A, then ``zoom`` passes of ``zoom_argmin``. On R the closed form a = deadbeat(omega) is
    used (U(omega_star) = 0).
    """
    mu, vs = np.atleast_1d(mu).astype(float), np.atleast_1d(vs).astype(float)
    grid_a = uniform_actions(n, n)
    h = ((S.A1_HI - S.A1_LO) / (n - 1), (S.A2_HI - S.A2_LO) / (n - 1))
    best = np.empty((mu.size, 2))
    q = np.empty(mu.size)
    for b in range(0, mu.size, block):
        mm, ss = mu[b:b + block, None], vs[b:b + block, None]
        Q = U(*S.f(mm, ss, grid_a[None, :, 0], grid_a[None, :, 1]))
        k = Q.argmin(1)
        best[b:b + block], q[b:b + block] = zoom_argmin(lambda c1, c2: U(*S.f(mm, ss, c1, c2)), grid_a[k],
                                                        Q[np.arange(len(k)), k], h, zoom)
    if closed_form:
        a1, a2, inside = S.deadbeat(mu, vs)
        best[inside] = np.c_[a1[inside], a2[inside]]
        q[inside] = 0.0
    return best, q


def bellman_residual(U, mu, vs, **kw):
    """|U - D - min_a U(f)| at the given points: the residual of eq. (rotated_bellman_identity_parameter)."""
    _, q = greedy(U, mu, vs, **kw)
    return np.abs(U(mu, vs) - S.dist2(mu, vs) - q)


def rollout_upper_bound(U, mu, vs, kappa2=S.KAPPA2, steps=25, **kw):
    """Rigorous upper bound on U_star^R: sum_{k<K} D(omega_k) + kappa2 D(omega_K) along the greedy policy of U.

    Valid for every K because D(f(omega, a_star)) <= (39/64) D(omega) bounds the cost of continuing with a_star
    by kappa2 D(omega_K) (Lemma D.1); points that reach omega_star (D below 1e-24) stop early.
    """
    mm, ss = np.array(mu, float), np.array(vs, float)
    total = np.zeros_like(mm)
    active = np.ones(mm.shape, bool)
    for _ in range(steps):
        d = S.dist2(mm, ss)
        stop = active & (d <= 1e-24)
        total[stop] += kappa2 * d[stop]
        active &= ~stop
        if not active.any():
            break
        total[active] += d[active]
        a, _ = greedy(U, mm[active], ss[active], **kw)
        mm[active], ss[active] = S.f(mm[active], ss[active], a[:, 0], a[:, 1])
    total[active] += kappa2 * S.dist2(mm[active], ss[active])
    return total


def trajectory(policy, omega0, steps, model=S.f):
    """Parameter trajectory omega_0, ..., omega_steps of the closed loop omega_{k+1} = model(omega_k, policy(omega_k))."""
    mm, ss = np.array([float(omega0[0])]), np.array([float(omega0[1])])
    out = [(mm[0], ss[0])]
    for _ in range(steps):
        a = policy(mm, ss)
        mm, ss = model(mm, ss, a[:, 0], a[:, 1])
        out.append((mm[0], ss[0]))
    return np.array(out)
