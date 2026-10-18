"""Tensor grids on Omega_com and on A for the value iterations and the interpolated policies (Appendix D).

A graded axis contains a centre point c as a node; the spacing grows with the distance d to c as
min(h_max, max(h_min, kg * d)). Graded grids on Omega_com are centred at omega_star, which must be a node: the
interpolated policy then takes the node action at omega_star, so that F(omega_star) = omega_star can hold.
"""

from __future__ import annotations

import numpy as np

from . import system as S


def graded_axis(lo, hi, c, hmin, kg, hmax):
    """Nodes on [lo, hi] containing c, with spacing min(hmax, max(hmin, kg * |x - c|))."""
    def side(length):
        if length <= 0:
            return np.array([0.0])
        d = [0.0]
        while d[-1] < length:
            h = min(hmax, max(hmin, kg * d[-1]))
            d.append(d[-1] + h)
        d = np.array(d)
        d[-1] = length
        if len(d) > 2 and d[-1] - d[-2] < 0.3 * h:       # merge a last cell shorter than 0.3 h
            d = np.delete(d, -2)
        return d

    left = c - side(c - lo)[::-1]
    right = c + side(hi - c)
    x = np.concatenate([left, right[1:]])
    x[0], x[-1] = lo, hi
    return x


class Grid:
    """Tensor grid of (mu, varsigma) nodes with bilinear interpolation of node values."""

    def __init__(self, mu, vs):
        self.mu, self.vs = np.asarray(mu, float), np.asarray(vs, float)
        self.M, self.V = np.meshgrid(self.mu, self.vs, indexing="ij")
        self.shape = self.M.shape

    @classmethod
    def graded(cls, hmin, kg, hmax_mu, hmax_vs, center=S.OMEGA_STAR):
        return cls(graded_axis(S.MU_LO, S.MU_HI, center[0], hmin, kg, hmax_mu),
                   graded_axis(S.VS_LO, S.VS_HI, center[1], hmin, kg, hmax_vs))

    @classmethod
    def uniform(cls, n_mu, n_vs):
        return cls(np.linspace(S.MU_LO, S.MU_HI, n_mu), np.linspace(S.VS_LO, S.VS_HI, n_vs))

    def has_node(self, point):
        return bool(np.any(self.mu == point[0]) and np.any(self.vs == point[1]))

    def locate(self, qm, qs):
        """Cell indices and local coordinates of the query points (clipped to the grid)."""
        mu, vs = self.mu, self.vs
        qm = np.clip(qm, mu[0], mu[-1])
        qs = np.clip(qs, vs[0], vs[-1])
        i = np.clip(np.searchsorted(mu, qm, side="right") - 1, 0, len(mu) - 2)
        j = np.clip(np.searchsorted(vs, qs, side="right") - 1, 0, len(vs) - 2)
        return i, j, (qm - mu[i]) / (mu[i + 1] - mu[i]), (qs - vs[j]) / (vs[j + 1] - vs[j])

    def interp(self, values, qm, qs):
        """Bilinear interpolation of node values of shape grid.shape (+ trailing axes) at the query points."""
        i, j, tm, ts = self.locate(qm, qs)
        if values.ndim > 2:
            tm, ts = tm[..., None], ts[..., None]
        return ((1 - tm) * ((1 - ts) * values[i, j] + ts * values[i, j + 1])
                + tm * ((1 - ts) * values[i + 1, j] + ts * values[i + 1, j + 1]))


def graded_actions(center, spec1, spec2):
    """Tensor grid of A graded around ``center``; spec = (hmin, kg, hmax) per component. The centre is a node."""
    a1 = graded_axis(S.A1_LO, S.A1_HI, center[0], *spec1)
    a2 = graded_axis(S.A2_LO, S.A2_HI, center[1], *spec2)
    g1, g2 = np.meshgrid(a1, a2, indexing="ij")
    return np.stack([g1.ravel(), g2.ravel()], 1)


def uniform_actions(n1, n2):
    """Uniform n1 x n2 tensor grid of A."""
    g1, g2 = np.meshgrid(np.linspace(S.A1_LO, S.A1_HI, n1), np.linspace(S.A2_LO, S.A2_HI, n2), indexing="ij")
    return np.stack([g1.ravel(), g2.ravel()], 1)
