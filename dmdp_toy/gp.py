"""Gaussian-process regression with an SE-ARD kernel and ML-II hyperparameters (Appendix C pipeline, Appendix D).

Inputs are scaled to [0, 1]^d by the box (lo, hi) and outputs are standardized. The kernel is
k(x, x') = sf^2 exp(-0.5 sum_i (x_i - x'_i)^2 / ell_i^2), the noise covariance sn^2 I + diag(v / sy^2) + 1e-8 I,
where v are optional known per-point noise variances (heteroscedastic GP of the model-free learner) and sy the
output standard deviation. theta = (log ell, log sf, log sn) maximizes the marginal likelihood (L-BFGS-B with
the analytic gradient, two initializations, at most 200 iterations; ell, sf in [0.05, 20], sn in [1e-6, 1]).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

JITTER = 1e-8
INITS = ((np.log(0.5), 0.0, np.log(0.05)), (np.log(1.5), 0.5, np.log(0.01)))
BOUNDS_ELL_SF = (np.log(0.05), np.log(20.0))
BOUNDS_SN = (np.log(1e-6), np.log(1.0))
CHUNK = 4000


class GP:
    def __init__(self, X, y, lo, hi, noise_var=None, theta=None):
        self.X_raw, self.y_raw = np.asarray(X, float), np.asarray(y, float)
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.X = self._scale(self.X_raw)
        self.ym, self.ys = self.y_raw.mean(), self.y_raw.std() + 1e-12
        self.y = (self.y_raw - self.ym) / self.ys
        self.v_raw = np.zeros(len(self.y)) if noise_var is None else np.asarray(noise_var, float)
        self.nv = self.v_raw / self.ys ** 2
        self._sq = None
        self.theta = self.fit() if theta is None else np.asarray(theta, float)
        self._factor()

    def _scale(self, X):
        return (np.atleast_2d(X) - self.lo) / (self.hi - self.lo)

    def _unpack(self, th):
        d = self.X.shape[1]
        return np.exp(th[:d]), np.exp(2 * th[d]), np.exp(2 * th[d + 1])

    def nll(self, th):
        """Negative log marginal likelihood of the standardized data and its gradient with respect to theta."""
        n, d = self.X.shape
        if self._sq is None:
            self._sq = np.stack([np.subtract.outer(self.X[:, k], self.X[:, k]) ** 2 for k in range(d)])
        ell, sf2, sn2 = self._unpack(th)
        Kf = sf2 * np.exp(-0.5 * np.tensordot(1.0 / ell ** 2, self._sq, axes=1))
        K = Kf + np.diag(sn2 + JITTER + self.nv)
        try:
            c = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10, np.zeros_like(th)
        a = cho_solve(c, self.y)
        value = 0.5 * self.y @ a + np.sum(np.log(np.diag(c[0]))) + 0.5 * n * np.log(2 * np.pi)
        W = np.outer(a, a) - cho_solve(c, np.eye(n))
        g = np.empty_like(th)
        for k in range(d):
            g[k] = -0.5 * np.sum(W * Kf * self._sq[k]) / ell[k] ** 2
        g[d] = -np.sum(W * Kf)
        g[d + 1] = -np.trace(W) * sn2
        return value, g

    def fit(self):
        d = self.X.shape[1]
        bounds = [BOUNDS_ELL_SF] * (d + 1) + [BOUNDS_SN]
        best = None
        for le, lsf, lsn in INITS:
            r = minimize(self.nll, np.r_[le * np.ones(d), lsf, lsn], jac=True, method="L-BFGS-B",
                         bounds=bounds, options={"maxiter": 200})
            if best is None or r.fun < best.fun:
                best = r
        self._sq = None
        return best.x

    def _factor(self):
        self.ell, self.sf2, self.sn2 = self._unpack(self.theta)
        K = self._kernel(self.X, self.X) + np.diag(self.sn2 + JITTER + self.nv)
        self.alpha = cho_solve(cho_factor(K, lower=True), self.y)

    def _kernel(self, A, B):
        As, Bs = A / self.ell, B / self.ell
        d2 = np.maximum(np.sum(As ** 2, 1)[:, None] + np.sum(Bs ** 2, 1)[None, :] - 2 * As @ Bs.T, 0)
        return self.sf2 * np.exp(-0.5 * d2)

    def mean(self, Xq, chunk=CHUNK):
        """Posterior mean at the rows of Xq, in chunks of at most ``chunk`` rows."""
        Xq = self._scale(Xq)
        out = np.empty(len(Xq))
        for s in range(0, len(Xq), chunk):
            out[s:s + chunk] = self._kernel(Xq[s:s + chunk], self.X) @ self.alpha
        return out * self.ys + self.ym

    def noise_var(self):
        """Learned homoscedastic noise variance in output units."""
        return self.sn2 * self.ys ** 2

    def state(self, prefix=""):
        return {prefix + "X": self.X_raw, prefix + "y": self.y_raw, prefix + "v": self.v_raw,
                prefix + "theta": self.theta, prefix + "lo": self.lo, prefix + "hi": self.hi}

    @classmethod
    def from_state(cls, st, prefix=""):
        """Rebuilds a fitted GP from ``state`` without refitting the hyperparameters."""
        v = st[prefix + "v"]
        return cls(st[prefix + "X"], st[prefix + "y"], st[prefix + "lo"], st[prefix + "hi"],
                   noise_var=v if np.any(v) else None, theta=st[prefix + "theta"])
