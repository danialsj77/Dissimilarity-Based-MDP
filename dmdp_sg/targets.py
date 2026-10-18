"""Target distributions rho_star and the exact 2-Wasserstein distance of a sample to them (Appendix E).

rho_star of Appendix E is the product of the velocity band U[0.70 v_lim, 0.95 v_lim] and the pitch target U[-0.2, 0.2]
(``config.band_target``, ``config.PITCH_HALF_WIDTH``). Each of its marginals is a member of one of the families in
``TARGET_FAMILIES``, specified by its mean mu_star and standard deviation sd_star:

    "gaussian"   N(mu_star, sd_star^2)
    "uniform"    U[mu_star - sqrt(3) sd_star, mu_star + sqrt(3) sd_star]

``target_mean`` belongs to the cost-derived targets of earlier designs (not reported in the paper): the fastest member of a family
with a given standard deviation whose expected episodic cost H * P_{v ~ rho_star}[v > v_limit] equals the benchmark cost
limit c_bar_0 = 25.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.stats import norm

TARGET_FAMILIES = ("gaussian", "uniform")
_SQRT3 = float(np.sqrt(3.0))


def standard_quantile(family: str, u: np.ndarray | float) -> np.ndarray | float:
    """Quantile function of the member of ``family`` with mean 0 and standard deviation 1."""
    if family == "gaussian":
        return norm.ppf(u)
    if family == "uniform":
        return _SQRT3 * (2.0 * np.asarray(u, dtype=np.float64) - 1.0)
    raise ValueError(f"family must be one of {TARGET_FAMILIES}, got {family!r}")


def target_mean(v_limit: float, sd_star: float, cost_limit: float, horizon: int, family: str = "gaussian") -> float:
    """Mean of the cost-derived velocity target of an earlier design (not reported in the paper).

    The target is the fastest member of ``family`` with standard deviation sd_star whose expected episodic cost
    H * P(v > v_limit) equals ``cost_limit`` (the benchmark cost limit c_bar_0 = 25 of earlier designs):

        mu_star = v_limit - q_{1 - c_bar_0 / H} * sd_star,

    where q_p is the p-quantile of the standardized member of the family (z_p for the Gaussian family).
    """
    return float(v_limit - standard_quantile(family, 1.0 - cost_limit / horizon) * sd_star)


class Target:
    """rho_star of a given family with mean ``mu_star`` and standard deviation ``sd_star``."""

    def __init__(self, family: str, mu_star: float, sd_star: float):
        if family not in TARGET_FAMILIES:
            raise ValueError(f"family must be one of {TARGET_FAMILIES}, got {family!r}")
        self.family = family
        self.mu_star = float(mu_star)
        self.sd_star = float(sd_star)

    def quantile(self, u: np.ndarray | float) -> np.ndarray | float:
        return self.mu_star + self.sd_star * standard_quantile(self.family, u)

    def integrated_quantile(self, u: np.ndarray) -> np.ndarray:
        """int_0^u F^{-1}(t) dt, where F^{-1} is the quantile function of rho_star."""
        u = np.asarray(u, dtype=np.float64)
        if self.family == "gaussian":
            with np.errstate(divide="ignore"):
                return self.mu_star * u - self.sd_star * norm.pdf(norm.ppf(u))     # phi(Phi^{-1}(0)) = phi(Phi^{-1}(1)) = 0
        return self.mu_star * u + _SQRT3 * self.sd_star * (u ** 2 - u)

    def w2_weights(self, n: int) -> np.ndarray:
        """m_i = int_{(i-1)/n}^{i/n} F^{-1}(u) du for i = 1, ..., n (see ``empirical_w2_squared``)."""
        g = self.integrated_quantile(np.arange(n + 1) / n)
        return g[1:] - g[:-1]

    def pdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.family == "gaussian":
            return norm.pdf(x, self.mu_star, self.sd_star)
        half = _SQRT3 * self.sd_star
        return np.where(np.abs(x - self.mu_star) <= half, 1.0 / (2.0 * half), 0.0)

    def tail(self, v_limit: float) -> float:
        """P_{v ~ rho_star}[v > v_limit]."""
        if self.family == "gaussian":
            return float(norm.sf(v_limit, self.mu_star, self.sd_star))
        half = _SQRT3 * self.sd_star
        return float(np.clip((self.mu_star + half - v_limit) / (2.0 * half), 0.0, 1.0))

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return np.asarray(self.quantile(rng.uniform(0.0, 1.0, n)))

    def label(self) -> str:
        """Description used in figure labels, e.g. ``U[2.25, 3.05]`` or ``N(2.23, 0.5^2)``."""
        if self.family == "gaussian":
            return f"N({self.mu_star:.2f}, {self.sd_star:.2g}^2)"
        half = _SQRT3 * self.sd_star
        return f"U[{self.mu_star - half:.2f}, {self.mu_star + half:.2f}]"


def empirical_w2_squared(samples: np.ndarray, target: Target, weights: np.ndarray | None = None) -> float:
    """Exact squared 2-Wasserstein distance between the empirical distribution of ``samples`` and rho_star.

    With the order statistics x_(1) <= ... <= x_(n), the optimal coupling maps x_(i) to the i-th n-quantile
    interval of rho_star, so that W_2^2 = (1/n) sum_i x_(i)^2 - 2 sum_i x_(i) m_i + mu_star^2 + sd_star^2 with
    the weights m_i of ``Target.w2_weights``. Unlike the closed form for two Gaussians evaluated at the window
    mean and standard deviation (``envs.gaussian_w2_squared``), it also compares the shape of the empirical
    distribution with rho_star.
    """
    x = np.sort(np.asarray(samples, dtype=np.float64))
    m = target.w2_weights(x.size) if weights is None else weights
    return max(0.0, float(np.mean(x ** 2) - 2.0 * np.dot(x, m) + target.mu_star ** 2 + target.sd_star ** 2))


def empirical_w2_squared_batch(samples: np.ndarray, target: Target) -> np.ndarray:
    """``empirical_w2_squared`` for every row of ``samples`` (shape (windows, n))."""
    x = np.sort(np.asarray(samples, dtype=np.float64), axis=1)
    m = target.w2_weights(x.shape[1])
    return np.maximum(0.0, np.mean(x ** 2, axis=1) - 2.0 * x @ m + target.mu_star ** 2 + target.sd_star ** 2)


@lru_cache(maxsize=None)
def _window_quantile(family: str, mu_star: float, sd_star: float, n: int, q: float, draws: int, seed: int) -> float:
    target = Target(family, mu_star, sd_star)
    rng = np.random.default_rng(seed)
    return float(np.quantile(empirical_w2_squared_batch(target.sample(rng, draws * n).reshape(draws, n), target), q))


def window_dissimilarity_quantile(target: Target, n: int, q: float = 0.5, draws: int = 400_000, seed: int = 0) -> float:
    """q-quantile of the dissimilarity W_2^2(rho_bar, rho_star) of a window of n i.i.d. samples from rho_star.

    The empirical distribution rho_bar of finitely many samples never equals rho_star, so the window dissimilarity
    of samples drawn from rho_star itself is positive; this function gives its q-quantile, estimated from ``draws``
    windows with a fixed seed (deterministic, cached). It is used only to characterize the threshold D_thr of the
    threshold reward r_tD (Appendix E), which is a fixed value (``config.D_THRESHOLD``).
    """
    return _window_quantile(target.family, target.mu_star, target.sd_star, int(n), float(q), int(draws), int(seed))


@lru_cache(maxsize=None)
def _window_within(family: str, mu_star: float, sd_star: float, n: int, threshold: float, draws: int, seed: int) -> float:
    target = Target(family, mu_star, sd_star)
    rng = np.random.default_rng(seed)
    return float(np.mean(empirical_w2_squared_batch(target.sample(rng, draws * n).reshape(draws, n), target) <= threshold))


def window_within_probability(target: Target, n: int, threshold: float, draws: int = 400_000, seed: int = 0) -> float:
    """Probability that the dissimilarity W_2^2(rho_bar, rho_star) of a window of n i.i.d. samples from rho_star is at
    most ``threshold``, estimated from ``draws`` windows with a fixed seed (deterministic, cached)."""
    return _window_within(target.family, target.mu_star, target.sd_star, int(n), float(threshold), int(draws), int(seed))
