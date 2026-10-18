"""Distributional-stability metrics computed from evaluation episodes (Section 5 and Appendix F).

The closed loop need not settle at the prescribed target rho_star, because the training objective
also contains the task reward and, for the methods with a cost constraint, the cost constraint.
Stability is therefore measured against a stable distribution rho_hat estimated from the evaluation
episodes themselves:

    rho_k    distribution of the forward velocity v_k at step k across evaluation episodes
             (``block = 1``); ``block > 1`` pools consecutive steps, which averages out changes of
             rho_k within a block, such as an oscillation of synchronized episodes
    rho_hat  distribution of v_k pooled over the steps k >= k0, estimated on the other half
             of the episodes (cross-fitting)
    D        1-Wasserstein distance between the two empirical distributions (exact in one dimension,
             ``wasserstein_1``)

All functions take ``values`` of shape (episodes, steps), with NaN after the end of an episode, except
``exceedance_probability``, which takes episodic returns: it estimates P[R >= R_thr], the probability
that the return R of an episode is at least a threshold R_thr, as the fraction of episodes whose return
reaches it (Section 5).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import wasserstein_distance

from .targets import Target


class Reference:
    """Sorted samples of a fixed reference distribution, prepared for repeated exact W1 computations."""

    def __init__(self, samples: np.ndarray):
        self.x = np.sort(np.asarray(samples, dtype=np.float64))
        self.m = self.x.size
        self.cumsum = np.concatenate([[0.0], np.cumsum(self.x)])

    def integrated_cdf(self, t: np.ndarray) -> np.ndarray:
        """int_{-inf}^t F(s) ds, where F is the empirical CDF of the reference."""
        k = np.searchsorted(self.x, t, side="right")
        return (k * t - self.cumsum[k]) / self.m


def wasserstein_1(sample: np.ndarray, ref: Reference) -> float:
    """Exact 1-Wasserstein distance int |F_sample - F_ref| dx between two one-dimensional empirical distributions.

    Equal to ``scipy.stats.wasserstein_distance(sample, ref.x)`` up to rounding, but it costs
    O(n log m) for a sample of size n once the reference of size m is prepared: on each interval
    between consecutive sample points, F_sample is a constant c, and the integral of |c - F_ref| is
    obtained from the integrated CDF of the reference, split where F_ref reaches c.
    """
    a = np.sort(np.asarray(sample, dtype=np.float64))
    n, m = a.size, ref.m
    edges = np.concatenate([[min(a[0], ref.x[0])], a, [max(a[-1], ref.x[-1])]])
    lo, hi = edges[:-1], edges[1:]                   # F_sample = c_i = i / n on [lo_i, hi_i)
    i = np.arange(n + 1)
    c = i / n
    j = -(-(i * m) // n)                             # ceil(i m / n): F_ref(x) >= c_i  <=>  x >= x_ref[j - 1]
    cross = np.where(j <= 0, -np.inf, ref.x[np.clip(j - 1, 0, m - 1)])
    r = np.clip(cross, lo, hi)
    g_lo, g_hi, g_r = ref.integrated_cdf(lo), ref.integrated_cdf(hi), ref.integrated_cdf(r)
    # rounding can make the sum slightly negative when the two distributions are equal (NaN stays NaN)
    return float(np.maximum(np.sum(c * (r - lo) - (g_r - g_lo) + (g_hi - g_r) - c * (hi - r)), 0.0))


def split_halves(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random split of the episode indices 0, ..., n - 1 into two disjoint halves."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return perm[: n // 2], perm[n // 2:]


def estimate_rho_hat(values: np.ndarray, k0: int) -> np.ndarray:
    """Samples of the stable distribution rho_hat: all finite values at steps k >= k0, pooled."""
    late = values[:, k0:]
    return late[np.isfinite(late)]


def stability_curve(values: np.ndarray, k0: int = 500, block: int = 1, min_samples: int = 4,
                    seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-fitted D(rho_k || rho_hat) at every step (``block = 1``) or for blocks of ``block`` steps.

    The episodes are split into two halves (``split_halves``); rho_hat is estimated on one half
    and rho_k on the other, and the two assignments are averaged, so that the metric is not zero
    by construction. Blocks with fewer than ``min_samples`` finite values are NaN.

    Returns ``(k_centers, D_values, survival)``, where ``survival[k]`` is the fraction of
    episodes still running at step k. If rho_hat has fewer than ``min_samples`` samples, the
    first two arrays are empty.
    """
    if block < 1:
        raise ValueError(f"block must be a positive integer, got {block}")
    n, K = values.shape
    D = _cross_fitted_blocks(values, k0, block, min_samples, seed, first=0)
    if D is None:
        return np.array([]), np.array([]), np.isfinite(values).mean(0)
    return np.arange(0, K, block) + block / 2, D, np.isfinite(values).mean(0)


def _cross_fitted_blocks(values: np.ndarray, k0: int, block: int, min_samples: int, seed: int,
                         first: int) -> np.ndarray | None:
    """D(rho_k || rho_hat) for the blocks starting at first, first + block, ... (None if rho_hat is too small)."""
    folds = split_halves(values.shape[0], seed)
    curves = []
    for a, b in (folds, folds[::-1]):
        late = estimate_rho_hat(values[a], k0)
        if late.size < min_samples:
            return None
        ref = Reference(late)
        cur = []
        for start in range(first, values.shape[1], block):
            sample = values[b, start:start + block]
            sample = sample[np.isfinite(sample)]
            cur.append(wasserstein_1(sample, ref) if sample.size >= min_samples else np.nan)
        curves.append(cur)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # a block that is NaN in both folds stays NaN
        return np.nanmean(np.array(curves), axis=0)


def sampling_floor(values: np.ndarray, k0: int, block: int, rng: np.random.Generator, draws: int = 10,
                   min_samples: int = 4, seed: int = 0) -> float:
    """Stationary reference value of the mean of D(rho_k || rho_hat) over the blocks with k >= k0 (Appendix F).

    Because rho_k and rho_hat are estimated from finitely many samples, D(rho_k || rho_hat) stays
    positive even for a stationary closed loop. The stationary reference value is the mean of D over
    the blocks with k >= k0, computed exactly as in ``stability_curve``, for a surrogate in which the
    segment k >= k0 of every episode is shifted circularly in time by an independent random amount. The
    distribution of the surrogate across episodes does not depend on k, because the phases of the
    episodes are independent. The surrogate has the same rho_hat as ``values`` and keeps the
    differences between episodes and the correlation between consecutive steps (except at the one
    wrap-around point per episode). The value is averaged over ``draws`` surrogates drawn from ``rng``.

    A ratio D_late / stationary reference value close to one is consistent with rho_k being stationary
    for k >= k0 up to sampling error; a ratio clearly above one indicates that rho_k still changes with
    k, for example through a drift or because the oscillations of the episodes are synchronized. A test
    that compares D_late with the surrogate values is approximate: a stationary process and its circular
    shift have the same one-step distributions, but not the same joint distribution at the wrap-around point.
    """
    late = values[:, k0:]
    floors = []
    for _ in range(draws):
        surrogate = np.full_like(values, np.nan, dtype=np.float64)
        for i, row in enumerate(late):
            finite = row[np.isfinite(row)]
            if finite.size:
                surrogate[i, k0:k0 + finite.size] = np.roll(finite, rng.integers(finite.size))
        first = -(-(2 * k0 - block) // (2 * block)) * block      # first block whose center is >= k0
        D = _cross_fitted_blocks(surrogate, k0, block, min_samples, seed, first=max(first, 0))
        floors.append(np.nanmean(D) if D is not None else np.nan)
    return float(np.mean(floors))


def distance_to_target(values: np.ndarray, mu_star: float, sd_star: float, k0: int = 500, n_ref: int = 20000,
                       family: str = "gaussian") -> float:
    """D(rho_hat || rho_star): 1-Wasserstein distance between the pooled values at k >= k0 and rho_star.

    rho_star, the member of ``family`` with mean mu_star and standard deviation sd_star (``targets.Target``),
    is represented by its ``n_ref`` midpoint quantiles F^{-1}((i + 1/2) / n_ref), i = 0, ..., n_ref - 1. The
    representation is deterministic and, unlike a sample of n_ref random draws, has no sampling error of its own.
    """
    target = Target(family, mu_star, sd_star)
    ref = np.asarray(target.quantile((np.arange(n_ref) + 0.5) / n_ref))
    late = estimate_rho_hat(values, k0)
    return wasserstein_distance(late, ref) if late.size else np.nan


def exceedance_probability(returns: np.ndarray, thresholds: float | np.ndarray) -> np.ndarray:
    """P[R >= R_thr] for each threshold R_thr: the fraction of the episodic returns that are at least R_thr.

    As a function of R_thr, this is the complementary distribution function of the return. A curve that
    falls over a wide range of thresholds belongs to a return that varies widely between episodes (or runs);
    the range of thresholds over which it falls from 0.9 to 0.1 is the 10-90% range of the return.
    NaN returns are ignored.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    r = r[np.isfinite(r)]
    t = np.atleast_1d(np.asarray(thresholds, dtype=np.float64))
    if r.size == 0:
        return np.full(t.shape, np.nan)
    return (r[None, :] >= t[:, None]).mean(axis=1)
