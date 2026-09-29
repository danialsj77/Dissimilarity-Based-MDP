"""Distributional-stability metrics computed from evaluation episodes (Section 7.1, Appendix D).

The closed loop need not settle at the prescribed target rho_star, because the training objective
also contains the task reward and the cost constraint. Stability is therefore measured against a
stable distribution rho_hat estimated from the evaluation episodes themselves:

    rho_k    distribution of the forward velocity v_k at step k across evaluation episodes,
             pooled over blocks of ``block`` consecutive steps
    rho_hat  distribution of v_k pooled over the late steps k >= k0, estimated on the other half
             of the episodes (cross-fitting)
    D        1-Wasserstein distance between the two empirical distributions (exact in one dimension)

All functions take ``values`` of shape (episodes, steps), with NaN after the end of an episode.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance


def split_halves(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random split of the episode indices 0, ..., n - 1 into two disjoint halves."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return perm[: n // 2], perm[n // 2:]


def estimate_rho_hat(values: np.ndarray, k0: int) -> np.ndarray:
    """Samples of the stable distribution rho_hat: all finite values at steps k >= k0, pooled."""
    late = values[:, k0:]
    return late[np.isfinite(late)]


def stability_curve(values: np.ndarray, k0: int = 500, block: int = 10, min_samples: int = 4,
                    seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-fitted D(rho_k || rho_hat) for blocks of ``block`` consecutive steps.

    The episodes are split into two halves (``split_halves``); rho_hat is estimated on one half
    and rho_k on the other, and the two assignments are averaged, so that the metric is not zero
    by construction. Blocks with fewer than ``min_samples`` finite values are NaN.

    Returns ``(k_centers, D_values, survival)``, where ``survival[k]`` is the fraction of
    episodes still running at step k. If rho_hat has fewer than ``min_samples`` samples, the
    first two arrays are empty.
    """
    n, K = values.shape
    folds = split_halves(n, seed)
    curves = []
    for a, b in (folds, folds[::-1]):
        ref = estimate_rho_hat(values[a], k0)
        if ref.size < min_samples:
            return np.array([]), np.array([]), np.isfinite(values).mean(0)
        cur = []
        for start in range(0, K, block):
            sample = values[b, start:start + block]
            sample = sample[np.isfinite(sample)]
            cur.append(wasserstein_distance(sample, ref) if sample.size >= min_samples else np.nan)
        curves.append(cur)
    centers = np.arange(0, K, block) + block / 2
    return centers, np.nanmean(np.array(curves), axis=0), np.isfinite(values).mean(0)


def sampling_floor(values: np.ndarray, k0: int, block: int, rng: np.random.Generator) -> float:
    """Sampling floor of D(rho_k || rho_hat) for a stationary process (Appendix D).

    Because rho_k and rho_hat are estimated from finitely many samples, D(rho_k || rho_hat) stays
    positive even for a stationary closed loop. The floor is the mean over all blocks of
    ``stability_curve`` applied to a surrogate that is stationary by construction but keeps the
    differences between episodes: every step of episode i is an independent draw (with
    replacement) from the late-time values (k >= k0) of the same episode i. The surrogate has the
    same number of steps as ``values``; it does not keep the correlation between consecutive steps.

    A ratio D_late / floor close to one indicates stationarity up to sampling error. The surrogate
    is drawn from ``rng`` episode by episode, so a shared generator gives reproducible results when
    the calls are made in a fixed order.
    """
    late = values[:, k0:]
    surrogate = np.stack([rng.choice(row[np.isfinite(row)], size=values.shape[1]) for row in late])
    _, D, _ = stability_curve(surrogate, k0=k0, block=block)
    return float(np.nanmean(D))


def distance_to_target(values: np.ndarray, mu_star: float, sd_star: float, k0: int = 500, n_ref: int = 20000,
                       seed: int = 0) -> float:
    """D(rho_hat || rho_star): 1-Wasserstein distance between the pooled late-time values and rho_star.

    rho_star = N(mu_star, sd_star^2) is represented by ``n_ref`` samples.
    """
    rng = np.random.default_rng(seed)
    ref = rng.normal(mu_star, sd_star, n_ref)
    late = estimate_rho_hat(values, k0)
    return wasserstein_distance(late, ref) if late.size else np.nan
