"""Aggregation of evaluation results into the quantities reported in Section 7 and Appendix D.

Results are stored as ``<root>/<env>/<method>/seed<s>/{config.json, training.csv, policy.pt,
evaluation.npz}``. For each trained policy and noise level nu, ``summarize_run`` reports:

    ret, cost        episodic return and cost (means over evaluation episodes)
    D_late           mean of D(rho_k || rho_hat) over k >= k0 (residual non-stationarity)
    D_early          mean of D(rho_k || rho_hat) over k < k_early (transient)
    D_target         D(rho_hat || rho_star), distance of the stable distribution to the target
    D_shift          D(rho_hat(nu) || rho_hat(0)), shift of the stable distribution caused by noise
    mu_hat, sd_hat   mean and standard deviation of rho_hat
    survival         fraction of episodes still running at the last step

All dissimilarities are 1-Wasserstein distances (see ``metrics``).
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from .metrics import distance_to_target, estimate_rho_hat, stability_curve

Evaluation = dict[float, dict[str, np.ndarray]]     # noise level nu -> recorded arrays


def run_dir(root: str, env: str, method: str, seed: int) -> str:
    """Output directory of one run."""
    return os.path.join(root, env, method, f"seed{seed}")


def load_evaluation(root: str, env: str, method: str, seed: int) -> Evaluation | None:
    """``evaluation.npz`` of a run as {nu: {name: array}}, or None if the run was not evaluated."""
    path = os.path.join(run_dir(root, env, method, seed), "evaluation.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path)
    out: Evaluation = {}
    for key in z.files:
        nu, name = key.split("/")                       # e.g. "nu0.20/velocity"
        out.setdefault(float(nu[2:]), {})[name] = z[key]
    return out


def load_config(root: str, env: str, method: str, seed: int) -> dict[str, Any]:
    """``config.json`` of a run."""
    with open(os.path.join(run_dir(root, env, method, seed), "config.json")) as f:
        return json.load(f)


def load_training(root: str, env: str, method: str, seed: int) -> pd.DataFrame | None:
    """``training.csv`` of a run, or None if it does not exist."""
    path = os.path.join(run_dir(root, env, method, seed), "training.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def summarize_run(ev: Evaluation, cfg: dict[str, Any], k0: int = 500, block: int = 10, k_early: int = 200
                  ) -> tuple[list[dict[str, float]], dict[float, tuple[np.ndarray, np.ndarray]]]:
    """Summary rows (one per noise level) and stability curves {nu: (k_centers, D)} of one run."""
    rows, curves = [], {}
    ref0 = estimate_rho_hat(ev[0.0]["velocity"], k0) if 0.0 in ev else None
    for nu, rec in sorted(ev.items()):
        v = rec["velocity"]
        centers, D, survival = stability_curve(v, k0=k0, block=block)
        late = estimate_rho_hat(v, k0)
        rows.append(dict(
            nu=nu,
            ret=float(np.nanmean(np.nansum(rec["reward"], 1))),
            cost=float(np.nanmean(np.nansum(rec["cost"], 1))),
            D_late=float(np.nanmean(D[centers >= k0])),
            D_early=float(np.nanmean(D[centers < k_early])),
            D_target=float(distance_to_target(v, cfg["mu_star"], cfg["sd_star"], k0=k0)),
            D_shift=float(wasserstein_distance(late, ref0)) if ref0 is not None else np.nan,
            mu_hat=float(np.mean(late)),
            sd_hat=float(np.std(late)),
            survival=float(survival[-1]),
        ))
        curves[nu] = (centers, D)
    return rows, curves


def summarize(root: str, env: str, methods: Iterable[str], seeds: Iterable[int], k0: int = 500, block: int = 10,
              k_early: int = 200) -> tuple[pd.DataFrame, dict[tuple[str, int, float], tuple[np.ndarray, np.ndarray]]]:
    """Per-seed table (one row per method, seed and nu) and the stability curves keyed by (method, seed, nu).

    Runs without ``evaluation.npz`` are skipped.
    """
    rows, curves = [], {}
    for m in methods:
        for s in seeds:
            ev = load_evaluation(root, env, m, s)
            if ev is None:
                continue
            cfg = load_config(root, env, m, s)
            r, c = summarize_run(ev, cfg, k0=k0, block=block, k_early=k_early)
            for row in r:
                row.update(method=m, seed=s)
            rows.extend(r)
            for nu, cur in c.items():
                curves[(m, s, nu)] = cur
    return pd.DataFrame(rows), curves


def aggregate(df: pd.DataFrame, keys: Iterable[str] = ("method", "nu")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean and standard error over seeds, grouped by ``keys``."""
    g = df.groupby(list(keys))
    mean = g.mean(numeric_only=True).drop(columns=["seed"], errors="ignore")
    sem = g.sem(numeric_only=True).drop(columns=["seed"], errors="ignore")
    return mean, sem
