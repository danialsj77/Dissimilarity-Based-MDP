"""Aggregation of evaluation results into the quantities reported in Section 5 and Appendix F.

Results are stored as ``<root>/<env>/<method>/seed<s>/{config.json, training.csv, episodes.csv, policy.pt,
evaluation.npz}``. For each trained policy and noise level nu, ``summarize_run`` reports:

    ret, cost        episodic return R of the reward the method is trained on, without the dissimilarity term -beta * D:
                     the benchmark reward r^b_t = v_t - 0.1 ||a_t||^2 for the baselines (PPO, PPO-Lag, SAC, SAC-Lag and the
                     Distributional and Implicit PPO and SAC, with and without the cost constraint), and r^b_t + r_tD for DMDP and DMDP-Lag (threshold
                     reward r_tD = +d if D <= D_thr, else 0), and episodic cost (means over evaluation episodes)
    ret_benchmark    episodic return of the benchmark reward v_t - 0.1 ||a_t||^2 for every method (r_tD and the cost penalty
                     removed)
    ret_eva          episodic return of the evaluation reward r^eva_t, the same for every method: r^eva_t = r^b_t + r_tD if
                     v_t <= v_limit (c_t = 0), and r^eva_t = -v_limit if v_t > v_limit (c_t = 1), with r^b_t = v_t - 0.1 ||a_t||^2
                     and the threshold reward r_tD recomputed from the recorded dissimilarity; see ``evaluation_rewards``
    D_late           mean of D(rho_k || rho_hat) over k >= k0 (residual non-stationarity)
    D_early          mean of D(rho_k || rho_hat) over k < k_early (transient)
    D_target         D(rho_hat || rho_star), distance of the stable velocity distribution to the velocity target
    D_target_pitch   the same distance for the pitch angle of the torso and the pitch target
    pitch, flipped   mean absolute pitch angle over k >= k0 and fraction of these steps with |pitch| > pi/2 (torso
                     flipped, i.e., upside down)
    D_shift          D(rho_hat(nu) || rho_hat(0)), shift of the stable distribution caused by noise
    within           fraction of the steps k >= k0 whose window dissimilarity D(rho_bar_k || rho_star) is at most the
                     threshold D_thr of the threshold reward r_tD (Appendix E), i.e. P[D <= D_thr]
    mu_hat, sd_hat   mean and standard deviation of rho_hat
    survival         fraction of episodes still running at the last step

All dissimilarities are 1-Wasserstein distances (see ``metrics``). The recorded arrays are NaN after the end of
an episode; ``within``, ``flipped`` and the other statistics over steps use the recorded (finite) steps only.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from .config import D_BONUS, D_THRESHOLD, V_LIMIT, TrainConfig
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


def _read_csv(path: str) -> pd.DataFrame | None:
    """The CSV file at ``path``, or None if it does not exist or is empty (no episode was completed)."""
    return pd.read_csv(path) if os.path.exists(path) and os.path.getsize(path) > 0 else None


def load_training(root: str, env: str, method: str, seed: int) -> pd.DataFrame | None:
    """``training.csv`` of a run, or None if it does not exist or is empty."""
    return _read_csv(os.path.join(run_dir(root, env, method, seed), "training.csv"))


def load_episodes(root: str, env: str, method: str, seed: int) -> pd.DataFrame | None:
    """``episodes.csv`` of a run (one row per completed training episode), or None if it does not exist or is empty."""
    return _read_csv(os.path.join(run_dir(root, env, method, seed), "episodes.csv"))


def penalty_per_cost(cfg: dict[str, Any]) -> float:
    """kappa^2, the reward that the cost penalty (kappa * c_t)^2 of a run removes at every step with c_t = 1 (0 for every
    reported method, which is trained on the benchmark reward; 4 for the runs of the design pilot, kappa = COST_PENALTY = 2)."""
    return float(cfg.get("cost_penalty", 0.0)) ** 2


def episode_returns(ev: Evaluation, nu: float) -> np.ndarray:
    """Episodic return R of every evaluation episode at nu: the return of the reward the method is trained on, without the
    dissimilarity term -beta * D.

    The recorded reward is the training reward of the run without -beta * D: r_t + r_tD with
    r_t = v_t - (kappa * c_t)^2 - 0.1 ||a_t||^2, i.e. the benchmark reward plus r_tD for the DMDP methods (kappa = 0, d = 10),
    the benchmark reward v_t - 0.1 ||a_t||^2 for the baselines (kappa = 0, d = 0), and the reward with the cost penalty
    kappa = 2 for the runs of the design pilot. The return of the training episodes is the column ``ret``
    of ``episodes.csv`` and, as a mean over the most recent episodes, ``ep_return`` of ``training.csv``.
    """
    return np.nansum(ev[nu]["reward"], axis=1)


def benchmark_returns(ev: Evaluation, nu: float, cfg: dict[str, Any]) -> np.ndarray:
    """Episodic return of the benchmark reward v_t - 0.1 ||a_t||^2 of every evaluation episode at nu, for any method:
    the recorded reward without r_tD and with the cost penalty of the run (``config.json``) added back."""
    rec = ev[nu]
    bonus = rec["bonus"] if "bonus" in rec else 0.0
    return np.nansum(rec["reward"] - bonus + penalty_per_cost(cfg) * rec["cost"], axis=1)


def dissimilarity_threshold(cfg: dict[str, Any]) -> float:
    """Threshold D_thr of the threshold reward of a run (``config.json``), the same for every window length. Runs without
    r_tD (d = 0), whose training does not depend on D_thr, use the current D_THRESHOLD, so that the fraction of the steps
    with D <= D_thr is measured with the same threshold for every method."""
    if not cfg.get("d_bonus", 0.0):
        return D_THRESHOLD
    return float(cfg.get("d_threshold", D_THRESHOLD))


def threshold_reward(rec: dict[str, np.ndarray], cfg: dict[str, Any], d: float = D_BONUS) -> np.ndarray:
    """r_tD = +d if D(rho_bar_t || rho_star) <= D_thr and 0 otherwise at every step, for any method (from the recorded
    window dissimilarity, with the threshold of training)."""
    return np.where(rec["dissimilarity"] <= dissimilarity_threshold(cfg), d, 0.0)


def evaluation_penalty(cfg: dict[str, Any]) -> float:
    """r_penalty, the evaluation reward of a step with v_t > v_limit: v_limit (``config.json``), the largest benchmark reward
    v_t - 0.1 ||a_t||^2 that a step with v_t <= v_limit can earn."""
    return float(cfg.get("v_limit", V_LIMIT))


def evaluation_rewards(rec: dict[str, np.ndarray], cfg: dict[str, Any]) -> np.ndarray:
    """Evaluation reward r^eva_t at every step of every evaluation episode, the same for every method:

        r^eva_t = r^b_t + r_tD  if v_t <= v_limit (c_t = 0),      r^eva_t = -r_penalty = -v_limit  if v_t > v_limit (c_t = 1),

    with the benchmark reward r^b_t = v_t - 0.1 ||a_t||^2 and the threshold reward r_tD = d if D(rho_bar_t || rho_star) <= D_thr
    and 0 otherwise (d = D_BONUS, ``threshold_reward``). r^b_t is the recorded reward without the recorded r_tD and with the
    cost penalty of the run added back (see ``benchmark_returns``), and r_tD is recomputed from the recorded window
    dissimilarity, so that the baselines, which are trained without r_tD, are evaluated with the same reward as the DMDP
    methods. r^eva_t contains neither the dissimilarity term nor a Lagrangian term. The steps after the end of an episode
    are NaN."""
    bonus = rec["bonus"] if "bonus" in rec else 0.0
    benchmark = rec["reward"] - bonus + penalty_per_cost(cfg) * rec["cost"]
    unsafe = np.nan_to_num(rec["cost"], nan=0.0) > 0.5
    return np.where(unsafe, -evaluation_penalty(cfg), benchmark + threshold_reward(rec, cfg))


def evaluation_returns(ev: Evaluation, nu: float, cfg: dict[str, Any]) -> np.ndarray:
    """Episodic return of the evaluation reward ``evaluation_rewards`` of every evaluation episode at nu, for any method."""
    return np.nansum(evaluation_rewards(ev[nu], cfg), axis=1)


def _finite_fraction(x: np.ndarray, condition: Callable[[np.ndarray], np.ndarray]) -> float:
    """Fraction of the finite entries of ``x`` that satisfy ``condition`` (NaN if there is none).

    NaN entries, the steps after the end of an episode, are ignored; they do not count as failing the condition.
    """
    x = x[np.isfinite(x)]
    return float(np.mean(condition(x))) if x.size else np.nan


def _pitch_summary(rec: dict[str, np.ndarray], cfg: dict[str, Any], k0: int) -> dict[str, float]:
    """Statistics of the pitch angle of the torso for k >= k0 (NaN for runs stored without the pitch)."""
    if "pitch" not in rec:
        return dict(pitch=np.nan, flipped=np.nan, D_target_pitch=np.nan)
    pitch = rec["pitch"]
    return dict(pitch=float(np.nanmean(np.abs(pitch[:, k0:]))),
                flipped=_finite_fraction(np.abs(pitch[:, k0:]), lambda a: a > np.pi / 2),
                D_target_pitch=float(distance_to_target(pitch, cfg.get("pitch_mu_star", TrainConfig.pitch_mu_star),
                                                        cfg.get("pitch_sd_star", TrainConfig.pitch_sd_star), k0=k0,
                                                        family="uniform")))


def summarize_run(ev: Evaluation, cfg: dict[str, Any], k0: int = 500, block: int = 1, k_early: int = 200
                  ) -> tuple[list[dict[str, float]], dict[float, tuple[np.ndarray, np.ndarray]]]:
    """Summary rows (one per noise level) and stability curves {nu: (k_centers, D)} of one run."""
    rows, curves = [], {}
    ref0 = estimate_rho_hat(ev[0.0]["velocity"], k0) if 0.0 in ev else None
    threshold = dissimilarity_threshold(cfg)
    for nu, rec in sorted(ev.items()):
        v = rec["velocity"]
        centers, D, survival = stability_curve(v, k0=k0, block=block)
        late = estimate_rho_hat(v, k0)
        rows.append(dict(
            nu=nu,
            ret=float(np.nanmean(episode_returns(ev, nu))),
            ret_benchmark=float(np.nanmean(benchmark_returns(ev, nu, cfg))),
            ret_eva=float(np.nanmean(evaluation_returns(ev, nu, cfg))),
            cost=float(np.nanmean(np.nansum(rec["cost"], 1))),
            D_late=float(np.nanmean(D[centers >= k0])),
            D_early=float(np.nanmean(D[centers < k_early])),
            D_target=float(distance_to_target(v, cfg["mu_star"], cfg["sd_star"], k0=k0,
                                              family=cfg.get("target_family", TrainConfig.target_family))),
            D_shift=float(wasserstein_distance(late, ref0)) if ref0 is not None else np.nan,
            within=_finite_fraction(rec["dissimilarity"][:, k0:], lambda d: d <= threshold),
            **_pitch_summary(rec, cfg, k0),
            mu_hat=float(np.mean(late)),
            sd_hat=float(np.std(late)),
            survival=float(survival[-1]),
        ))
        curves[nu] = (centers, D)
    return rows, curves


def summarize(root: str, env: str, methods: Iterable[str], seeds: Iterable[int], k0: int = 500, block: int = 1,
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
