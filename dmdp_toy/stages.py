"""The five stages of the computable example (Appendix D), each resumable from the files it writes.

    true       U~ ~ U_star^R by value iteration, its error indicators, optimal trajectories, particle check (Theorem 2)
    offline    one file per (level, run): learned model, value, policy and the quantities of Theorem 3
    online     plug-in deployment of the deployed offline policy (Theorem 4); needs the calibration runs of the top level
    modelfree  one file per run: model-free learner and the quantities of Theorem 5
    outputs    calibration, checks, summary.json, the two LaTeX tables and the figure

Runs 0-48 calibrate the residuals and run 49 (its own seed) is deployed. Seeds:
SeedSequence([MASTER_SEED, code, level, run]) with code 2 = particle check, 3 = offline, 4 = online, 5 = model-free.
Existing run files are skipped, so a stage can be repeated after an interruption. online.npz records a checksum of
the deployed offline policy and of U~ it was computed from; ``stage_online`` and ``summarize`` stop if they changed.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from dataclasses import asdict, dataclass

import numpy as np

from . import certify as C
from . import modelfree as MF
from . import offline as OF
from . import online as ON
from . import system as S
from . import vi
from .grids import Grid, graded_actions

MASTER_SEED = 20261006
DEPLOYED = 49
ALPHA = 0.1                    # 1 - q of the order statistic
EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class Settings:
    name: str = "full"
    out: str = "results_toy"
    figdir: str = "figures"
    use_tex: bool = True
    true_grid: tuple = (0.003, 0.15, 0.06, 0.03)                  # 111 x 62 nodes
    true_actions: tuple = ((0.002, 0.25, 0.02), (0.002, 0.25, 0.05))
    ref_grid: tuple = (0.006, 0.3, 0.12, 0.06)                     # 57 x 33 nodes
    ref_actions: tuple = ((0.004, 0.25, 0.04), (0.004, 0.25, 0.1))
    test_grid: tuple = (161, 61)
    n_near: int = 2000
    fine_uniform: int = 60_000                                     # independent check set of the deployed runs
    fine_near: int = 20_000
    levels: tuple = (1, 2, 3, 4)
    runs: tuple = tuple(range(50))
    checkpoints: tuple = MF.CHECKPOINTS
    online_grid: tuple = (161, 61)
    online_per_start: int = 25
    online_steps: int = 40
    mf_per_start: int = 5
    particles: int = 200_000
    traj_steps: int = 12


def quick_settings(out="results_toy_quick", figdir="figures", use_tex=False):
    return Settings(name="quick", out=out, figdir=figdir, use_tex=use_tex,
                    true_grid=(0.006, 0.3, 0.12, 0.06), true_actions=((0.004, 0.25, 0.04), (0.004, 0.25, 0.1)),
                    ref_grid=(0.01, 0.3, 0.15, 0.075), ref_actions=((0.01, 0.35, 0.03), (0.01, 0.35, 0.1)),
                    test_grid=(41, 16), n_near=200, fine_uniform=4000, fine_near=1000, levels=(1,),
                    runs=(0, 1, DEPLOYED),
                    checkpoints=(20,), online_grid=(16, 10), online_per_start=2, online_steps=10, mf_per_start=1)


# ----------------------------------------------------------------------------------------------- files
def _path(cfg, *parts):
    return os.path.join(cfg.out, *parts)


def _offline_path(cfg, level, run):
    return _path(cfg, "offline", f"L{level}", f"run_{run:02d}.npz")


def _modelfree_path(cfg, run):
    return _path(cfg, "modelfree", f"run_{run:02d}.npz")


def _save(path, **arrays):
    """Atomic write, so that an interrupted stage never leaves a partial run file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _load(path):
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def _seed(code, level, run):
    return np.random.SeedSequence([MASTER_SEED, code, level, run])


def _calib_runs(cfg):
    return [r for r in cfg.runs if r != DEPLOYED]


def _tests(cfg):
    return C.test_points(*cfg.test_grid, n_near=cfg.n_near)


def _fine_tests(cfg):
    return C.independent_points(cfg.fine_uniform, cfg.fine_near)


def _portable(path):
    """``path`` relative to the Experiments folder (no local absolute paths in summary.json); outside the project
    folder only its last component."""
    try:
        rel = os.path.relpath(path, EXPERIMENTS)
    except ValueError:                                  # another drive
        return os.path.basename(os.path.normpath(path))
    return os.path.basename(os.path.normpath(path)) if rel.startswith(os.path.join(os.pardir, os.pardir)) else rel


def _online_source(cfg, level):
    """Checksum of the inputs of the online stage (deployed policy of ``level`` and U~), and the run file's mtime."""
    dep, tv = _offline_path(cfg, level, DEPLOYED), _path(cfg, "true_value.npz")
    h = hashlib.sha256()
    for path, keys in ((dep, ("a_hat", "theta", "grid_mu", "grid_vs")), (tv, ("Z", "mu", "sg"))):
        d = _load(path)
        for k in keys:
            h.update(np.ascontiguousarray(d[k], dtype=np.float64).tobytes())
    return h.hexdigest(), os.path.getmtime(dep)


def _check_online(cfg, on):
    """Stops if online.npz was computed from another deployed policy or another U~ than the current files. The
    checksum is compared, not the mtime (copying the files changes the mtime, not the content)."""
    sha, _ = _online_source(cfg, int(on["level"]))
    if "source_sha" not in on or str(on["source_sha"]) != sha:
        raise SystemExit(f"{_path(cfg, 'online.npz')} was computed from another deployed offline policy (level "
                         f"{int(on['level'])}, run {DEPLOYED}) or another U~: delete it and run --stage online again")


def _true_value(cfg):
    path = _path(cfg, "true_value.npz")
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing: run --stage true first")
    d = _load(path)
    return vi.ValueFunction(Grid(d["mu"], d["sg"]), d["Z"]), d


def _schedule(cfg):
    """M_t for the plug-in runs, cached in schedule.json (delta_on = 0.05, alpha_on = 0.05)."""
    path = _path(cfg, "schedule.json")
    if os.path.exists(path):
        with open(path) as fh:
            d = json.load(fh)
        if len(d["M_t"]) >= cfg.online_steps:
            return np.array(d["M_t"][:cfg.online_steps]), {int(k): v for k, v in d["extra"].items()}
    sched, extra = ON.sample_size_schedule(cfg.online_steps, extra=(100, 1000))
    os.makedirs(cfg.out, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"M_t": sched.tolist(), "extra": {str(k): v for k, v in extra.items()}}, fh)
    return sched, extra


def _plugin_starts(per_start):
    return [w for w in OF.LIMIT_STARTS for _ in range(per_start)]


# ----------------------------------------------------------------------------------------------- stage true
def stage_true(cfg):
    path = _path(cfg, "true_value.npz")
    if os.path.exists(path):
        print(f"  {path} exists, skipped")
        return
    t_stage = time.time()
    grid = Grid.graded(*cfg.true_grid)
    actions = graded_actions(S.A_STAR, *cfg.true_actions)
    res = vi.value_iteration_ratio(grid, actions)
    print(f"  value iteration: grid {grid.shape[0]} x {grid.shape[1]}, {len(actions)} actions, "
          f"{res.iterations} iterations, change {res.change:.1e}, {res.seconds:.1f} s")
    ref_grid = Grid.graded(*cfg.ref_grid)
    ref = vi.value_iteration_ratio(ref_grid, graded_actions(S.A_STAR, *cfg.ref_actions))
    U, U_ref = vi.ValueFunction(grid, res.Z), vi.ValueFunction(ref_grid, ref.Z)
    mu, vs = _tests(cfg)
    d, Ut = S.dist2(mu, vs), U(mu, vs)
    pos = d > 0
    t0 = time.time()
    e_bell = float(np.max(vi.bellman_residual(U, mu, vs)))
    U_bar = vi.rollout_upper_bound(U, mu, vs)
    _, _, inside = S.deadbeat(mu, vs)
    ind = dict(e_bell=e_bell, e_up=float(np.max(np.maximum(U_bar - Ut, 0))),
               e_over=float(np.max(np.maximum(Ut - U_bar, 0))), e_grid=float(np.max(np.abs(Ut - U_ref(mu, vs)))),
               e_R=float(np.max(np.abs(Ut[inside] - d[inside]))) if inside.any() else 0.0)
    ind["eps_num"] = max(ind["e_bell"], ind["e_up"], ind["e_over"], ind["e_grid"])
    policy = lambda m, s: vi.greedy(U, m, s)[0]                                          # noqa: E731
    starts = np.array([(S.MU_LO, S.VS_HI), (S.MU_HI, S.VS_LO), (1.0, S.VS_HI)])
    trajs = np.array([vi.trajectory(policy, w0, cfg.traj_steps) for w0 in starts])
    # particle check: the sample moments follow the parameter trajectory (exact closure)
    rng = np.random.default_rng(_seed(2, 0, 0))
    x = S.MU_LO + S.VS_HI * rng.standard_normal(cfg.particles)
    w = np.array([[S.MU_LO, S.VS_HI]])
    err_mean = err_std = 0.0
    for _ in range(6):
        a = policy(w[:, 0], w[:, 1])[0]
        x = S.step_particles(x, a[0], a[1], rng)
        w = np.array([S.f(w[0, 0], w[0, 1], a[0], a[1])])
        err_mean, err_std = max(err_mean, abs(x.mean() - w[0, 0])), max(err_std, abs(x.std() - w[0, 1]))
    print(f"  error indicators ({time.time() - t0:.1f} s): " + ", ".join(f"{k} {v:.1e}" for k, v in ind.items()))
    _save(path, mu=grid.mu, sg=grid.vs, Z=res.Z, K=res.K, A=actions, mu_medium=ref_grid.mu, sg_medium=ref_grid.vs,
          Z_medium=ref.Z, iterations=res.iterations, change=res.change, seconds=res.seconds,
          ratio_min=float(np.min(Ut[pos] / d[pos])), ratio_max=float(np.max(Ut[pos] / d[pos])), n_test=mu.size,
          traj_starts=starts, trajs=trajs, particle_err_mean=err_mean, particle_err_std=err_std,
          stage_seconds=time.time() - t_stage, **ind)


# ----------------------------------------------------------------------------------------------- stage offline
def stage_offline(cfg, levels=None, runs=None):
    U, _ = _true_value(cfg)
    tests, mtests = _tests(cfg), C.model_test_points()
    fine = None
    for level in levels or cfg.levels:
        B, M = OF.LEVELS[level]
        for run in runs if runs is not None else cfg.runs:
            path = _offline_path(cfg, level, run)
            if os.path.exists(path):
                continue
            t0 = time.time()
            learned = OF.learn(B, M, np.random.default_rng(_seed(3, level, run)))
            if run == DEPLOYED and fine is None:
                fine = _fine_tests(cfg)
            ev = OF.evaluate_run(learned, U, tests, mtests, fine if run == DEPLOYED else None)
            ev.update(seconds=time.time() - t0, t_fit=learned["t_fit"], vi_seconds=learned["vi"].seconds)
            if run == DEPLOYED:
                ev.update(OF.deployed_arrays(learned))
            _save(path, **ev)
            print(f"  L{level} run {run:2d}: Delta {ev['Delta']:.3f}, c_hat {ev['c_hat']:.3f}, e_f {ev['e_f']:.4f}, "
                  f"offset {ev['offset']:.1e}, {ev['seconds']:.1f} s")


def _offline_files(cfg, level, require=True):
    files = {r: _offline_path(cfg, level, r) for r in cfg.runs}
    missing = [r for r, p in files.items() if not os.path.exists(p)]
    if missing and require:
        raise SystemExit(f"offline level {level}: runs {missing} are missing: run --stage offline first")
    return {r: _load(p) for r, p in files.items() if os.path.exists(p)}


def calibrate(scores):
    """Delta_bar = max of the calibration scores (``certify.order_statistic``) and its confidence 1 - (1 - ALPHA)^n.

    The numerical error eps_num of U~ is not added; it is reported separately (summary['true']['eps_num'] and row 4
    of the hypothesis table)."""
    bar, conf = C.order_statistic(scores, 1 - ALPHA)
    return dict(Delta_bar=bar, confidence=conf, n_calib=len(scores))


def _offline_calibration(cfg, level):
    runs = _offline_files(cfg, level)
    return runs, calibrate([float(runs[r]["score"]) for r in _calib_runs(cfg)])


# ----------------------------------------------------------------------------------------------- stage online
def stage_online(cfg):
    path = _path(cfg, "online.npz")
    if os.path.exists(path):
        _check_online(cfg, _load(path))
        print(f"  {path} exists, skipped")
        return
    t_stage = time.time()
    U, _ = _true_value(cfg)
    level = max(cfg.levels)
    runs, cal = _offline_calibration(cfg, level)
    sha, mtime = _online_source(cfg, level)
    policy = OF.load_policy(runs[DEPLOYED])
    sched, extra = _schedule(cfg)
    d_on = np.array([ON.plugin_perturbation(U, policy, dl, cfg.online_grid) for dl in ON.DELTAS])
    d_on_used = float(d_on[ON.DELTAS.index(ON.DELTA_ON)])
    starts = _plugin_starts(cfg.online_per_start)
    sim = ON.simulate_plugin(policy, U, starts, [_seed(4, 0, j) for j in range(len(starts))], sched, OF.C_SIGMA)
    bound = cal["Delta_bar"] + d_on_used
    viol = int(np.sum(sim["dec"][sim["event"]] > bound)) if np.isfinite(bound) else 0
    _save(path, level=level, Delta_on=d_on, deltas=np.array(ON.DELTAS), Delta_on_used=d_on_used, M_t=sched,
          M_extra_t=np.array(list(extra)), M_extra=np.array(list(extra.values())), starts=np.array(starts),
          source_sha=sha, source_mtime=mtime, stage_seconds=time.time() - t_stage, **sim)
    print(f"  Delta_on(0.05) {d_on_used:.2e}, Delta_bar_off {cal['Delta_bar']:.3f}, events {int(sim['event'].sum())}"
          f"/{len(starts)}, violating steps {viol}, M_0 {sched[0]}, M_{len(sched) - 1} {sched[-1]}")


# ----------------------------------------------------------------------------------------------- stage modelfree
def stage_modelfree(cfg, runs=None):
    tests = _tests(cfg)
    fine = None
    for run in runs if runs is not None else cfg.runs:
        path = _modelfree_path(cfg, run)
        if os.path.exists(path):
            continue
        t0 = time.time()
        learned = MF.run_learner(np.random.default_rng(_seed(5, 0, run)), cfg.checkpoints)
        out = {}
        for n, (gp, policy) in learned.items():
            q = MF.decrease_quantities(gp, policy, *tests)
            lim, spread = OF.limit_distance(policy)
            for k in ("Delta_dec", "Delta_dec_raw", "e_n", "viol_radius", "Delta_min"):
                out[f"n{n}_{k}"] = q[k]
            out[f"n{n}_limit"], out[f"n{n}_limit_spread"], out[f"n{n}_offset"] = lim, spread, MF.offset(policy)
            out[f"n{n}_theta_gp"] = gp.theta
            if run == DEPLOYED:
                fine = fine if fine is not None else _fine_tests(cfg)
                qf = MF.decrease_quantities(gp, policy, *fine)
                out[f"n{n}_fine"] = np.array([qf[k] for k in MF.FINE_KEYS])
                out[f"n{n}_fine_change"] = np.array([C.relative_change(qf[k], q[k]) for k in MF.FINE_KEYS])
                out[f"n{n}_Delta_omega"] = ON.plugin_perturbation(S.dist2, policy, ON.DELTA_ON, cfg.online_grid)
                starts = _plugin_starts(cfg.mf_per_start)
                sched, _ = _schedule(cfg)
                sim = ON.simulate_plugin(policy, S.dist2, starts, [_seed(5, n, j) for j in range(len(starts))],
                                         sched, MF.C_SIGMA)
                for k in ("dec", "err", "event"):
                    out[f"n{n}_plug_{k}"] = sim[k]
                out.update({f"n{n}_theta": policy.theta, **gp.state(f"n{n}_gp_")})
                traj = MF.deployed_trajectory(policy)
                out[f"n{n}_traj_D"] = S.dist2(traj[:, 0], traj[:, 1])
                if n == max(cfg.checkpoints):
                    out["r"], out["true_dec"] = q["r"], q["true_dec"]
        out["seconds"] = time.time() - t0
        _save(path, **out)
        n = max(cfg.checkpoints)
        print(f"  run {run:2d}: Delta_dec {out[f'n{n}_Delta_dec']:.3f} (raw {out[f'n{n}_Delta_dec_raw']:.3f}), "
              f"e_n {out[f'n{n}_e_n']:.1e}, {out['seconds']:.1f} s")


# ----------------------------------------------------------------------------------------------- stage outputs
def _f(x):
    """JSON-safe float (infinity as the string 'inf')."""
    x = float(x)
    return x if np.isfinite(x) else "inf"


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in (o.tolist() if isinstance(o, np.ndarray) else o)]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return _f(o)
    return o


def summarize(cfg):
    """Collects every number of the tables and the checks into one dictionary (written as summary.json)."""
    _, tv = _true_value(cfg)
    eps = float(tv["eps_num"])
    true = {k: float(tv[k]) for k in ("e_bell", "e_up", "e_over", "e_grid", "e_R", "eps_num", "ratio_min",
                                      "ratio_max", "particle_err_mean", "particle_err_std", "seconds", "change")}
    true.update(grid=list(Grid(tv["mu"], tv["sg"]).shape), n_actions=len(tv["A"]), iterations=int(tv["iterations"]),
                n_test=int(tv["n_test"]))
    trajs = {}
    for w0, tr in zip(tv["traj_starts"], tv["trajs"]):
        dk = S.dist2(tr[:, 0], tr[:, 1])
        hit = np.nonzero(dk < 1e-12)[0]
        trajs[f"({w0[0]:g}, {w0[1]:.4f})"] = dict(D_k=dk, first_k_at_target=int(hit[0]) if hit.size else None)
    true["trajectories"] = trajs

    offline = {}
    for level in cfg.levels:
        runs, cal = _offline_calibration(cfg, level)
        B, M = OF.LEVELS[level]
        dep = runs[DEPLOYED]
        col = lambda k: np.array([float(runs[r][k]) for r in cfg.runs])                    # noqa: E731
        scores = col("score")
        bar = cal["Delta_bar"]
        deployed = {k: float(dep[k]) for k in ("Delta_U", "Delta_f", "Delta", "c_hat", "e_f", "offset", "limit",
                                               "limit_spread", "Delta_min", "a_err", "eq_res", "score")}
        deployed.update(a_hat=dep["a_hat"], fine=dict(zip(OF.FINE_KEYS, dep["fine"].tolist())),
                        fine_change=dict(zip(OF.FINE_KEYS, dep["fine_change"].tolist())),
                        vi_iterations=int(dep["vi_iterations"]), S_le_bar=bool(dep["score"] <= bar),
                        Delta_min_le_bar=bool(dep["Delta_min"] <= bar))
        offline[level] = dict(B=B, T=OF.T_STEPS, M=M, n_tr=B * OF.T_STEPS, N=B * OF.T_STEPS * M, n_runs=len(runs),
                              failed=int(np.sum(~np.isfinite(scores))), radius=C.residual_radius(bar, OF.C_SIGMA),
                              e_f_median=float(np.median(col("e_f"))), e_f_max=float(np.max(col("e_f"))),
                              Delta_median=float(np.median(col("Delta"))), Delta_U_median=float(np.median(col("Delta_U"))),
                              Delta_f_median=float(np.median(col("Delta_f"))), c_hat_min=float(np.min(col("c_hat"))),
                              offset_median=float(np.median(col("offset"))), limit_median=float(np.median(col("limit"))),
                              Delta_min_median=float(np.median(col("Delta_min"))),
                              a_err_median=float(np.median(col("a_err"))), limit_flags=int(np.sum(col("limit_flag"))),
                              seconds_per_run=float(np.mean(col("seconds"))),
                              seconds_total=float(np.sum(col("seconds"))), deployed=deployed, **cal)

    on = _load(_path(cfg, "online.npz"))
    _check_online(cfg, on)
    top = offline[int(on["level"])]
    bar_on = top["Delta_bar"] + float(on["Delta_on_used"])
    T = on["dec"].shape[1]
    tail = on["dist"][:, int(0.75 * T):T]
    online = dict(level=int(on["level"]), n_tr=top["n_tr"], delta_on=ON.DELTA_ON, alpha_on=ON.ALPHA_ON,
                  Delta_bar_off=top["Delta_bar"], Delta_on={str(d): float(v) for d, v in zip(on["deltas"], on["Delta_on"])},
                  Delta_on_used=float(on["Delta_on_used"]), radius=C.residual_radius(bar_on, OF.C_SIGMA),
                  Delta_min=float(np.max(on["dec"])), limit_median=float(np.median(np.max(tail, axis=1))),
                  n_runs=int(len(on["event"])), failed=int(np.sum(~on["event"])),
                  violations=int(np.sum(on["dec"][on["event"]] > bar_on)) if np.isfinite(bar_on) else None,
                  M_t=on["M_t"], M_extra=dict(zip(on["M_extra_t"].tolist(), on["M_extra"].tolist())),
                  probability=(1 - ALPHA) * (1 - ON.ALPHA_ON), steps=T)
    online["Delta_min_le_bound"] = bool(online["Delta_min"] <= bar_on)

    mf_runs = {r: _load(_modelfree_path(cfg, r)) for r in cfg.runs if os.path.exists(_modelfree_path(cfg, r))}
    missing = [r for r in cfg.runs if r not in mf_runs]
    if missing:
        raise SystemExit(f"model-free runs {missing} are missing: run --stage modelfree first")
    modelfree = {}
    for n in cfg.checkpoints:
        col = lambda k: np.array([float(mf_runs[r][f"n{n}_{k}"]) for r in cfg.runs])      # noqa: E731
        cal = calibrate([float(mf_runs[r][f"n{n}_Delta_dec"]) for r in _calib_runs(cfg)])
        dep = mf_runs[DEPLOYED]
        d_om = float(dep[f"n{n}_Delta_omega"])
        bound = cal["Delta_bar"] + d_om
        plug = dep[f"n{n}_plug_dec"]
        ev = dep[f"n{n}_plug_event"]
        deployed = {k: float(dep[f"n{n}_{k}"]) for k in ("Delta_dec", "Delta_dec_raw", "e_n", "viol_radius",
                                                          "Delta_min", "limit", "limit_spread", "offset")}
        deployed.update(Delta_omega=d_om, fine=dict(zip(MF.FINE_KEYS, dep[f"n{n}_fine"].tolist())),
                        fine_change=dict(zip(MF.FINE_KEYS, dep[f"n{n}_fine_change"].tolist())),
                        Delta_min_with_plugin=max(deployed["Delta_min"], float(np.max(plug))),
                        plugin_runs=int(len(ev)), plugin_events=int(np.sum(ev)),
                        plugin_violations=int(np.sum(plug[ev] > bound)))
        deployed["Delta_min_le_bound"] = bool(deployed["Delta_min_with_plugin"] <= bound)
        modelfree[n] = dict(N_n=n * MF.EPISODE_STEPS, n_runs=len(mf_runs), failed=int(np.sum(col("e_n") > 0)),
                            Delta_dec_median=float(np.median(col("Delta_dec"))),
                            Delta_dec_raw_median=float(np.median(col("Delta_dec_raw"))),
                            e_n_median=float(np.median(col("e_n"))), e_n_max=float(np.max(col("e_n"))),
                            viol_radius_max=float(np.max(col("viol_radius"))),
                            limit_median=float(np.median(col("limit"))),
                            radius=C.residual_radius(bound, MF.C_SIGMA), alpha_n=1 - (1 - ALPHA) * (1 - ON.ALPHA_ON),
                            deployed=deployed, **cal)
    mf_seconds = np.array([float(r["seconds"]) for r in mf_runs.values()])

    n_on = cfg.online_grid[0] * cfg.online_grid[1]
    test_sets = dict(T_omega=int(tv["n_test"]), T_omega_a=int(C.model_test_points()[0].size),
                     independent=cfg.fine_uniform + cfg.fine_near, perturbation_grid=n_on,
                     perturbations=len(ON.RADII) * ON.N_ANGLES)
    stage = lambda d: float(d["stage_seconds"]) if "stage_seconds" in d else None            # noqa: E731
    timings = dict(true=stage(tv), true_vi=true["seconds"],
                   offline_per_run={l: offline[l]["seconds_per_run"] for l in offline},
                   offline_total={l: offline[l]["seconds_total"] for l in offline}, online=stage(on),
                   modelfree_per_run=float(np.mean(mf_seconds)), modelfree_total=float(np.sum(mf_seconds)))
    settings_detail = {**asdict(cfg), "out": _portable(cfg.out), "figdir": _portable(cfg.figdir)}
    s = dict(settings=cfg.name, true=true, offline=offline, online=online, modelfree=modelfree, test_sets=test_sets,
             seeds=dict(master=MASTER_SEED, codes={"particle check": 2, "offline": 3, "online": 4, "model-free": 5},
                        calibration_runs=_calib_runs(cfg), deployed_run=DEPLOYED, test_points=7, model_test_points=123,
                        independent_points=11),
             timings=timings, versions=_versions(), settings_detail=settings_detail)
    # change of the maxima on the independent check set: the certificate quantities (acceptance) and Delta_min
    fine = [abs(offline[l]["deployed"]["fine_change"][k]) for l in offline for k in ("Delta_U", "Delta_f", "Delta",
                                                                                       "c_hat")]
    fine += [abs(modelfree[n]["deployed"]["fine_change"]["Delta_dec"]) for n in modelfree]
    fine_min = [abs(offline[l]["deployed"]["fine_change"]["Delta_min"]) for l in offline]
    fine_min += [abs(modelfree[n]["deployed"]["fine_change"]["Delta_min"]) for n in modelfree]
    s["fine_change_max"] = max(fine)
    s["fine_change_Delta_min_max"] = max(fine_min)
    rows = C.hypothesis_rows(s)
    s["hypotheses"] = [dict(result=r[0], hypothesis=r[1], status=r[2], evidence=r[3]) for r in rows]
    s["acceptance"] = dict(
        residual_rows=all([offline[l]["deployed"]["Delta_min_le_bar"] for l in offline] + [online["Delta_min_le_bound"]]
                          + [modelfree[n]["deployed"]["Delta_min_le_bound"] for n in modelfree]),
        eps_num_below_5e_3=eps < 5e-3, fine_change_max=max(fine), fine_change_below_5pct=max(fine) < 0.05,
        ratio_in_lemma_bounds=bool(true["ratio_min"] >= 1 - 1e-6 and true["ratio_max"] <= S.KAPPA2 + 1e-6))
    return s


def _versions():
    import matplotlib
    import scipy
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                matplotlib=matplotlib.__version__)


def stage_outputs(cfg):
    from . import plots
    t_stage = time.time()
    s = summarize(cfg)
    os.makedirs(cfg.figdir, exist_ok=True)
    plots.write_tables(s, cfg.figdir)
    plots.fig_certificates(figure_data(cfg, s), os.path.join(cfg.figdir, "toy_certificates"), use_tex=cfg.use_tex)
    s["timings"]["outputs"] = time.time() - t_stage                     # summary.json is written last
    with open(_path(cfg, "summary.json"), "w") as fh:
        json.dump(_jsonable(s), fh, indent=1)
    print(f"  summary {_path(cfg, 'summary.json')}; tables and figure in {cfg.figdir}")
    print("  acceptance: " + ", ".join(f"{k} {v}" for k, v in s["acceptance"].items()))
    return s


def figure_data(cfg, s):
    """Arrays of the four panels of the figure."""
    U, tv = _true_value(cfg)
    level = s["online"]["level"]
    dep = _load(_offline_path(cfg, level, DEPLOYED))
    on = _load(_path(cfg, "online.npz"))
    n = max(cfg.checkpoints)
    mf = _load(_modelfree_path(cfg, DEPLOYED))
    off, mfs = s["offline"][level], s["modelfree"][n]
    j = cfg.online_per_start                                     # first plug-in run from (-1, VS_HI)
    return dict(U=U, trajs=tv["trajs"], off_r=dep["r"], off_dec=dep["true_dec"], off_bar=off["Delta_bar"],
                off_radius=off["radius"], mf_r=mf["r"], mf_dec=mf["true_dec"],
                mf_bound=mfs["Delta_bar"] + mfs["deployed"]["Delta_omega"], mf_radius=mfs["radius"],
                D_opt=S.dist2(tv["trajs"][0][:, 0], tv["trajs"][0][:, 1]), D_off=dep["traj_D"],
                D_plug=on["dist"][j] ** 2, D_mf=mf[f"n{n}_traj_D"], n_label=n, n_tr=off["n_tr"])
