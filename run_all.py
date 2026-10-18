"""Train and evaluate every (environment, method, seed) run in parallel processes.

Each run trains one policy (``dmdp_sg.train``, or ``dmdp_sg.sac`` for the SAC learner) and evaluates it at every noise
level in ``EVAL_NOISES`` (``dmdp_sg.evaluate``). The files of a run are written to
``results/<tag>/<env>/<method>/seed<s>/``.

Finished work is skipped, so the script can be re-run safely after an interruption:
  * training is skipped if ``policy.pt`` exists and its ``config.json`` equals the requested
    configuration; a run whose stored configuration differs (for example after a change of
    ``METHODS``) is reported as failed and must be moved away or written under a new tag;
    a directory without ``policy.pt`` (an interrupted run) is trained again;
  * evaluation is skipped for the noise levels already stored in ``evaluation.npz``. Adding a level to
    ``EVAL_NOISES`` evaluates only that level, and deleting ``evaluation.npz`` re-evaluates the saved
    policy at every level, in both cases without retraining.

The configuration of every requested run is built before the first run starts, so that an invalid request fails at
once with a message and not inside every worker.

Commands that reproduce the results folders (each run uses one core; ``--workers`` sets the number of runs in parallel
and does not change the results):

    # main study: 8 methods (PPO, PPO-Lag, Dist-PPO, Dist-PPO-Lag, Implicit-PPO, Implicit-PPO-Lag, DMDP, DMDP-Lag), seeds 0-4,
    # 3M steps (Appendix F), results/main/
    python run_all.py --workers 18

    # design pilot of DMDP-Lag on seeds 10-12 (initial action std 1), trained with the cost penalty (2 c_t)^2 in the reward,
    # results/pilot/<setting>/, one command per setting
    python run_all.py --methods DMDP-Lag --seeds 10 11 12 --tag pilot/beta0.5_c10 --beta 0.5 --cost-limit 10 --init-log-std 0 --cost-penalty 2 --workers 3
    python run_all.py --methods DMDP-Lag --seeds 10 11 12 --tag pilot/beta1_c10 --beta 1 --cost-limit 10 --init-log-std 0 --cost-penalty 2 --workers 3
    python run_all.py --methods DMDP-Lag --seeds 10 11 12 --tag pilot/beta2_c10 --beta 2 --cost-limit 10 --init-log-std 0 --cost-penalty 2 --workers 3
    python run_all.py --methods DMDP-Lag --seeds 10 11 12 --tag pilot/beta1_c5 --beta 1 --cost-limit 5 --init-log-std 0 --cost-penalty 2 --workers 3
    python run_all.py --methods DMDP-Lag --seeds 10 11 12 --tag pilot/beta1_c2.5 --beta 1 --cost-limit 2.5 --init-log-std 0 --cost-penalty 2 --workers 3
    python run_all.py --methods DMDP-Lag --seeds 10 11 12 --tag pilot/beta0.5_c5 --beta 0.5 --cost-limit 5 --init-log-std 0 --cost-penalty 2 --workers 3
    python run_all.py --methods DMDP-Lag --seeds 10 11 12 --tag pilot/beta0.5_c2.5 --beta 0.5 --cost-limit 2.5 --init-log-std 0 --cost-penalty 2 --workers 3
    # reference of the rule for c_bar in the design pilot: PPO and PPO-Lag (c_bar = 10) on the pilot seeds, trained, as in the pilot
    # that chose c_bar = 10, with the cost penalty of the pilot runs of DMDP-Lag, results/pilot/baselines_c10/
    python run_all.py --methods PPO PPO-Lag --seeds 10 11 12 --tag pilot/baselines_c10 --cost-limit 10 --init-log-std 0 --cost-penalty 2 --workers 6
    # the same runs with the benchmark reward of the reported baselines (with them the rule would give c_bar = 5), results/pilot/baselines_c10_benchmark_reward/
    python run_all.py --methods PPO PPO-Lag --seeds 10 11 12 --tag pilot/baselines_c10_benchmark_reward --cost-limit 10 --init-log-std 0 --workers 6

    # SAC study: 8 methods, seeds 0-4, 1M steps (Section 5.1), results/sac/
    python run_all.py --learner sac --methods SAC SAC-Lag Dist-SAC Dist-SAC-Lag Implicit-SAC Implicit-SAC-Lag DMDP DMDP-Lag --tag sac --workers 20

    # short test run
    python run_all.py --methods PPO DMDP --seeds 0 --steps 200000 --tag test
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from multiprocessing import get_context

from dmdp_sg.config import (ENVIRONMENTS, EVAL_EPISODES, EVAL_NOISES, LEARNERS, METHODS, SEEDS, TrainConfig,
                            config_from_dict, make_config)

HERE = os.path.dirname(os.path.abspath(__file__))

Job = tuple[str, str, int, int, str, dict]


def job_config(job: Job) -> TrainConfig:
    """Configuration of a run: the defaults of its method, with the number of steps (if given) and the overrides."""
    env, method, seed, steps, _, overrides = job
    return make_config(env, method, seed, **({"total_steps": steps} if steps else {}), **overrides)


def run_one(job: Job) -> tuple[str, str, int, str]:
    """Trains (if needed) and evaluates (if needed) one run; returns "ok" or the traceback."""
    env, method, seed, _, root, _ = job
    from dmdp_sg.evaluate import evaluate_policy
    from dmdp_sg.sac import train_sac
    from dmdp_sg.train import train
    out = os.path.join(root, env, method, f"seed{seed}")
    policy = os.path.join(out, "policy.pt")
    evaluation = os.path.join(out, "evaluation.npz")
    try:
        cfg = job_config(job)
        if os.path.exists(policy):
            with open(os.path.join(out, "config.json")) as f:
                if config_from_dict(json.load(f)) != cfg:
                    raise ValueError(f"{out} was trained with a different configuration")
        else:
            if os.path.exists(evaluation):
                os.remove(evaluation)       # evaluation of an earlier policy that no longer exists
            (train_sac if cfg.learner == "sac" else train)(cfg, out, verbose=True, log_every=50)
        evaluate_policy(policy, EVAL_NOISES, EVAL_EPISODES, evaluation)     # only the missing noise levels
        return env, method, seed, "ok"
    except Exception:
        return env, method, seed, traceback.format_exc()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train and evaluate the DMDP study; finished training runs and stored noise levels are "
                    "skipped (delete evaluation.npz to re-evaluate a saved policy without retraining).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--envs", nargs="+", default=["HalfCheetah"], choices=list(ENVIRONMENTS),
                   help="environments")
    p.add_argument("--learner", default="ppo", choices=list(LEARNERS),
                   help="learner of DMDP and DMDP-Lag (PPO, PPO-Lag and the Dist-PPO and Implicit-PPO methods always use PPO; "
                        "SAC, SAC-Lag and the Dist-SAC and Implicit-SAC methods always use SAC)")
    p.add_argument("--methods", nargs="+", default=None, choices=list(METHODS),
                   help="methods (see dmdp_sg/config.py); default: the eight methods of the learner")
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS), help="random seeds")
    p.add_argument("--steps", type=int, default=0,
                   help="environment steps per run; 0 uses the default (3M for PPO on HalfCheetah, SAC_TOTAL_STEPS = 1M for SAC)")
    p.add_argument("--workers", type=int, default=15, help="parallel processes (one run per process)")
    p.add_argument("--tag", default="main", help="results are written to results/<tag>/")
    p.add_argument("--window", type=int, default=0,
                   help="window length n_w of the distribution estimator; 0 uses the default (10)")
    p.add_argument("--cost-limit", type=float, default=-1.0,
                   help="cost limit c_bar of the constrained methods; negative uses the default (COST_LIMIT in dmdp_sg/config.py)")
    p.add_argument("--velocity-weight", type=float, default=None,
                   help="weight of the velocity term of D (0 with --beta gives a run with the pitch term only, "
                        "for which --beta also applies to PPO and PPO-Lag; not used in the paper)")
    p.add_argument("--init-log-std", type=float, default=None,
                   help="initial log standard deviation of the actions; default: INIT_LOG_STD in dmdp_sg/config.py (0 is the PPO default, std 1)")
    p.add_argument("--d-threshold", type=float, default=None,
                   help="threshold D_thr of the threshold reward r_tD; default: D_THRESHOLD in dmdp_sg/config.py")
    p.add_argument("--beta", type=float, default=-1.0,
                   help="dissimilarity weight beta for the methods with beta > 0; negative uses the default (BETA in dmdp_sg/config.py)")
    p.add_argument("--lagrangian-lr", type=float, default=None,
                   help="Adam learning rate of the Lagrange multiplier of the constrained methods, for both learners; "
                        "default: lagrangian_lr in dmdp_sg/config.py")
    p.add_argument("--cost-penalty", type=float, default=None,
                   help="kappa of the cost penalty (kappa c_t)^2 for every selected method; default: cost_penalty of each "
                        "method in METHODS (0 for every method; 2 in the design pilot)")
    return p.parse_args(argv)


def results_root(tag: str) -> str:
    return os.path.join(HERE, "results", tag)


def build_jobs(args: argparse.Namespace) -> list[Job]:
    """The runs requested by ``args``; raises ``SystemExit`` with a message if one of them is invalid."""
    methods = args.methods
    if methods is None:
        base = "PPO" if args.learner == "ppo" else "SAC"
        methods = [base, f"{base}-Lag", f"Dist-{base}", f"Dist-{base}-Lag", f"Implicit-{base}", f"Implicit-{base}-Lag", "DMDP", "DMDP-Lag"]
    for method in methods:
        if METHODS[method].get("learner", args.learner) != args.learner:
            raise SystemExit(f"{method} uses the {METHODS[method]['learner']} learner; pass --learner {METHODS[method]['learner']}")
    overrides = {"learner": args.learner}
    if args.d_threshold is not None:
        overrides["d_threshold"] = args.d_threshold
    if args.learner == "sac" and args.init_log_std is not None:
        raise SystemExit("--init-log-std applies to the PPO learner only")
    if args.window:
        overrides["window"] = args.window
    if args.cost_limit >= 0:
        overrides["cost_limit"] = args.cost_limit
    if args.init_log_std is not None:
        overrides["init_log_std"] = args.init_log_std
    if args.lagrangian_lr is not None:
        if args.lagrangian_lr < 0:
            raise SystemExit("--lagrangian-lr must not be negative")
        overrides["lagrangian_lr"] = args.lagrangian_lr
    if args.cost_penalty is not None:
        if args.cost_penalty < 0:
            raise SystemExit("--cost-penalty must not be negative")
        overrides["cost_penalty"] = args.cost_penalty
    control = args.velocity_weight is not None                 # control runs: --beta applies to every selected method
    if control:
        overrides["velocity_weight"] = args.velocity_weight
    if args.beta >= 0 and not control and not any(METHODS[m]["beta"] > 0 for m in methods):
        raise SystemExit("--beta applies only to the methods with beta > 0 (DMDP, DMDP-Lag); none of them is selected")
    root = results_root(args.tag)
    jobs = []
    for env in args.envs:
        for method in methods:
            for seed in args.seeds:
                ov = dict(overrides)
                if args.beta >= 0 and (METHODS[method]["beta"] > 0 or control):
                    ov["beta"] = args.beta
                jobs.append((env, method, seed, args.steps, root, ov))
    if len({job[:3] for job in jobs}) != len(jobs):     # two workers would write the same directory
        raise SystemExit("a run is requested twice; remove the repeated environment, method or seed")
    for job in jobs:                    # the configuration every worker will build, checked before any run starts
        try:
            job_config(job)
        except ValueError as error:
            raise SystemExit(f"invalid run {job[0]}/{job[1]}/seed{job[2]}: {error}")
    return jobs


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    jobs = build_jobs(args)
    print(f"{len(jobs)} runs -> {results_root(args.tag)}", flush=True)
    failed = 0
    with get_context("spawn").Pool(args.workers) as pool:
        for env, method, seed, status in pool.imap_unordered(run_one, jobs):
            print(f"DONE {env}/{method}/seed{seed}: {status if status == 'ok' else 'FAILED'}", flush=True)
            if status != "ok":
                failed += 1
                print(status, flush=True)
    if failed:
        raise SystemExit(f"{failed} of {len(jobs)} runs failed")


if __name__ == "__main__":
    main()
