"""Train and evaluate every (environment, method, seed) run in parallel processes.

Each run trains one policy (``dmdp_sg.train``) and evaluates it at every noise level in
``EVAL_NOISES`` (``dmdp_sg.evaluate``). The files of a run are written to
``results/<tag>/<env>/<method>/seed<s>/``.

Finished work is skipped, so the script can be re-run safely after an interruption:
  * training is skipped if ``policy.pt`` exists;
  * evaluation is skipped for the noise levels already stored in ``evaluation.npz``. Adding a level to
    ``EVAL_NOISES`` evaluates only that level, and deleting ``evaluation.npz`` re-evaluates the saved
    policy at every level, in both cases without retraining.

Examples:
    python run_all.py --workers 14                                           # main study (Section 7)
    python run_all.py --methods DMDP --tag beta4 --beta 4 --workers 5        # sensitivity to beta
    python run_all.py --methods PPO DMDP --seeds 0 --steps 200000 --tag test # short test run
"""

from __future__ import annotations

import argparse
import os
import traceback
from multiprocessing import get_context

from dmdp_sg.config import ENVIRONMENTS, EVAL_EPISODES, EVAL_NOISES, METHODS, SEEDS, make_config

HERE = os.path.dirname(os.path.abspath(__file__))

Job = tuple[str, str, int, int, str, dict]


def run_one(job: Job) -> tuple[str, str, int, str]:
    """Trains (if needed) and evaluates (if needed) one run; returns "ok" or the traceback."""
    env, method, seed, steps, root, overrides = job
    from dmdp_sg.evaluate import evaluate_policy
    from dmdp_sg.train import train
    out = os.path.join(root, env, method, f"seed{seed}")
    policy = os.path.join(out, "policy.pt")
    evaluation = os.path.join(out, "evaluation.npz")
    try:
        if not os.path.exists(policy):
            cfg = make_config(env, method, seed, **({"total_steps": steps} if steps else {}), **overrides)
            train(cfg, out, verbose=True, log_every=50)
        evaluate_policy(policy, EVAL_NOISES, EVAL_EPISODES, evaluation)     # only the missing noise levels
        return env, method, seed, "ok"
    except Exception:
        return env, method, seed, traceback.format_exc()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train and evaluate the DMDP study; finished training runs and stored noise levels are "
                    "skipped (delete evaluation.npz to re-evaluate a saved policy without retraining).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--envs", nargs="+", default=["HalfCheetah"], choices=list(ENVIRONMENTS),
                   help="environments")
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS),
                   help="methods (see dmdp_sg/config.py)")
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS), help="random seeds")
    p.add_argument("--steps", type=int, default=0,
                   help="environment steps per run; 0 uses the value in ENVIRONMENTS (3M for HalfCheetah)")
    p.add_argument("--workers", type=int, default=15, help="parallel processes (one run per process)")
    p.add_argument("--tag", default="main", help="results are written to results/<tag>/")
    p.add_argument("--window", type=int, default=0,
                   help="window length n_w of the distribution estimator; 0 uses the default (10)")
    p.add_argument("--beta", type=float, default=-1.0,
                   help="dissimilarity weight beta for the methods with beta > 0; negative uses the default (1)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {}
    if args.window:
        overrides["window"] = args.window
    root = os.path.join(HERE, "results", args.tag)
    jobs = []
    for env in args.envs:
        for method in args.methods:
            for seed in args.seeds:
                ov = dict(overrides)
                if args.beta >= 0 and METHODS[method]["beta"] > 0:
                    ov["beta"] = args.beta
                jobs.append((env, method, seed, args.steps, root, ov))
    print(f"{len(jobs)} runs -> {root}", flush=True)
    with get_context("spawn").Pool(args.workers) as pool:
        for env, method, seed, status in pool.imap_unordered(run_one, jobs):
            print(f"DONE {env}/{method}/seed{seed}: {status if status == 'ok' else 'FAILED'}", flush=True)
            if status != "ok":
                print(status, flush=True)


if __name__ == "__main__":
    main()
