"""Computes the certificates of the scalar example of Appendix D, stage by stage (one process, 2 threads).

Every stage skips the run files that already exist, so an interrupted command can be repeated. Run one command at a
time; each takes 6 minutes or less on the machine used for the paper, with a peak memory below 0.6 GB:

    nice -n 10 python run_toy.py --stage true                                 # U~ ~ U_star^R and its error, 0.5 min
    nice -n 10 python run_toy.py --stage offline --levels 1 2 3               # Theorem 3, 150 runs, 5 min
    nice -n 10 python run_toy.py --stage offline --levels 4 --runs 0-24       # Theorem 3, 3.5 min
    nice -n 10 python run_toy.py --stage offline --levels 4 --runs 25-49      # Theorem 3, 3.5 min
    nice -n 10 python run_toy.py --stage online                               # Theorem 4, 0.5 min
    nice -n 10 python run_toy.py --stage modelfree --runs 0-24                # Theorem 5, 3.5 min
    nice -n 10 python run_toy.py --stage modelfree --runs 25-49               # Theorem 5, 3.5 min
    nice -n 10 python run_toy.py --stage outputs                              # summary, tables, figure, 0.5 min

Data go to results_toy/ and the figure and tables to ../Manuscript/figures/ (or figures/ if that folder does not
exist). ``--quick`` runs all stages of a reduced setting in about 10 seconds; its data, figure and tables go to
results_toy_quick/.
"""

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "2"

import argparse                                     # noqa: E402
import time                                         # noqa: E402

from dmdp_toy import stages                         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = ("true", "offline", "online", "modelfree", "outputs")


def run_range(text):
    a, _, b = text.partition("-")
    return tuple(range(int(a), int(b or a) + 1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", choices=STAGES, help="stage to run (required without --quick)")
    p.add_argument("--levels", type=int, nargs="+", choices=(1, 2, 3, 4), help="offline data levels (default: all)")
    p.add_argument("--runs", type=run_range, help="runs a-b (default: all)")
    p.add_argument("--quick", action="store_true", help="reduced setting; without --stage it runs every stage")
    p.add_argument("--out", help="data folder (default: results_toy, or results_toy_quick with --quick)")
    p.add_argument("--figdir", help="folder of the figure and tables (default: ../Manuscript/figures or figures/; "
                                    "with --quick: <out>/figures)")
    p.add_argument("--no-tex", action="store_true", help="matplotlib mathtext instead of LaTeX")
    args = p.parse_args(argv)
    if args.stage is None and not args.quick:
        p.error("--stage is required (there is no stage that runs everything)")
    manuscript = os.path.normpath(os.path.join(HERE, "..", "Manuscript", "figures"))
    if args.quick:                                  # the quick setting never writes into the manuscript
        out = args.out or os.path.join(HERE, "results_toy_quick")
        cfg = stages.quick_settings(out=out, figdir=args.figdir or os.path.join(out, "figures"))
    else:
        figdir = args.figdir or (manuscript if os.path.isdir(manuscript) else os.path.join(HERE, "figures"))
        cfg = stages.Settings(out=args.out or os.path.join(HERE, "results_toy"), figdir=figdir, use_tex=not args.no_tex)
    runs = tuple(r for r in args.runs if r in cfg.runs) if args.runs else None
    calls = {"true": lambda: stages.stage_true(cfg),
             "offline": lambda: stages.stage_offline(cfg, args.levels, runs),
             "online": lambda: stages.stage_online(cfg),
             "modelfree": lambda: stages.stage_modelfree(cfg, runs),
             "outputs": lambda: stages.stage_outputs(cfg)}
    total = time.time()
    for name in [args.stage] if args.stage else STAGES:
        t0 = time.time()
        print(f"stage {name} ({cfg.name})")
        calls[name]()
        print(f"stage {name}: {time.time() - t0:.1f} s")
    if not args.stage:
        print(f"all stages: {time.time() - total:.1f} s")


if __name__ == "__main__":
    main()
