"""Caps the BLAS and OpenMP threads at 2 and makes the repository root importable, so that ``pytest`` works from any
directory. pytest loads this file before the test modules, so the caps hold before numpy is first imported (variables
the user has exported keep their values)."""

import os
import sys

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "2")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
