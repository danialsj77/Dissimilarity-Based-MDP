"""The stored results in results/ load with the current code (backward compatibility)."""

import os

import numpy as np
import pytest

from dmdp_sg.analysis import load_config, load_evaluation, load_training, run_dir
from dmdp_sg.config import EVAL_EPISODES, EVAL_NOISES, METHODS, TrainConfig
from dmdp_sg.evaluate import load_policy

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "main")
pytestmark = pytest.mark.skipif(not os.path.isdir(ROOT), reason="stored results not available")


@pytest.mark.parametrize("method", list(METHODS))
def test_stored_run_loads(method):
    cfg = load_config(ROOT, "HalfCheetah", method, 0)
    assert TrainConfig(**cfg).method == method
    training = load_training(ROOT, "HalfCheetah", method, 0)
    assert list(training.columns) == ["step", "ep_return", "ep_cost", "ep_dissimilarity", "ep_length",
                                      "lagrange_multiplier", "seconds"]
    ev = load_evaluation(ROOT, "HalfCheetah", method, 0)
    if ev is None:
        pytest.skip("evaluation.npz is not stored in git; regenerate it with run_all.py (see README)")
    assert sorted(ev) == sorted(EVAL_NOISES)
    assert ev[0.0]["velocity"].shape == (EVAL_EPISODES, 1000)


def test_stored_policy_loads():
    agent, rms, cfg = load_policy(os.path.join(run_dir(ROOT, "HalfCheetah", "DMDP", 0), "policy.pt"))
    assert cfg.policy_input == "hybrid" and rms.mean.shape == (19,)
    assert np.all(np.isfinite(rms.var))
