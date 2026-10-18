"""The stored results in results/ load with the current code (backward compatibility)."""

import os

import numpy as np
import pytest

from dmdp_sg.analysis import load_config, load_episodes, load_evaluation, load_training, run_dir
from dmdp_sg.config import EVAL_EPISODES, EVAL_NOISES, METHODS, config_from_dict
from dmdp_sg.evaluate import load_policy

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
ROOT = os.path.join(RESULTS, "main")                       # PPO study (Section 7.1)
SAC_ROOT = os.path.join(RESULTS, "sac")                    # SAC study (Section 7.2)
STUDIES = [(ROOT, m, "ppo") for m in ("PPO", "PPO-Lag", "Dist-PPO", "Dist-PPO-Lag", "Implicit-PPO", "Implicit-PPO-Lag", "DMDP", "DMDP-Lag")] + \
          [(SAC_ROOT, m, "sac") for m in ("SAC", "SAC-Lag", "Dist-SAC", "Dist-SAC-Lag", "Implicit-SAC", "Implicit-SAC-Lag", "DMDP", "DMDP-Lag")]


@pytest.mark.parametrize("root,method,learner", STUDIES)
def test_stored_run_loads(root, method, learner):
    if not os.path.exists(os.path.join(run_dir(root, "HalfCheetah", method, 0), "policy.pt")):
        pytest.skip("run not finished")
    cfg = load_config(root, "HalfCheetah", method, 0)
    stored = config_from_dict(cfg)                         # reads entries of earlier versions, like the code that loads runs
    assert stored.method == method and stored.learner == learner and stored.n_quantiles == METHODS[method].get("n_quantiles", 0)
    # every method is trained on the benchmark reward (the DMDP methods add r_tD and -beta * D)
    assert cfg["cost_penalty"] == METHODS[method]["cost_penalty"] == 0.0
    training = load_training(root, "HalfCheetah", method, 0)
    assert list(training.columns) == ["step", "ep_return", "ep_task_return", "ep_cost", "ep_dissimilarity",
                                      "ep_dissimilarity_median", "ep_within", "ep_pitch", "ep_flipped", "ep_length",
                                      "lagrange_multiplier", "seconds"] + (["alpha"] if learner == "sac" else [])
    episodes = load_episodes(root, "HalfCheetah", method, 0)
    assert list(episodes.columns) == ["step", "ret", "task_ret", "cost", "dis", "dis_median", "within", "pitch",
                                      "flipped"] and len(episodes) >= (2900 if learner == "ppo" else 990)
    if METHODS[method]["d_bonus"] == 0:
        assert np.allclose(episodes.ret, episodes.task_ret)             # the baselines receive no threshold reward
    ev = load_evaluation(root, "HalfCheetah", method, 0)
    if ev is None:
        pytest.skip("evaluation.npz is not stored in git; regenerate it with run_all.py (see README)")
    assert sorted(ev) == sorted(EVAL_NOISES)
    assert ev[0.0]["velocity"].shape == (EVAL_EPISODES, 1000)


def test_stored_policy_loads():
    if not os.path.exists(os.path.join(run_dir(ROOT, "HalfCheetah", "DMDP", 0), "policy.pt")):
        pytest.skip("run not finished")
    agent, rms, cfg = load_policy(os.path.join(run_dir(ROOT, "HalfCheetah", "DMDP", 0), "policy.pt"))
    assert cfg.policy_input == "hybrid" and rms.mean.shape == (21,)
    assert np.all(np.isfinite(rms.var))
