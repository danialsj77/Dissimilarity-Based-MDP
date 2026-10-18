"""Tests of run_all.py: reuse of finished runs, check of the stored configuration, evaluation of missing noise levels
only, validation of the command line and the commands documented in the module docstring.

The runs are tiny: one PPO update of 2 environments x 64 steps, evaluated on 2 episodes at 2 noise levels.
"""

import os
import shlex
import types

import numpy as np
import pytest

import dmdp_sg.evaluate as evaluate_module
import dmdp_sg.sac as sac_module
import dmdp_sg.train as train_module
import run_all
from dmdp_sg.config import make_config

NOISES = (0.0, 0.2)
FILES = {"config.json", "episodes.csv", "evaluation.npz", "policy.pt", "training.csv"}


@pytest.fixture
def tiny_evaluation(monkeypatch):
    monkeypatch.setattr(run_all, "EVAL_NOISES", NOISES)
    monkeypatch.setattr(run_all, "EVAL_EPISODES", 2)


def _job(root, steps=128, **overrides):
    return ("HalfCheetah", "PPO", 0, steps, str(root), {"learner": "ppo", "num_envs": 2, "num_steps": 64, **overrides})


def _run_dir(root):
    return root / "HalfCheetah" / "PPO" / "seed0"


def _snapshot(root):
    """Modification time of every file under ``root``."""
    return {str(p.relative_to(root)): p.stat().st_mtime_ns for p in sorted(root.rglob("*")) if p.is_file()}


def _forbid(monkeypatch, *targets):
    for module, name in targets:
        def refuse(*args, _name=name, **kwargs):
            raise AssertionError(f"{_name} must not be called")
        monkeypatch.setattr(module, name, refuse)


def _spy_on_evaluation(monkeypatch):
    """Records the noise level of every evaluation rollout."""
    levels, real = [], evaluate_module.rollout

    def spy(agent, rms, cfg, nu, episodes, **kwargs):
        levels.append(nu)
        return real(agent, rms, cfg, nu, episodes, **kwargs)
    monkeypatch.setattr(evaluate_module, "rollout", spy)
    return levels


def _load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture
def finished(tmp_path, tiny_evaluation):
    """Results root with one trained and evaluated run."""
    assert run_all.run_one(_job(tmp_path)) == ("HalfCheetah", "PPO", 0, "ok")
    assert {p.name for p in _run_dir(tmp_path).iterdir()} == FILES
    return tmp_path


def test_finished_run_is_skipped(finished, monkeypatch):
    before = _snapshot(finished)
    _forbid(monkeypatch, (train_module, "train"), (sac_module, "train_sac"), (evaluate_module, "rollout"))
    assert run_all.run_one(_job(finished)) == ("HalfCheetah", "PPO", 0, "ok")
    assert _snapshot(finished) == before                    # no file was written again


@pytest.mark.parametrize("changed", [dict(cost_limit=2.5), dict(steps=256)])
def test_changed_configuration_is_reported_and_nothing_is_overwritten(finished, monkeypatch, changed):
    before = _snapshot(finished)
    _forbid(monkeypatch, (train_module, "train"), (sac_module, "train_sac"), (evaluate_module, "rollout"))
    status = run_all.run_one(_job(finished, **changed))[3]
    assert status != "ok" and "was trained with a different configuration" in status
    assert _snapshot(finished) == before


def test_missing_noise_level_is_evaluated_alone(finished, monkeypatch):
    path = _run_dir(finished) / "evaluation.npz"
    stored = _load(path)
    assert {key.split("/")[0] for key in stored} == {"nu0.00", "nu0.20"}
    np.savez_compressed(path, **{k: v for k, v in stored.items() if not k.startswith("nu0.20/")})
    others = {k: v for k, v in _snapshot(finished).items() if not k.endswith("evaluation.npz")}
    levels = _spy_on_evaluation(monkeypatch)
    _forbid(monkeypatch, (train_module, "train"), (sac_module, "train_sac"))

    assert run_all.run_one(_job(finished))[3] == "ok"
    assert levels == [0.2]
    restored = _load(path)
    assert set(restored) == set(stored)
    for key, array in stored.items():
        np.testing.assert_array_equal(restored[key], array)      # the kept level and the recomputed level
    assert {k: v for k, v in _snapshot(finished).items() if not k.endswith("evaluation.npz")} == others


def test_deleted_evaluation_is_recomputed_without_training(finished, monkeypatch):
    path = _run_dir(finished) / "evaluation.npz"
    stored = _load(path)
    os.remove(path)
    levels = _spy_on_evaluation(monkeypatch)
    _forbid(monkeypatch, (train_module, "train"), (sac_module, "train_sac"))

    assert run_all.run_one(_job(finished))[3] == "ok"
    assert levels == list(NOISES)
    restored = _load(path)
    assert set(restored) == set(stored)
    for key, array in stored.items():
        np.testing.assert_array_equal(restored[key], array)


def test_directory_without_policy_is_trained_again(finished, monkeypatch):
    # what an interrupted save leaves behind: config.json is written first and policy.pt last
    run = _run_dir(finished)
    os.remove(run / "policy.pt")
    trained, real = [], train_module.train
    monkeypatch.setattr(train_module, "train", lambda *args, **kwargs: (trained.append(1), real(*args, **kwargs))[1])
    assert run_all.run_one(_job(finished))[3] == "ok"
    assert trained == [1] and {p.name for p in run.iterdir()} == FILES


@pytest.mark.parametrize("argv, message", [
    (["--learner", "sac", "--methods", "PPO"], "PPO cannot be trained with the sac learner"),
    (["--learner", "sac", "--methods", "DMDP", "PPO-Lag"], "PPO-Lag cannot be trained with the sac learner"),
    (["--learner", "ppo", "--methods", "SAC"], "pass --learner sac"),
    (["--methods", "Dist-SAC"], "pass --learner sac"),
    (["--learner", "sac", "--methods", "Dist-PPO"], "pass --learner ppo"),
    (["--methods", "Implicit-SAC"], "pass --learner sac"),
    (["--learner", "sac", "--methods", "Implicit-PPO"], "pass --learner ppo"),
    (["--methods", "Implicit-SAC-Lag"], "pass --learner sac"),
    (["--learner", "sac", "--methods", "Dist-PPO-Lag"], "pass --learner ppo"),
    (["--learner", "sac", "--init-log-std", "0"], "PPO learner only"),
    (["--methods", "PPO", "--beta", "2"], "--beta applies only"),
    (["--lagrangian-lr", "-0.1"], "must not be negative"),
    (["--seeds", "0", "1", "0"], "requested twice"),
    (["--methods", "PPO", "PPO"], "requested twice"),
])
def test_invalid_request_is_rejected_before_the_pool_starts(monkeypatch, argv, message):
    def refuse(*args):
        raise AssertionError("the worker pool must not start")
    monkeypatch.setattr(run_all, "get_context", refuse)
    with pytest.raises(SystemExit) as error:
        run_all.main(argv)
    assert message in str(error.value)


def _fake_pool(status):
    """Replaces the worker pool: every run finishes at once with ``status``."""
    class Pool:
        def __init__(self, workers):
            self.workers = workers

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def imap_unordered(self, function, jobs):
            return ((job[0], job[1], job[2], status) for job in jobs)
    return lambda method: types.SimpleNamespace(Pool=Pool)


def test_valid_request_runs_every_job(monkeypatch, capsys):
    monkeypatch.setattr(run_all, "get_context", _fake_pool("ok"))
    run_all.main(["--learner", "sac", "--methods", "SAC-Lag", "DMDP-Lag", "--seeds", "10", "11", "--tag", "unused"])
    out = capsys.readouterr().out
    assert "4 runs" in out and out.count("DONE") == 4 and "FAILED" not in out


def test_failed_runs_are_reported_and_fail_the_command(monkeypatch, capsys):
    monkeypatch.setattr(run_all, "get_context", _fake_pool("Traceback ..."))
    with pytest.raises(SystemExit) as error:
        run_all.main(["--methods", "PPO", "--seeds", "0", "1", "--tag", "unused"])
    assert "2 of 2 runs failed" in str(error.value) and capsys.readouterr().out.count("FAILED") == 2


@pytest.mark.parametrize("learner, methods", [("sac", ["SAC-Lag", "DMDP-Lag"]), ("ppo", ["PPO-Lag", "DMDP-Lag"])])
def test_lagrangian_lr_overrides_the_rate_of_the_multiplier(learner, methods):
    argv = ["--learner", learner, "--methods", *methods, "--seeds", "10", "11"]
    default = make_config("HalfCheetah", methods[0], 0, learner=learner).lagrangian_lr
    assert default != 0.005
    jobs = run_all.build_jobs(run_all.parse_args(argv + ["--lagrangian-lr", "0.005"]))
    assert len(jobs) == 4 and {run_all.job_config(job).lagrangian_lr for job in jobs} == {0.005}
    jobs = run_all.build_jobs(run_all.parse_args(argv))
    assert {run_all.job_config(job).lagrangian_lr for job in jobs} == {default}


# folder under results/ -> learner, methods, seeds and the fields of the configuration that differ from the defaults of
# make_config(env, method, seed, learner=learner); read from the config.json files of the stored runs
EIGHT_PPO = ("PPO", "PPO-Lag", "Dist-PPO", "Dist-PPO-Lag", "Implicit-PPO", "Implicit-PPO-Lag", "DMDP", "DMDP-Lag")
EIGHT_SAC = ("SAC", "SAC-Lag", "Dist-SAC", "Dist-SAC-Lag", "Implicit-SAC", "Implicit-SAC-Lag", "DMDP", "DMDP-Lag")
PILOT = {"init_log_std": 0.0, "cost_penalty": 2.0}           # design pilot: initial action std 1 and the cost penalty (2 c_t)^2
RESULTS_FOLDERS = {
    "main": ("ppo", EIGHT_PPO, range(5), {}),
    "pilot/beta0.5_c10": ("ppo", ("DMDP-Lag",), (10, 11, 12), {**PILOT}),
    "pilot/beta1_c10": ("ppo", ("DMDP-Lag",), (10, 11, 12), {**PILOT, "beta": 1.0}),
    "pilot/beta2_c10": ("ppo", ("DMDP-Lag",), (10, 11, 12), {**PILOT, "beta": 2.0}),
    "pilot/beta1_c5": ("ppo", ("DMDP-Lag",), (10, 11, 12), {**PILOT, "beta": 1.0, "cost_limit": 5.0}),
    "pilot/beta1_c2.5": ("ppo", ("DMDP-Lag",), (10, 11, 12), {**PILOT, "beta": 1.0, "cost_limit": 2.5}),
    "pilot/beta0.5_c5": ("ppo", ("DMDP-Lag",), (10, 11, 12), {**PILOT, "cost_limit": 5.0}),
    "pilot/beta0.5_c2.5": ("ppo", ("DMDP-Lag",), (10, 11, 12), {**PILOT, "cost_limit": 2.5}),
    "pilot/baselines_c10": ("ppo", ("PPO", "PPO-Lag"), (10, 11, 12), {**PILOT}),
    "pilot/baselines_c10_benchmark_reward": ("ppo", ("PPO", "PPO-Lag"), (10, 11, 12), {"init_log_std": 0.0}),
    "sac": ("sac", EIGHT_SAC, range(5), {}),
    "test": ("ppo", ("PPO", "DMDP"), (0,), {"total_steps": 200_000}),      # the short test run of the docstring
}


def test_documented_commands_reproduce_the_results_folders():
    commands = [shlex.split(line, comments=True)[2:] for line in run_all.__doc__.splitlines()
                if line.lstrip().startswith("python run_all.py")]
    documented = {}
    for argv in commands:
        args = run_all.parse_args(argv)
        assert args.tag not in documented, f"{args.tag} is documented twice"
        jobs = run_all.build_jobs(args)
        assert all(job[4] == os.path.join(run_all.HERE, "results", args.tag) for job in jobs)
        differences = []
        for job in jobs:
            base = make_config(job[0], job[1], job[2], learner=args.learner).to_dict()
            differences.append({k: v for k, v in run_all.job_config(job).to_dict().items() if v != base[k]})
        assert all(d == differences[0] for d in differences), args.tag
        documented[args.tag] = (args.learner, tuple(dict.fromkeys(job[1] for job in jobs)),
                                tuple(dict.fromkeys(job[2] for job in jobs)), differences[0])
    expected = {tag: (learner, methods, tuple(seeds), diff) for tag, (learner, methods, seeds, diff) in RESULTS_FOLDERS.items()}
    assert documented == expected
