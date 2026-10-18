"""Smoke test of run_toy.py: the --quick stages in a temporary directory write every file and the summary; the
summary has no local absolute paths; a stage resumes; the online stage stops when its inputs changed."""

import json

import numpy as np
import pytest

import run_toy

KEYS = {"true", "offline", "online", "modelfree", "test_sets", "hypotheses", "acceptance", "seeds", "timings",
        "versions"}


def test_quick_stages(tmp_path):
    out, fig = tmp_path / "results", tmp_path / "figures"
    args = ["--out", str(out), "--figdir", str(fig)]
    run_toy.main(["--quick", *args])
    for f in ("true_value.npz", "online.npz", "summary.json", "offline/L1/run_00.npz", "offline/L1/run_49.npz",
              "modelfree/run_01.npz", "modelfree/run_49.npz"):
        assert (out / f).exists(), f
    for f in ("toy_certificates.pdf", "toy_certificates.png", "table_toy_hypotheses.tex", "table_toy_results.tex"):
        assert (fig / f).exists(), f
    text = (out / "summary.json").read_text()
    s = json.loads(text)
    assert KEYS <= set(s)
    assert len(s["hypotheses"]) == 12 and s["acceptance"]["residual_rows"]
    assert s["true"]["eps_num"] < 5e-3 and 1 - 1e-9 <= s["true"]["ratio_min"] <= s["true"]["ratio_max"] <= 2.56
    assert s["online"]["M_t"][:2] == [5950, 7916]
    assert all(s["timings"][k] > 0 for k in ("true", "true_vi", "online", "outputs", "modelfree_total"))
    assert set(s["offline"]["1"]["deployed"]["fine_change"]) == {"Delta_U", "Delta_f", "Delta", "c_hat", "Delta_min"}
    assert str(tmp_path) not in text and s["settings_detail"]["out"] == "results"      # no local absolute paths
    stamp = (out / "offline/L1/run_00.npz").stat().st_mtime_ns
    run_toy.main(["--quick", "--stage", "offline", *args])                                # resumes: skipped
    assert (out / "offline/L1/run_00.npz").stat().st_mtime_ns == stamp
    # the deployed offline policy changes after online.npz was written: the online stage and the summary stop
    path = out / "offline/L1/run_49.npz"
    with np.load(path) as d:
        run = {k: d[k] for k in d.files}
    run["theta"] = run["theta"] + 1e-3
    np.savez(path, **run)
    with pytest.raises(SystemExit, match="another deployed offline policy"):
        run_toy.main(["--quick", "--stage", "online", *args])
    with pytest.raises(SystemExit, match="another deployed offline policy"):
        run_toy.main(["--quick", "--stage", "outputs", *args])
