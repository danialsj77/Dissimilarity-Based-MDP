"""DMDP_PPO.ipynb and DMDP_SAC.ipynb show the code of the package without drift.

The notebooks are built from the sources of ``dmdp_sg`` by ``build_code_notebooks.py``. The tests check that the stored
notebooks equal the current build, that every definition in a code cell is the definition of the package, and that the code of a
notebook, run in one namespace, trains and evaluates a policy exactly as the package does.
"""

import __future__
import ast
import os
import sys
import types

import nbformat
import numpy as np
import pytest
import torch

from build_code_notebooks import build, bound_names
from code_notebooks.slicing import Module, PKG

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ("DMDP_PPO.ipynb", "DMDP_SAC.ipynb")


@pytest.fixture(scope="module")
def built():
    return build(write=False)


def _package_statements() -> set[str]:
    """Dumps of the top-level statements (without imports and docstrings) of every module of the package."""
    dumps = set()
    for name in sorted(f[:-3] for f in os.listdir(PKG) if f.endswith(".py") and f != "__init__.py"):
        for it in Module(name).items:
            if it.kind not in ("import", "docstring"):
                dumps.add(ast.dump(it.node))
    return dumps


@pytest.mark.parametrize("file", FILES)
def test_the_stored_notebook_is_the_current_build(built, file):
    stored = nbformat.read(os.path.join(REPO_ROOT, file), as_version=4)
    assert [(c.cell_type, c.source) for c in stored.cells] == [(c.cell_type, c.source) for c in built[file].cells]


@pytest.mark.parametrize("file", FILES)
def test_every_definition_in_a_package_cell_is_a_definition_of_the_package(built, file):
    nb, known, foreign = built[file], _package_statements(), []
    for i in sorted(nb.package):
        for node in ast.parse(nb.cells[i].source).body:
            if isinstance(node, (ast.Import, ast.ImportFrom)) or ast.dump(node) in known:
                continue
            foreign.append((i, ast.unparse(node).split("\n")[0]))
    # the cell of the setup, the alias of the import in config.py, and load_policy of one learner are not in the package as such
    allowed = {"with contextlib.redirect_stderr(io.StringIO()):", "_target_mean = target_mean", "def load_policy(path: str) -> tuple[Agent, RunningMeanStd, TrainConfig]:",
               "def load_policy(path: str) -> tuple[SACAgent, IdentityNormalizer, TrainConfig]:"}
    assert [line for _, line in foreign if line not in allowed] == []


@pytest.mark.parametrize("file", FILES)
def test_example_cells_do_not_rebind_names_of_the_package(built, file):
    nb, defined = built[file], set()
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        names = bound_names(cell.source)
        if i in nb.package:
            defined |= names
        else:
            assert not {n for n in names if n in defined and not n.startswith("_")} - {"pd", "np", "torch", "os"}


def _namespace(nb) -> dict:
    """Runs the package cells of a notebook in one namespace, as the kernel does."""
    module = types.ModuleType("notebook_namespace")
    sys.modules[module.__name__] = module
    for i in sorted(nb.package):
        code = compile(nb.cells[i].source, f"<cell {i}>", "exec", flags=__future__.annotations.compiler_flag, dont_inherit=True)
        exec(code, module.__dict__)
    return module.__dict__


def _history(rows):
    return [{k: v for k, v in row.items() if k != "seconds"} for row in rows]


def _state(path):
    return torch.load(path, map_location="cpu", weights_only=False)["agent"]


# one environment ends an episode at the time limit (truncation) after 1000 steps, so the PPO run ends with the truncation and the
# SAC run continues for 100 steps; a negative cost limit makes the multiplier rise from the first update, so the cost critic and the
# multiplier take part in the comparison
PPO_SETTINGS = dict(total_steps=1000, num_envs=1, num_steps=250, update_epochs=2, num_minibatches=4, cost_limit=-1.0)
SAC_SETTINGS = dict(total_steps=1100, num_envs=1, sac_hidden=32, sac_batch_size=32, sac_learning_starts=100, sac_lambda_interval=100,
                    cost_limit=-1.0)


@pytest.mark.parametrize("file, method, learner, settings", [
    ("DMDP_PPO.ipynb", "DMDP-Lag", "ppo", PPO_SETTINGS),
    ("DMDP_PPO.ipynb", "Implicit-PPO-Lag", "ppo", PPO_SETTINGS),
    ("DMDP_SAC.ipynb", "DMDP-Lag", "sac", SAC_SETTINGS),
    ("DMDP_SAC.ipynb", "Implicit-SAC-Lag", "sac", SAC_SETTINGS),
])
def test_the_notebook_code_trains_and_evaluates_like_the_package(built, tmp_path, file, method, learner, settings):
    ns = _namespace(built[file])
    from dmdp_sg import evaluate, sac, train
    package_train = sac.train_sac if learner == "sac" else train.train
    notebook_train = ns["train_sac"] if learner == "sac" else ns["train"]

    cfg_package = __import__("dmdp_sg.config", fromlist=["make_config"]).make_config("HalfCheetah", method, 0, learner=learner, **settings)
    cfg_notebook = ns["make_config"]("HalfCheetah", method, 0, learner=learner, **settings)
    assert cfg_notebook.to_dict() == cfg_package.to_dict()

    history_package = package_train(cfg_package, str(tmp_path / "package"), verbose=False)
    history_notebook = notebook_train(cfg_notebook, str(tmp_path / "notebook"), verbose=False)
    assert _history(history_notebook) == _history(history_package)
    state_package, state_notebook = _state(tmp_path / "package" / "policy.pt"), _state(tmp_path / "notebook" / "policy.pt")
    assert state_package.keys() == state_notebook.keys()
    assert all(torch.equal(state_package[k], state_notebook[k]) for k in state_package)

    # the policy of the package is loaded and evaluated by both versions of the code
    path = str(tmp_path / "package" / "policy.pt")
    agent_package, _, loaded_package = evaluate.load_policy(path)
    agent_notebook, _, loaded_notebook = ns["load_policy"](path)
    assert type(agent_notebook).__name__ == type(agent_package).__name__ and loaded_notebook.to_dict() == loaded_package.to_dict()
    result_package = evaluate.evaluate_policy(path, (0.0, 0.2), episodes=2, horizon=6)
    result_notebook = ns["evaluate_policy"](path, (0.0, 0.2), episodes=2, horizon=6)
    assert result_package.keys() == result_notebook.keys()
    assert all(np.array_equal(result_package[k], result_notebook[k], equal_nan=True) for k in result_package)


# ---------------------------------------------------------------------------------------------------------------------
# the cutting of a module
# ---------------------------------------------------------------------------------------------------------------------
SOURCE = '''"""Docstring."""

from __future__ import annotations

import os
from typing import Any

from .targets import target_mean as _target_mean
from .config import TrainConfig

LIMIT = 3   # a constant

# A comment in front of the function
# and a second line.
def f(x):
    """Doc."""
    return x   # comment inside


@decorator
class C:
    pass
'''


def _module(tmp_path):
    path = tmp_path / "toy.py"
    path.write_text(SOURCE)
    return Module("toy", str(path))


def test_the_cell_of_a_definition_keeps_the_comments_in_front_of_it(tmp_path):
    mod = _module(tmp_path)
    kinds = [(it.kind, it.names) for it in mod.items]
    assert kinds == [("docstring", ()), ("import", ()), ("import", ()), ("import", ()), ("import", ()), ("import", ()),
                     ("const", ("LIMIT",)), ("def", ("f",)), ("class", ("C",))]
    assert mod.text_of(6, 6) == "LIMIT = 3   # a constant"
    assert mod.text_of(7, 7) == '# A comment in front of the function\n# and a second line.\ndef f(x):\n    """Doc."""\n    return x   # comment inside'
    assert mod.text_of(8, 8) == "@decorator\nclass C:\n    pass"


def test_relative_imports_are_commented_out_and_an_alias_is_assigned(tmp_path):
    mod = _module(tmp_path)
    text, aliases = mod.import_cell()
    assert aliases == ["_target_mean = target_mean"]
    assert text.splitlines() == [
        "# The imports between the modules of the package are commented out: their names are defined in the cells above.",
        "import os", "from typing import Any", "",
        "# from .targets import target_mean as _target_mean", "# from .config import TrainConfig",
        "_target_mean = target_mean   # the alias of the import above"]
    assert mod.relative_imports() == {"targets": ["target_mean"], "config": ["TrainConfig"]}
    only, _ = mod.import_cell({"os"})
    assert only == "import os"


def test_the_closure_of_a_definition_contains_the_definitions_that_it_uses(tmp_path):
    path = tmp_path / "toy2.py"
    path.write_text("A = 1\n\n\ndef g():\n    return A\n\n\ndef h():\n    return g()\n\n\ndef unused():\n    return 0\n")
    mod = Module("toy2", str(path))
    assert [mod.items[i].names[0] for i in mod.closure(["h"])] == ["A", "g", "h"]
