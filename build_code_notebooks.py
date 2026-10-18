"""Builds DMDP_PPO.ipynb and DMDP_SAC.ipynb, the notebooks that show the complete code of the PPO and the SAC learner.

Usage:
    python build_code_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 DMDP_PPO.ipynb
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 DMDP_SAC.ipynb

The code cells are copied without change from the package ``dmdp_sg/`` (``code_notebooks/slicing.py`` cuts the modules at
their top-level statements), so a notebook never drifts from the code that produced the results; only the imports between
the modules are commented out and ``evaluate.load_policy`` shows the branch of one learner. The markdown cells and the example
cells are written in ``code_notebooks/``. ``tests/test_code_notebooks.py`` checks that the notebooks equal this build.
"""

from __future__ import annotations

import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from code_notebooks import ppo, sac                       # noqa: E402
from code_notebooks.builder import Notebook               # noqa: E402

METADATA = {"kernelspec": {"display_name": "torch_safety", "language": "python", "name": "torch_safety"}}


class _Names(ast.NodeVisitor):
    """Names that the top-level statements of a cell bind (not those of function, class and comprehension scopes)."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_FunctionDef(self, node):
        self.names.add(node.name)

    visit_AsyncFunctionDef = visit_ClassDef = visit_FunctionDef

    def visit_Lambda(self, node):
        pass

    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = visit_Lambda

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)

    def visit_Import(self, node):
        self.names.update((a.asname or a.name).split(".")[0] for a in node.names)

    visit_ImportFrom = visit_Import


def bound_names(code: str) -> set[str]:
    visitor = _Names()
    visitor.visit(ast.parse(code))
    return visitor.names


def check(nb: Notebook, name: str) -> None:
    """The cells compile, and an example cell does not rebind a name of the package cells above it."""
    defined: set[str] = set()
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        try:
            names = bound_names(cell.source)
        except SyntaxError as error:
            raise SystemExit(f"{name}: cell {i} does not compile: {error}")
        if i in nb.package:
            defined |= names
        else:
            clash = {n for n in names if n in defined and not n.startswith("_")} - {"pd", "np", "torch", "os"}
            if clash:
                raise SystemExit(f"{name}: example cell {i} rebinds the names {sorted(clash)} of the package cells")


def build(write: bool = True) -> dict[str, Notebook]:
    notebooks = {ppo.FILE: ppo.build(), sac.FILE: sac.build()}
    for file, nb in notebooks.items():
        check(nb, file)
        if write:
            nb.write(os.path.join(HERE, file), METADATA)
    return notebooks


if __name__ == "__main__":
    for file, nb in build().items():
        print(f"{file}: {len(nb.cells)} cells, {len(nb.package)} copied from the package")
