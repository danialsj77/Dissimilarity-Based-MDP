"""Assembles a notebook from markdown cells, code cells copied from the package, and example cells.

``Notebook.take`` copies the source of named definitions of a module without change, and ``Notebook.finish`` checks that
every definition of a module was copied, so that no part of a module is left out of its section by accident.
"""

from __future__ import annotations

import nbformat as nbf

from .slicing import Module, used_names


class Notebook:
    def __init__(self) -> None:
        self.cells: list = []
        self.package: set[int] = set()        # indices of the code cells that come from the package (or set the notebook up)
        self._modules: dict[str, Module] = {}
        self._taken: dict[str, set[int]] = {}

    # ------------------------------------------------------------------ cells
    def md(self, text: str) -> None:
        self.cells.append(nbf.v4.new_markdown_cell(text.strip("\n"), id=f"cell-{len(self.cells):03d}"))     # fixed ids: reproducible

    def code(self, text: str, package: bool = False) -> None:
        if package:
            self.package.add(len(self.cells))
        self.cells.append(nbf.v4.new_code_cell(text.strip("\n"), id=f"cell-{len(self.cells):03d}"))

    # ------------------------------------------------------------------ package sources
    def module(self, name: str) -> Module:
        if name not in self._modules:
            self._modules[name] = Module(name)
            self._taken[name] = set()
        return self._modules[name]

    def docstring(self, name: str) -> None:
        """The module docstring, collapsed under a short summary."""
        text = self.module(name).docstring()
        self.md(f"<details>\n<summary>Docstring of <code>{name}.py</code></summary>\n\n```text\n{text}\n```\n\n</details>")

    def imports(self, name: str, used: set[str] | None = None) -> None:
        """The import cell of a module (relative imports commented out), or only the imports that bind a name of ``used``."""
        text, _ = self.module(name).import_cell(used)
        if text.strip():
            self.code(text, package=True)

    def _index(self, mod: Module, name: str) -> int:
        hits = [i for i, it in enumerate(mod.items) if it.kind != "import" and name in it.names]
        if len(hits) != 1:
            raise KeyError(f"{mod.name}: {len(hits)} definitions of {name!r}")
        return hits[0]

    def take(self, name: str, *names: str, merge: bool = False) -> None:
        """Code cells with the source of the definitions ``names`` of module ``name``: one cell each, or one cell for the
        consecutive definitions (``merge``), including the comments in front of them."""
        mod = self.module(name)
        idx = [self._index(mod, n) for n in names]
        for i in idx:
            if i in self._taken[name]:
                raise ValueError(f"{name}.{mod.items[i].names} is copied twice")
            self._taken[name].add(i)
        if merge:
            lo, hi = min(idx), max(idx)
            if sorted(idx) != list(range(lo, hi + 1)):
                raise ValueError(f"{name}: {names} are not consecutive definitions")
            self.code(mod.text_of(lo, hi), package=True)
        else:
            for i in idx:
                self.code(mod.text_of(i, i), package=True)

    def skip(self, name: str, *names: str) -> None:
        """Marks definitions as shown by a custom cell (``take`` is not used for them)."""
        mod = self.module(name)
        self._taken[name].update(self._index(mod, n) for n in names)

    def finish(self, name: str) -> None:
        """Every definition of the module must have been copied."""
        mod = self.module(name)
        missing = [it.names or (f"line {it.start}",) for i, it in enumerate(mod.items)
                   if it.kind not in ("import", "docstring") and i not in self._taken[name]]
        if missing:
            raise ValueError(f"{name}.py: definitions not copied: {missing}")

    # ------------------------------------------------------------------ writing
    def write(self, path: str, metadata: dict) -> None:
        nb = nbf.v4.new_notebook()
        nb.cells = self.cells
        nb.metadata = metadata
        nbf.validate(nb)
        nbf.write(nb, path)


__all__ = ["Notebook", "Module", "used_names"]
