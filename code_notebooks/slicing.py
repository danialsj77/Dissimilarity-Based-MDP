"""Splits the modules of ``dmdp_sg`` into the code cells of the notebooks ``DMDP_PPO.ipynb`` and ``DMDP_SAC.ipynb``.

A module is cut at its top-level statements with ``ast``; the comments and blank lines in front of a statement stay with
it, so a cell shows the source exactly as it is in the package. Only three things change:

* the imports between the modules of the package (``from .config import ...``) are commented out, because every name
  is defined in a cell above; an import with an alias (``... import x as y``) is followed by the assignment ``y = x``;
* ``from __future__ import annotations`` appears once, in the first cell of the notebook;
* a module can be reduced to some of its definitions (``only``), and a definition can be rewritten by a function
  (``transform``); the builder uses both for the learner-specific parts (``evaluate.load_policy``) and for the helpers
  that the SAC learner takes from the PPO modules.
"""

from __future__ import annotations

import ast
import os
import sys
from dataclasses import dataclass

PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dmdp_sg")
ROOT = os.path.dirname(PKG)


@dataclass
class Item:
    kind: str                 # "docstring", "import", "const", "def", "class" or "other"
    names: tuple[str, ...]    # names defined by the statement
    start: int                # first line of the statement (with its decorators), 1-indexed
    end: int                  # last line of the statement
    lead: int                 # first line of the comments in front of the statement that belong to it
    node: ast.AST


def _targets(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Assign):
        out: list[str] = []
        for t in node.targets:
            out.extend(n.id for n in ast.walk(t) if isinstance(n, ast.Name))
        return tuple(out)
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        return tuple(n.id for n in ast.walk(node.target) if isinstance(n, ast.Name))
    return ()


class Module:
    """One module of the package or a script of the repository, cut into its top-level statements."""

    def __init__(self, name: str, path: str | None = None):
        self.name = name
        self.path = path or os.path.join(PKG, name + ".py")
        with open(self.path) as f:
            self.text = f.read()
        self.lines = self.text.split("\n")
        self.tree = ast.parse(self.text)
        self.items = self._items()

    def _items(self) -> list[Item]:
        items: list[Item] = []
        prev_end = 0
        for i, node in enumerate(self.tree.body):
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            lead = start
            while lead - 1 > prev_end and (not self.lines[lead - 2].strip() or self.lines[lead - 2].lstrip().startswith("#")):
                lead -= 1
            while lead < start and not self.lines[lead - 1].strip():      # blank lines at the top of the comment block
                lead += 1
            if i == 0 and isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) \
                    and isinstance(node.value.value, str):
                kind, names = "docstring", ()
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                kind, names = "import", ()
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind, names = "def", (node.name,)
            elif isinstance(node, ast.ClassDef):
                kind, names = "class", (node.name,)
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                kind, names = "const", _targets(node)
            else:
                kind, names = "other", ()
            items.append(Item(kind, names, start, node.end_lineno, lead, node))
            prev_end = node.end_lineno
        return items

    # ---------------------------------------------------------------- text ranges
    def text_of(self, first: int, last: int) -> str:
        """The source of the items ``first`` to ``last`` (indices), with the comments in front of them."""
        a, b = self.items[first].lead, self.items[last].end
        return "\n".join(self.lines[a - 1:b])

    def docstring(self) -> str | None:
        return ast.get_docstring(self.tree, clean=False)

    def defined_names(self) -> set[str]:
        return {n for it in self.items for n in it.names}

    # ---------------------------------------------------------------- imports
    def imports(self) -> list[Item]:
        return [it for it in self.items if it.kind == "import"]

    def import_cell(self, used: set[str] | None = None) -> tuple[str, list[str]]:
        """The import cell of the module and the aliases (``y = x``) of its relative imports.

        With ``used`` (a set of names), only the imports that bind a name of the set are kept, rewritten from the syntax
        tree; otherwise the import block of the module is kept as it is. Relative imports are commented out.
        """
        imps = self.imports()
        if not imps:
            return "", []
        aliases: list[str] = []
        if used is not None:
            standard: list[str] = []
            other: list[str] = []
            for it in imps:
                node = it.node
                if isinstance(node, ast.ImportFrom) and (node.module == "__future__" or node.level > 0):
                    continue
                keep = [a for a in node.names if (a.asname or a.name.split(".")[0]) in used]
                if keep:
                    new = ast.ImportFrom(module=node.module, names=keep, level=0) if isinstance(node, ast.ImportFrom) \
                        else ast.Import(names=keep)
                    top = (node.module if isinstance(node, ast.ImportFrom) else keep[0].name).split(".")[0]
                    (standard if top in sys.stdlib_module_names else other).append(ast.unparse(new))
            return "\n\n".join("\n".join(group) for group in (standard, other) if group), aliases
        first, last = imps[0], imps[-1]
        out: list[str] = []
        commented = False
        i = first.start
        by_start = {it.start: it for it in imps}
        while i <= last.end:
            it = by_start.get(i)
            if it is None:
                out.append(self.lines[i - 1])
                i += 1
                continue
            node = it.node
            seg = self.lines[it.start - 1:it.end]
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                pass                                                                   # kept once, in the first cell
            elif isinstance(node, ast.ImportFrom) and node.level > 0:
                out.extend("# " + line for line in seg)
                commented = True
                for a in node.names:
                    if a.asname and a.asname != a.name:
                        aliases.append(f"{a.asname} = {a.name}")
            else:
                out.extend(seg)
            i = it.end + 1
        text = "\n".join(out).strip("\n")
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")
        if commented:
            text = "# The imports between the modules of the package are commented out: their names are defined in the cells above.\n" + text
        if aliases:
            text += "\n" + "\n".join(f"{a}   # the alias of the import above" for a in aliases)
        return text, aliases

    def relative_imports(self) -> dict[str, list[str]]:
        """The names that the module imports from the other modules of the package, by module."""
        out: dict[str, list[str]] = {}
        for it in self.imports():
            node = it.node
            if isinstance(node, ast.ImportFrom) and node.level > 0:
                out.setdefault(node.module, []).extend(a.name for a in node.names)
        return out

    def closure(self, names: list[str]) -> list[int]:
        """Indices (in source order) of the items that define ``names`` and, transitively, the names of the module that they use."""
        defined = {n: i for i, it in enumerate(self.items) if it.kind != "import" for n in it.names}
        todo, seen = list(names), set()
        while todo:
            i = defined[todo.pop()]
            if i in seen:
                continue
            seen.add(i)
            todo.extend(m for m in used_names([self.items[i].node]) if m in defined and defined[m] != i)
        return sorted(seen)


def used_names(nodes: list[ast.AST]) -> set[str]:
    """Names that the syntax trees read (the first part of every dotted name)."""
    used: set[str] = set()
    for node in nodes:
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                used.add(n.id)
    return used
