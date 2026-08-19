"""A small set of ready-made tools.

The filesystem tools are confined to a root directory. Any path that escapes it
after resolution is rejected, so a model that asks for `../../etc/passwd` gets a
tool error instead of the file.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .tools import Tool, tool

_OPERATORS: dict[type[ast.AST], Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

#: Guards against `9**9**9` locking up the process on a single expression.
MAX_EXPONENT = 1000


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"unsupported constant: {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp):
        op = _OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError(f"exponent above {MAX_EXPONENT} is not allowed")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        return op(_evaluate(node.operand))
    raise ValueError(f"unsupported expression: {type(node).__name__}")


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression such as "2 * (3 + 4)".

    Args:
        expression: the arithmetic to evaluate
    """
    # Parsed to an AST and walked node by node — `eval` on model output would
    # be arbitrary code execution.
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"could not parse {expression!r}: {exc.msg}") from exc
    return str(_evaluate(tree.body))


def filesystem_tools(root: str | Path = ".", *, writable: bool = False) -> list[Tool]:
    """Build read (and optionally write) tools confined to `root`.

    Args:
        root: directory the tools may touch
        writable: include `write_file` as well as the read-only tools
    """
    base = Path(root).resolve()

    def resolve(relative: str) -> Path:
        target = (base / relative).resolve()
        # `is_relative_to` compares resolved paths, so symlinks pointing out of
        # the sandbox are caught too.
        if not target.is_relative_to(base):
            raise ValueError(f"path {relative!r} is outside the allowed directory")
        return target

    @tool
    def read_file(path: str) -> str:
        """Read a UTF-8 text file.

        Args:
            path: file path relative to the working directory
        """
        target = resolve(path)
        if not target.is_file():
            raise ValueError(f"no such file: {path}")
        return target.read_text(encoding="utf-8", errors="replace")

    @tool
    def list_files(path: str = ".") -> str:
        """List the entries in a directory.

        Args:
            path: directory relative to the working directory
        """
        target = resolve(path)
        if not target.is_dir():
            raise ValueError(f"not a directory: {path}")
        entries = sorted(
            f"{child.name}/" if child.is_dir() else child.name for child in target.iterdir()
        )
        return "\n".join(entries) or "(empty)"

    tools = [read_file, list_files]

    if writable:

        @tool
        def write_file(path: str, content: str) -> str:
            """Write UTF-8 text to a file, creating parent directories.

            Args:
                path: file path relative to the working directory
                content: the text to write
            """
            target = resolve(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"wrote {len(content)} characters to {path}"

        tools.append(write_file)

    return tools
