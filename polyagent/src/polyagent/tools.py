"""Tool definition and dispatch.

A tool is an ordinary Python function. The `@tool` decorator derives a JSON
Schema from its type hints so the same definition can be handed to any
provider; each adapter reshapes that schema into its own tool format.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_args, get_origin, get_type_hints

from .errors import ToolExecutionError

_JSON_TYPES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _schema_for(annotation: Any) -> dict[str, Any]:
    origin = get_origin(annotation)
    if origin is None:
        return {"type": _JSON_TYPES.get(annotation, "string")}
    if origin in (list, set, tuple):
        args = get_args(annotation)
        item = _schema_for(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item}
    if origin is dict:
        return {"type": "object"}
    # Optional[X] / Union[X, None] -> schema of X. A wider union has no faithful
    # single-type representation, so fall back to the first member.
    args = [a for a in get_args(annotation) if a is not type(None)]
    return _schema_for(args[0]) if args else {"type": "string"}


def _split_docstring(func: Callable[..., Any]) -> tuple[str, dict[str, str]]:
    """Return (summary, {param: description}) from a Google-style docstring."""
    doc = inspect.getdoc(func) or ""
    summary_lines: list[str] = []
    params: dict[str, str] = {}
    in_args = False
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped.lower() in ("args:", "arguments:", "parameters:"):
            in_args = True
            continue
        if in_args:
            if not stripped:
                continue
            if stripped.lower().startswith(("returns:", "raises:")):
                in_args = False
                continue
            name, sep, desc = stripped.partition(":")
            if sep:
                params[name.strip()] = desc.strip()
        else:
            summary_lines.append(stripped)
    return " ".join(line for line in summary_lines if line).strip(), params


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

    def call(self, arguments: dict[str, Any]) -> str:
        """Invoke the tool and normalise its return value to a string.

        Argument validation is intentionally minimal: models occasionally emit
        an extra key, and rejecting the whole call for that is worse than
        letting Python raise a TypeError we can report back as a tool error.
        """
        try:
            result = self.func(**arguments)
        except TypeError as exc:
            raise ToolExecutionError(f"invalid arguments for {self.name}: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - surfaced to the model, not swallowed
            raise ToolExecutionError(f"{self.name} failed: {exc}") from exc
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, default=str)
        except (TypeError, ValueError):
            return str(result)


def tool(func: Callable[..., Any]) -> Tool:
    """Turn a function into a Tool, deriving its schema from type hints.

    The docstring summary becomes the tool description and any `Args:` entries
    become per-parameter descriptions, so a well-documented function needs no
    separate schema.
    """
    hints = get_type_hints(func)
    hints.pop("return", None)
    signature = inspect.signature(func)
    summary, param_docs = _split_docstring(func)

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in signature.parameters.items():
        if name in ("self", "cls"):
            continue
        schema = _schema_for(hints.get(name, str))
        if name in param_docs:
            schema["description"] = param_docs[name]
        properties[name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return Tool(
        name=func.__name__,
        description=summary or f"Call {func.__name__}.",
        parameters={
            "type": "object",
            "properties": properties,
            "required": required,
        },
        func=func,
    )


class Toolbox:
    """A name-indexed collection of tools."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, t: Tool) -> None:
        if t.name in self._tools:
            raise ValueError(f"duplicate tool name: {t.name}")
        self._tools[t.name] = t

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    def __contains__(self, name: object) -> bool:
        return name in self._tools
