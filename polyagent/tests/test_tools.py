from __future__ import annotations

import pytest

from polyagent.errors import ToolExecutionError
from polyagent.tools import Toolbox, tool


def test_schema_is_derived_from_type_hints():
    @tool
    def search(query: str, limit: int = 10, tags: list[str] | None = None) -> str:
        """Search the index.

        Args:
            query: what to look for
            limit: how many results
        """
        return ""

    assert search.name == "search"
    assert search.description == "Search the index."
    props = search.parameters["properties"]
    assert props["query"] == {"type": "string", "description": "what to look for"}
    assert props["limit"]["type"] == "integer"
    assert props["tags"] == {"type": "array", "items": {"type": "string"}}
    # Only parameters without defaults are required.
    assert search.parameters["required"] == ["query"]


def test_description_falls_back_when_docstring_missing():
    @tool
    def bare(x: int) -> int:
        return x

    assert bare.description == "Call bare."


def test_non_string_return_is_json_encoded():
    @tool
    def stats() -> dict:
        """Return counters."""
        return {"hits": 2}

    assert stats.call({}) == '{"hits": 2}'


def test_unserialisable_return_falls_back_to_str():
    @tool
    def weird() -> object:
        """Return something json cannot encode."""
        return {1, 2}

    assert "1" in weird.call({})


def test_tool_exception_is_wrapped(exploding_tool):
    with pytest.raises(ToolExecutionError, match="explode failed: boom"):
        exploding_tool.call({})


def test_bad_arguments_are_wrapped(echo_tool):
    with pytest.raises(ToolExecutionError, match="invalid arguments"):
        echo_tool.call({"nope": 1})


def test_toolbox_rejects_duplicates(echo_tool):
    box = Toolbox([echo_tool])
    assert "echo" in box and len(box) == 1
    with pytest.raises(ValueError, match="duplicate tool name"):
        box.add(echo_tool)
