"""Shared fakes. No test in this suite performs network I/O."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polyagent.providers.base import LLMProvider  # noqa: E402
from polyagent.tools import Tool, tool  # noqa: E402
from polyagent.types import (  # noqa: E402
    Message,
    ModelResponse,
    StopReason,
    ToolCall,
    Usage,
)


class ScriptedProvider(LLMProvider):
    """Replays a fixed list of ModelResponses and records what it was sent."""

    name = "scripted"

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    @property
    def model(self) -> str:
        return "scripted-1"

    def complete(self, messages, *, tools=None, system=None, max_tokens=16000):
        self.calls.append(
            {
                "messages": list(messages),
                "tools": list(tools or []),
                "system": system,
                "max_tokens": max_tokens,
            }
        )
        if not self._responses:
            raise AssertionError("ScriptedProvider ran out of scripted responses")
        return self._responses.pop(0)


def text_response(content: str, *, usage: Usage | None = None) -> ModelResponse:
    return ModelResponse(
        message=Message.assistant(content=content),
        stop_reason=StopReason.END_TURN,
        usage=usage or Usage(10, 5),
    )


def tool_response(*calls: ToolCall, usage: Usage | None = None) -> ModelResponse:
    return ModelResponse(
        message=Message.assistant(tool_calls=list(calls)),
        stop_reason=StopReason.TOOL_CALLS,
        usage=usage or Usage(10, 5),
    )


class Block:
    """Stand-in for an Anthropic SDK content block."""

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not (exclude_none and v is None)}


class SDKResponse:
    def __init__(self, content: list[Block], stop_reason: str, usage: Any = None) -> None:
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage or Block(input_tokens=7, output_tokens=3)


@pytest.fixture
def echo_tool() -> Tool:
    @tool
    def echo(text: str) -> str:
        """Echo the given text back.

        Args:
            text: what to echo
        """
        return f"echo:{text}"

    return echo


@pytest.fixture
def exploding_tool() -> Tool:
    @tool
    def explode() -> str:
        """Always raises, to exercise tool error handling."""
        raise RuntimeError("boom")

    return explode
