"""Provider-neutral data types.

Every provider adapter converts its wire format into these types, and converts
these types back into its wire format. Nothing outside `providers/` imports a
vendor SDK, which is what lets the agent loop stay provider-agnostic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass(frozen=True)
class ToolCall:
    """A model's request to invoke a tool.

    `id` is the provider's correlation id. Anthropic and OpenAI both supply one;
    Ollama does not, so its adapter synthesises a stable id instead. The agent
    loop relies on the id to pair results with calls, so it is never optional.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    """One turn of conversation, in provider-neutral form.

    `provider_state` carries the adapter's original payload for this turn so it
    can be replayed byte-for-byte. Anthropic models emit `thinking` blocks that
    must be echoed back unchanged on the following request; flattening a turn to
    plain text would drop them. Adapters write their own key here and ignore
    everyone else's, so a transcript stays portable even when it has been
    round-tripped through a provider that needs extra state.
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    provider_state: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(
        cls,
        content: str = "",
        tool_calls: list[ToolCall] | None = None,
        provider_state: dict[str, Any] | None = None,
    ) -> Message:
        return cls(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=tool_calls or [],
            provider_state=provider_state or {},
        )

    @classmethod
    def tool_output(cls, results: list[ToolResult]) -> Message:
        """Tool results travel as a user-role message.

        Anthropic models tool results as `tool_result` blocks inside a user turn;
        OpenAI uses a dedicated `tool` role. Storing them on a user message keeps
        one canonical shape, and each adapter re-splits them on the way out.
        """
        return cls(role=Role.USER, tool_results=results)


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class StopReason(str, Enum):
    END_TURN = "end_turn"
    TOOL_CALLS = "tool_calls"
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"
    OTHER = "other"


@dataclass(frozen=True)
class ModelResponse:
    message: Message
    stop_reason: StopReason
    usage: Usage = field(default_factory=Usage)
    raw: Any = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.message.tool_calls)


def parse_arguments(raw: Any) -> dict[str, Any]:
    """Coerce a provider's argument payload into a dict.

    OpenAI sends tool arguments as a JSON *string*; Anthropic sends a decoded
    object. Malformed JSON is surfaced as an empty dict so the agent can return
    a tool error to the model rather than crashing the process.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}
