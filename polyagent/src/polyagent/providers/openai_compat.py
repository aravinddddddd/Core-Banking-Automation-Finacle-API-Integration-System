"""Adapter for any service speaking the OpenAI chat-completions protocol.

One adapter covers OpenAI, Groq, Together, OpenRouter, Fireworks, DeepSeek, and
self-hosted vLLM or LM Studio servers — they differ only by base URL, key, and
model name, so they are configuration rather than code.
"""

from __future__ import annotations

import json
import os
from typing import Any

from ..errors import ConfigurationError
from ..tools import Tool
from ..types import Message, ModelResponse, Role, StopReason, ToolCall, Usage, parse_arguments
from ._http import post_json
from .base import LLMProvider, with_retries

DEFAULT_BASE_URL = "https://api.openai.com/v1"

STATE_KEY = "openai_message"

_FINISH_REASONS = {
    "stop": StopReason.END_TURN,
    "tool_calls": StopReason.TOOL_CALLS,
    "function_call": StopReason.TOOL_CALLS,
    "length": StopReason.MAX_TOKENS,
    "content_filter": StopReason.REFUSAL,
}


class OpenAICompatProvider(LLMProvider):
    """Calls `POST {base_url}/chat/completions`."""

    name = "openai"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 120.0,
        transport: Any = post_json,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._transport = transport
        self._api_key = api_key or os.environ.get(api_key_env)
        if not self._api_key:
            raise ConfigurationError(
                f"No API key for {base_url}. Pass api_key= or set ${api_key_env}."
            )

    @property
    def model(self) -> str:
        return self._model

    # -- translation ------------------------------------------------------

    @staticmethod
    def encode_tools(tools: list[Tool]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    @staticmethod
    def encode_messages(messages: list[Message], system: str | None = None) -> list[dict[str, Any]]:
        """Convert neutral messages into OpenAI's `messages` array.

        Tool results become one `tool`-role message each — the mirror image of
        Anthropic, where they are blocks inside a single user turn.
        """
        out: list[dict[str, Any]] = []
        if system:
            out.append({"role": "system", "content": system})

        for msg in messages:
            if msg.role is Role.ASSISTANT:
                stored = msg.provider_state.get(STATE_KEY)
                if stored:
                    out.append(stored)
                    continue
                entry: dict[str, Any] = {"role": "assistant", "content": msg.content or None}
                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                        for call in msg.tool_calls
                    ]
                out.append(entry)
                continue

            if msg.tool_results:
                out.extend(
                    {"role": "tool", "tool_call_id": r.id, "content": r.content}
                    for r in msg.tool_results
                )
            elif msg.content:
                out.append({"role": "user", "content": msg.content})
        return out

    @classmethod
    def decode_response(cls, payload: dict[str, Any]) -> ModelResponse:
        choices = payload.get("choices") or [{}]
        choice = choices[0]
        raw_message = choice.get("message") or {}

        calls = [
            ToolCall(
                # Some OpenAI-compatible servers omit the id; index keeps it unique.
                id=call.get("id") or f"call_{index}",
                name=(call.get("function") or {}).get("name", ""),
                arguments=parse_arguments((call.get("function") or {}).get("arguments")),
            )
            for index, call in enumerate(raw_message.get("tool_calls") or [])
        ]

        finish = choice.get("finish_reason") or ""
        stop = _FINISH_REASONS.get(finish, StopReason.TOOL_CALLS if calls else StopReason.OTHER)
        usage = payload.get("usage") or {}

        return ModelResponse(
            message=Message.assistant(
                content=raw_message.get("content") or "",
                tool_calls=calls,
                provider_state={STATE_KEY: raw_message},
            ),
            stop_reason=stop,
            usage=Usage(
                input_tokens=usage.get("prompt_tokens", 0) or 0,
                output_tokens=usage.get("completion_tokens", 0) or 0,
            ),
            raw=payload,
        )

    # -- request ----------------------------------------------------------

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[Tool] | None = None,
        system: str | None = None,
        max_tokens: int = 16000,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": self.encode_messages(messages, system),
        }
        if tools:
            payload["tools"] = self.encode_tools(tools)

        def send() -> ModelResponse:
            return self.decode_response(
                self._transport(
                    f"{self._base_url}/chat/completions",
                    payload,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    timeout=self._timeout,
                )
            )

        return with_retries(send)
