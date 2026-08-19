"""Adapter for a local Ollama server.

Ollama needs no API key and no network egress, so it is the provider to use
when you want to run this agent offline — or when someone wants to try the
project without signing up for anything.
"""

from __future__ import annotations

import os
from typing import Any

from ..tools import Tool
from ..types import Message, ModelResponse, Role, StopReason, ToolCall, Usage, parse_arguments
from ._http import post_json
from .base import LLMProvider, with_retries

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1"

STATE_KEY = "ollama_message"


class OllamaProvider(LLMProvider):
    """Calls `POST {host}/api/chat`."""

    name = "ollama"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        host: str | None = None,
        timeout: float = 300.0,
        transport: Any = post_json,
    ) -> None:
        self._model = model
        self._host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        self._timeout = timeout
        self._transport = transport

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
        """Convert neutral messages into Ollama's `messages` array.

        Ollama's `tool` messages carry no `tool_call_id`, so results are
        correlated by order. Emitting them in call order is what keeps a
        parallel tool turn coherent.
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
                entry: dict[str, Any] = {"role": "assistant", "content": msg.content}
                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {"function": {"name": c.name, "arguments": c.arguments}}
                        for c in msg.tool_calls
                    ]
                out.append(entry)
                continue

            if msg.tool_results:
                out.extend({"role": "tool", "content": r.content} for r in msg.tool_results)
            elif msg.content:
                out.append({"role": "user", "content": msg.content})
        return out

    @classmethod
    def decode_response(cls, payload: dict[str, Any]) -> ModelResponse:
        raw_message = payload.get("message") or {}
        calls = [
            ToolCall(
                # Ollama returns no call id at all — synthesise a positional one
                # so the agent can still pair each result with its call.
                id=f"ollama_call_{index}",
                name=(call.get("function") or {}).get("name", ""),
                arguments=parse_arguments((call.get("function") or {}).get("arguments")),
            )
            for index, call in enumerate(raw_message.get("tool_calls") or [])
        ]

        stop = StopReason.TOOL_CALLS if calls else StopReason.END_TURN
        if payload.get("done_reason") == "length":
            stop = StopReason.MAX_TOKENS

        return ModelResponse(
            message=Message.assistant(
                content=raw_message.get("content") or "",
                tool_calls=calls,
                provider_state={STATE_KEY: raw_message},
            ),
            stop_reason=stop,
            usage=Usage(
                input_tokens=payload.get("prompt_eval_count", 0) or 0,
                output_tokens=payload.get("eval_count", 0) or 0,
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
            "messages": self.encode_messages(messages, system),
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        if tools:
            payload["tools"] = self.encode_tools(tools)

        def send() -> ModelResponse:
            return self.decode_response(
                self._transport(f"{self._host}/api/chat", payload, timeout=self._timeout)
            )

        return with_retries(send)
