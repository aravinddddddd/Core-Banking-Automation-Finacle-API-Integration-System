"""Anthropic (Claude) adapter, built on the official `anthropic` SDK."""

from __future__ import annotations

from typing import Any

from ..errors import ConfigurationError, ProviderError
from ..tools import Tool
from ..types import Message, ModelResponse, Role, StopReason, ToolCall, Usage
from .base import LLMProvider, with_retries

DEFAULT_MODEL = "claude-opus-5"

#: Where this adapter stashes the turn's original content blocks. Claude emits
#: `thinking` blocks that must be replayed unchanged on the next request, so a
#: turn cannot be reconstructed from its text alone.
STATE_KEY = "anthropic_blocks"

_STOP_REASONS = {
    "end_turn": StopReason.END_TURN,
    "tool_use": StopReason.TOOL_CALLS,
    "max_tokens": StopReason.MAX_TOKENS,
    "refusal": StopReason.REFUSAL,
    "stop_sequence": StopReason.END_TURN,
}


def _load_sdk() -> Any:
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ConfigurationError(
            "The Anthropic provider needs the `anthropic` package. "
            "Install it with: pip install 'polyagent[anthropic]'"
        ) from exc
    return anthropic


class AnthropicProvider(LLMProvider):
    """Calls Claude through `POST /v1/messages`.

    Thinking defaults to `adaptive`, which is what current Claude models expect.
    Models predating adaptive thinking reject it, so pass `thinking=None` to
    omit the parameter entirely for those.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        client: Any = None,
        thinking: dict[str, Any] | None = None,
        effort: str | None = None,
        enable_server_fallback: bool = False,
    ) -> None:
        self._model = model
        self._thinking = {"type": "adaptive"} if thinking is None else thinking
        self._effort = effort
        self._enable_server_fallback = enable_server_fallback
        if client is not None:
            self._client = client
        else:
            anthropic = _load_sdk()
            # A bare constructor resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
            # or an `ant auth login` profile — so an unset env var is not
            # necessarily an unconfigured client.
            self._client = (
                anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
            )

    @property
    def model(self) -> str:
        return self._model

    # -- translation ------------------------------------------------------

    @staticmethod
    def encode_tools(tools: list[Tool]) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in tools
        ]

    @staticmethod
    def encode_messages(messages: list[Message]) -> list[dict[str, Any]]:
        """Convert neutral messages into Anthropic's `messages` array.

        Consecutive tool results are merged into a single user turn: splitting
        them across turns teaches the model to stop issuing parallel calls.
        """
        out: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role is Role.ASSISTANT:
                blocks = msg.provider_state.get(STATE_KEY)
                if blocks:
                    out.append({"role": "assistant", "content": blocks})
                    continue
                content: list[dict[str, Any]] = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for call in msg.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        }
                    )
                if content:
                    out.append({"role": "assistant", "content": content})
                continue

            if msg.tool_results:
                blocks = [
                    {
                        "type": "tool_result",
                        "tool_use_id": r.id,
                        "content": r.content,
                        **({"is_error": True} if r.is_error else {}),
                    }
                    for r in msg.tool_results
                ]
                out.append({"role": "user", "content": blocks})
            elif msg.content:
                out.append({"role": "user", "content": msg.content})
        return out

    @classmethod
    def decode_response(cls, response: Any) -> ModelResponse:
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        blocks: list[dict[str, Any]] = []

        for block in response.content:
            blocks.append(
                block.model_dump(exclude_none=True) if hasattr(block, "model_dump") else dict(block)
            )
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))

        stop = _STOP_REASONS.get(getattr(response, "stop_reason", "") or "", StopReason.OTHER)
        usage = getattr(response, "usage", None)
        message = Message.assistant(
            content="".join(text_parts),
            tool_calls=calls,
            provider_state={STATE_KEY: blocks},
        )
        return ModelResponse(
            message=message,
            stop_reason=stop,
            usage=Usage(
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
            ),
            raw=response,
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
        params: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": self.encode_messages(messages),
        }
        if self._thinking:
            params["thinking"] = self._thinking
        if self._effort:
            params["output_config"] = {"effort": self._effort}
        if system:
            params["system"] = system
        if tools:
            params["tools"] = self.encode_tools(tools)
        if self._enable_server_fallback:
            # Routes refusals to a fallback model server-side. Off by default:
            # it is a beta surface, and a provider-neutral caller that named a
            # model should not silently get a different one.
            params["betas"] = ["server-side-fallback-2026-07-01"]
            params["fallbacks"] = "default"

        def send() -> ModelResponse:
            try:
                target = (
                    self._client.beta.messages
                    if self._enable_server_fallback
                    else self._client.messages
                )
                return self.decode_response(target.create(**params))
            except Exception as exc:  # noqa: BLE001 - re-raised as ProviderError below
                raise self._translate(exc) from exc

        return with_retries(send)

    @staticmethod
    def _translate(exc: Exception) -> Exception:
        """Map SDK exceptions onto ProviderError, preserving retryability."""
        if isinstance(exc, ProviderError):
            return exc
        try:
            import anthropic
        except ImportError:  # pragma: no cover
            return ProviderError(str(exc))

        if isinstance(exc, anthropic.NotFoundError):
            return ProviderError(f"model or endpoint not found: {exc}", status_code=404)
        if isinstance(exc, anthropic.AuthenticationError):
            return ConfigurationError(f"Anthropic rejected the credentials: {exc}")
        if isinstance(exc, anthropic.RateLimitError):
            return ProviderError(f"rate limited: {exc}", status_code=429, retryable=True)
        if isinstance(exc, anthropic.APIStatusError):
            status = getattr(exc, "status_code", None)
            return ProviderError(
                str(exc), status_code=status, retryable=bool(status and status >= 500)
            )
        if isinstance(exc, (anthropic.APIConnectionError, anthropic.APITimeoutError)):
            return ProviderError(f"connection to Anthropic failed: {exc}", retryable=True)
        return exc
