"""Round-trip tests for each adapter's wire-format translation."""

from __future__ import annotations

import pytest
from conftest import Block, SDKResponse

from polyagent.errors import ConfigurationError, ProviderError
from polyagent.providers import load
from polyagent.providers.anthropic_provider import STATE_KEY as ANTHROPIC_STATE
from polyagent.providers.anthropic_provider import AnthropicProvider
from polyagent.providers.base import with_retries
from polyagent.providers.ollama import OllamaProvider
from polyagent.providers.openai_compat import OpenAICompatProvider
from polyagent.types import Message, StopReason, ToolCall, ToolResult, parse_arguments

# -- Anthropic ------------------------------------------------------------


def test_anthropic_encodes_tool_results_as_blocks_in_one_user_turn(echo_tool):
    messages = [
        Message.user("hi"),
        Message.assistant(tool_calls=[ToolCall(id="tu_1", name="echo", arguments={"text": "a"})]),
        Message.tool_output(
            [
                ToolResult(id="tu_1", name="echo", content="ok"),
                ToolResult(id="tu_2", name="echo", content="bad", is_error=True),
            ]
        ),
    ]
    encoded = AnthropicProvider.encode_messages(messages)

    assert encoded[1]["content"][0]["type"] == "tool_use"
    results = encoded[2]
    assert results["role"] == "user" and len(results["content"]) == 2
    assert results["content"][0] == {"type": "tool_result", "tool_use_id": "tu_1", "content": "ok"}
    assert results["content"][1]["is_error"] is True


def test_anthropic_replays_original_blocks_so_thinking_survives():
    """Thinking blocks must be echoed back unchanged on the next request."""
    blocks = [
        {"type": "thinking", "thinking": "reasoning", "signature": "sig123"},
        {"type": "text", "text": "answer"},
    ]
    message = Message.assistant("answer", provider_state={ANTHROPIC_STATE: blocks})

    encoded = AnthropicProvider.encode_messages([message])

    assert encoded[0]["content"] == blocks  # byte-for-byte, signature intact


def test_anthropic_decodes_tool_use_and_captures_blocks():
    response = SDKResponse(
        content=[
            Block(type="thinking", thinking="hmm", signature="s1"),
            Block(type="text", text="calling now", citations=None),
            Block(type="tool_use", id="tu_9", name="echo", input={"text": "x"}),
        ],
        stop_reason="tool_use",
    )
    decoded = AnthropicProvider.decode_response(response)

    assert decoded.stop_reason is StopReason.TOOL_CALLS
    assert decoded.wants_tools
    assert decoded.message.content == "calling now"
    expected = ToolCall(id="tu_9", name="echo", arguments={"text": "x"})
    assert decoded.message.tool_calls[0] == expected
    assert decoded.usage.input_tokens == 7
    stored = decoded.message.provider_state[ANTHROPIC_STATE]
    assert stored[0]["signature"] == "s1"
    assert "citations" not in stored[1]  # exclude_none keeps the replay payload valid


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("end_turn", StopReason.END_TURN),
        ("max_tokens", StopReason.MAX_TOKENS),
        ("refusal", StopReason.REFUSAL),
        ("something_new", StopReason.OTHER),
    ],
)
def test_anthropic_stop_reason_mapping(raw, expected):
    decoded = AnthropicProvider.decode_response(
        SDKResponse(content=[Block(type="text", text="x")], stop_reason=raw)
    )
    assert decoded.stop_reason is expected


def test_anthropic_tool_schema_uses_input_schema_key(echo_tool):
    encoded = AnthropicProvider.encode_tools([echo_tool])[0]
    assert set(encoded) == {"name", "description", "input_schema"}
    assert encoded["input_schema"]["properties"]["text"]["type"] == "string"


def test_anthropic_sends_adaptive_thinking_by_default(echo_tool):
    captured: dict = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SDKResponse(content=[Block(type="text", text="hi")], stop_reason="end_turn")

    client = type("C", (), {"messages": FakeMessages()})()
    provider = AnthropicProvider("claude-opus-5", client=client)
    provider.complete([Message.user("hi")], tools=[echo_tool], system="sys")

    assert captured["thinking"] == {"type": "adaptive"}
    assert captured["model"] == "claude-opus-5"
    assert captured["system"] == "sys"
    assert "output_config" not in captured  # effort omitted unless configured


def test_anthropic_thinking_can_be_disabled_for_older_models():
    class FakeMessages:
        def create(self, **kwargs):
            assert "thinking" not in kwargs
            return SDKResponse(content=[Block(type="text", text="hi")], stop_reason="end_turn")

    client = type("C", (), {"messages": FakeMessages()})()
    provider = AnthropicProvider("claude-haiku-4-5", client=client, thinking={})
    assert provider.complete([Message.user("hi")]).message.content == "hi"


# -- OpenAI-compatible ----------------------------------------------------


def test_openai_encodes_tool_results_as_separate_tool_messages(echo_tool):
    messages = [
        Message.user("hi"),
        Message.assistant(tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "a"})]),
        Message.tool_output(
            [
                ToolResult(id="c1", name="echo", content="r1"),
                ToolResult(id="c2", name="echo", content="r2"),
            ]
        ),
    ]
    encoded = OpenAICompatProvider.encode_messages(messages, system="sys")

    assert encoded[0] == {"role": "system", "content": "sys"}
    # Arguments go out as a JSON *string*, unlike Anthropic's decoded object.
    assert encoded[2]["tool_calls"][0]["function"]["arguments"] == '{"text": "a"}'
    # One `tool` message per result — not one message holding both.
    assert [m["role"] for m in encoded[3:]] == ["tool", "tool"]
    assert encoded[3]["tool_call_id"] == "c1"


def test_openai_decodes_string_arguments_into_a_dict():
    decoded = OpenAICompatProvider.decode_response(
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "echo", "arguments": '{"text": "hi"}'},
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 4},
        }
    )

    assert decoded.stop_reason is StopReason.TOOL_CALLS
    assert decoded.message.content == ""  # null content normalised to empty string
    assert decoded.message.tool_calls[0].arguments == {"text": "hi"}
    assert decoded.usage.input_tokens == 11


def test_openai_missing_call_id_gets_a_positional_one():
    decoded = OpenAICompatProvider.decode_response(
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "tool_calls": [
                            {"function": {"name": "a", "arguments": "{}"}},
                            {"function": {"name": "b", "arguments": "{}"}},
                        ]
                    },
                }
            ]
        }
    )
    assert [c.id for c in decoded.message.tool_calls] == ["call_0", "call_1"]


def test_openai_provider_requires_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="No API key"):
        OpenAICompatProvider("gpt-4o-mini")


def test_openai_sends_expected_payload(echo_tool):
    seen: dict = {}

    def transport(url, payload, headers=None, timeout=None):
        seen.update(url=url, payload=payload, headers=headers)
        return {"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]}

    provider = OpenAICompatProvider("gpt-4o-mini", api_key="k", transport=transport)
    assert provider.complete([Message.user("q")], tools=[echo_tool]).message.content == "hi"
    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer k"
    assert seen["payload"]["tools"][0]["type"] == "function"


# -- Ollama ---------------------------------------------------------------


def test_ollama_synthesises_call_ids():
    decoded = OllamaProvider.decode_response(
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "echo", "arguments": {"text": "x"}}}],
            },
            "prompt_eval_count": 5,
            "eval_count": 2,
        }
    )

    assert decoded.stop_reason is StopReason.TOOL_CALLS
    assert decoded.message.tool_calls[0].id == "ollama_call_0"
    assert decoded.message.tool_calls[0].arguments == {"text": "x"}


def test_ollama_results_are_emitted_in_call_order():
    """Ollama has no tool_call_id, so ordering is the only correlation."""
    encoded = OllamaProvider.encode_messages(
        [
            Message.tool_output(
                [
                    ToolResult(id="ollama_call_0", name="a", content="first"),
                    ToolResult(id="ollama_call_1", name="b", content="second"),
                ]
            )
        ]
    )
    assert [m["content"] for m in encoded] == ["first", "second"]
    assert all("tool_call_id" not in m for m in encoded)


def test_ollama_needs_no_api_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    provider = OllamaProvider("llama3.1")
    assert provider.model == "llama3.1"


# -- shared behaviour -----------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ({"a": 1}, {"a": 1}),
        ('{"a": 1}', {"a": 1}),
        ("", {}),
        ("not json", {}),
        ("[1,2]", {}),  # valid JSON but not an object
        (None, {}),
    ],
)
def test_parse_arguments_is_total(raw, expected):
    assert parse_arguments(raw) == expected


def test_retries_only_retryable_errors():
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise ProviderError("429", status_code=429, retryable=True)
        return "ok"

    assert with_retries(flaky, sleep=lambda _: None, jitter=lambda: 0.5) == "ok"
    assert len(attempts) == 3


def test_non_retryable_error_fails_immediately():
    attempts = []

    def bad_request():
        attempts.append(1)
        raise ProviderError("400", status_code=400, retryable=False)

    with pytest.raises(ProviderError):
        with_retries(bad_request, sleep=lambda _: None)
    assert len(attempts) == 1


def test_retries_give_up_after_attempts():
    with pytest.raises(ProviderError):
        with_retries(
            lambda: (_ for _ in ()).throw(ProviderError("503", status_code=503, retryable=True)),
            attempts=2,
            sleep=lambda _: None,
        )


def test_backoff_grows_exponentially():
    delays: list[float] = []

    def always_fail():
        raise ProviderError("x", retryable=True)

    with pytest.raises(ProviderError):
        with_retries(
            always_fail, attempts=4, base_delay=1.0, sleep=delays.append, jitter=lambda: 0.5
        )

    assert delays == [1.0, 2.0, 4.0]


def test_registry_builds_configured_openai_clones(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    provider = load("groq:llama-3.3-70b-versatile")
    assert provider.name == "groq"
    assert provider.model == "llama-3.3-70b-versatile"


def test_registry_applies_default_model(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert load("ollama").model == "llama3.1"


def test_registry_rejects_unknown_provider():
    with pytest.raises(ConfigurationError, match="Unknown provider"):
        load("gpt5000")
