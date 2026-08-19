"""Error-path tests for HTTP transport and vendor-exception translation."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest
from conftest import Block, SDKResponse

from polyagent.errors import ConfigurationError, ProviderError
from polyagent.providers import _http
from polyagent.providers.anthropic_provider import AnthropicProvider
from polyagent.providers.ollama import OllamaProvider
from polyagent.types import Message

# -- post_json ------------------------------------------------------------


def test_post_json_returns_decoded_body(monkeypatch):
    captured = {}

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        captured["headers"] = request.headers
        return Response(b'{"ok": true}')

    monkeypatch.setattr(_http.urllib.request, "urlopen", fake_urlopen)
    result = _http.post_json("http://x/api", {"a": 1}, headers={"Authorization": "Bearer k"})

    assert result == {"ok": True}
    assert captured["body"] == {"a": 1}
    assert captured["headers"]["Content-type"] == "application/json"
    assert captured["headers"]["Authorization"] == "Bearer k"


@pytest.mark.parametrize(
    "status,retryable",
    [(429, True), (500, True), (503, True), (400, False), (401, False), (404, False)],
)
def test_http_errors_carry_correct_retryability(monkeypatch, status, retryable):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, status, "err", {}, io.BytesIO(b"details"))

    monkeypatch.setattr(_http.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ProviderError) as info:
        _http.post_json("http://x/api", {})

    assert info.value.status_code == status
    assert info.value.retryable is retryable
    assert "details" in str(info.value)


def test_connection_failure_is_retryable(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(_http.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ProviderError, match="could not reach") as info:
        _http.post_json("http://x/api", {})
    assert info.value.retryable is True


def test_non_json_body_is_reported(monkeypatch):
    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        _http.urllib.request, "urlopen", lambda request, timeout=None: Response(b"<html>oops")
    )
    with pytest.raises(ProviderError, match="non-JSON body"):
        _http.post_json("http://x/api", {})


# -- Ollama request shape -------------------------------------------------


def test_ollama_posts_expected_payload(echo_tool):
    seen = {}

    def transport(url, payload, headers=None, timeout=None):
        seen.update(url=url, payload=payload)
        return {"message": {"role": "assistant", "content": "hi"}}

    provider = OllamaProvider("llama3.1", host="http://box:11434", transport=transport)
    response = provider.complete(
        [Message.user("q")], tools=[echo_tool], system="sys", max_tokens=64
    )

    assert response.message.content == "hi"
    assert seen["url"] == "http://box:11434/api/chat"
    assert seen["payload"]["stream"] is False
    assert seen["payload"]["options"]["num_predict"] == 64
    assert seen["payload"]["messages"][0] == {"role": "system", "content": "sys"}
    assert seen["payload"]["tools"][0]["function"]["name"] == "echo"


def test_ollama_host_comes_from_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://elsewhere:1234/")
    seen = {}

    def transport(url, payload, headers=None, timeout=None):
        seen["url"] = url
        return {"message": {"content": "ok"}}

    OllamaProvider("m", transport=transport).complete([Message.user("q")])
    assert seen["url"] == "http://elsewhere:1234/api/chat"


def test_ollama_length_stop_is_reported():
    decoded = OllamaProvider.decode_response(
        {"message": {"content": "truncated"}, "done_reason": "length"}
    )
    assert decoded.stop_reason.value == "max_tokens"


# -- Anthropic exception translation --------------------------------------


def _anthropic_error(cls_name: str, message: str = "boom"):
    """Build a real SDK exception without making a network call."""
    import anthropic
    import httpx

    cls = getattr(anthropic, cls_name)
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    status = {
        "NotFoundError": 404,
        "AuthenticationError": 401,
        "RateLimitError": 429,
        "InternalServerError": 500,
        "BadRequestError": 400,
    }[cls_name]
    response = httpx.Response(status, request=request, json={"error": {"message": message}})
    return cls(message, response=response, body=None)


@pytest.mark.parametrize(
    "cls_name,expected_type,retryable",
    [
        ("NotFoundError", ProviderError, False),
        ("AuthenticationError", ConfigurationError, False),
        ("RateLimitError", ProviderError, True),
        ("InternalServerError", ProviderError, True),
        ("BadRequestError", ProviderError, False),
    ],
)
def test_sdk_exceptions_map_to_package_errors(cls_name, expected_type, retryable):
    translated = AnthropicProvider._translate(_anthropic_error(cls_name))

    assert isinstance(translated, expected_type)
    if isinstance(translated, ProviderError):
        assert translated.retryable is retryable


def test_timeout_is_translated_as_retryable():
    import anthropic
    import httpx

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    translated = AnthropicProvider._translate(anthropic.APITimeoutError(request=request))
    assert isinstance(translated, ProviderError) and translated.retryable is True


def test_rate_limit_is_retried_then_succeeds(monkeypatch):
    attempts = []

    class FakeMessages:
        def create(self, **kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                raise _anthropic_error("RateLimitError")
            return SDKResponse(
                content=[Block(type="text", text="recovered")], stop_reason="end_turn"
            )

    monkeypatch.setattr("polyagent.providers.base.time.sleep", lambda _: None)
    client = type("C", (), {"messages": FakeMessages()})()
    provider = AnthropicProvider("claude-opus-5", client=client)

    assert provider.complete([Message.user("hi")]).message.content == "recovered"
    assert len(attempts) == 2


def test_bad_request_is_not_retried(monkeypatch):
    attempts = []

    class FakeMessages:
        def create(self, **kwargs):
            attempts.append(1)
            raise _anthropic_error("BadRequestError")

    monkeypatch.setattr("polyagent.providers.base.time.sleep", lambda _: None)
    client = type("C", (), {"messages": FakeMessages()})()

    with pytest.raises(ProviderError):
        AnthropicProvider("claude-opus-5", client=client).complete([Message.user("hi")])
    assert len(attempts) == 1


def test_server_fallback_targets_the_beta_endpoint():
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SDKResponse(content=[Block(type="text", text="ok")], stop_reason="end_turn")

    beta = type("B", (), {"messages": FakeMessages()})()
    client = type("C", (), {"beta": beta, "messages": None})()
    provider = AnthropicProvider("claude-opus-5", client=client, enable_server_fallback=True)
    provider.complete([Message.user("hi")])

    assert captured["fallbacks"] == "default"
    assert captured["betas"] == ["server-side-fallback-2026-07-01"]


def test_effort_is_sent_inside_output_config():
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SDKResponse(content=[Block(type="text", text="ok")], stop_reason="end_turn")

    client = type("C", (), {"messages": FakeMessages()})()
    AnthropicProvider("claude-opus-5", client=client, effort="low").complete([Message.user("hi")])

    assert captured["output_config"] == {"effort": "low"}
