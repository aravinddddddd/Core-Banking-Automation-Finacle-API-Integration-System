"""Provider registry.

`load("anthropic:claude-opus-5")` is the single place that maps a string onto a
concrete adapter, so switching backends is a config change rather than a code
change.
"""

from __future__ import annotations

from ..errors import ConfigurationError
from .base import LLMProvider, with_retries

__all__ = ["LLMProvider", "load", "available", "with_retries"]

#: Preset base URLs for services that speak the OpenAI protocol.
OPENAI_COMPATIBLE: dict[str, tuple[str, str]] = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
}

DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-opus-5",
    "openai": "gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
    "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "openrouter": "meta-llama/llama-3.3-70b-instruct",
    "deepseek": "deepseek-chat",
    "ollama": "llama3.1",
}


def available() -> list[str]:
    return ["anthropic", "ollama", *OPENAI_COMPATIBLE]


def load(spec: str, **kwargs) -> LLMProvider:
    """Build a provider from a `"<provider>"` or `"<provider>:<model>"` string.

    Raises ConfigurationError for an unknown provider rather than guessing, so
    a typo in a config file fails loudly at startup instead of at request time.
    """
    provider_name, _, model = spec.partition(":")
    provider_name = provider_name.strip().lower()
    model = model.strip() or DEFAULT_MODELS.get(provider_name, "")

    if provider_name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(model, **kwargs)

    if provider_name == "ollama":
        from .ollama import OllamaProvider

        return OllamaProvider(model, **kwargs)

    if provider_name in OPENAI_COMPATIBLE:
        from .openai_compat import OpenAICompatProvider

        base_url, key_env = OPENAI_COMPATIBLE[provider_name]
        kwargs.setdefault("base_url", base_url)
        kwargs.setdefault("api_key_env", key_env)
        provider = OpenAICompatProvider(model, **kwargs)
        provider.name = provider_name
        return provider

    raise ConfigurationError(
        f"Unknown provider {provider_name!r}. Available: {', '.join(available())}"
    )
