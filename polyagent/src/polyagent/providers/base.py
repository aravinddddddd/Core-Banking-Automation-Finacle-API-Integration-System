"""The contract every provider adapter implements."""

from __future__ import annotations

import abc
import random
import time
from collections.abc import Callable
from typing import TypeVar

from ..errors import ProviderError
from ..tools import Tool
from ..types import Message, ModelResponse

T = TypeVar("T")


class LLMProvider(abc.ABC):
    """Adapter for one model backend.

    Implementations own three translations: neutral messages to the wire
    format, neutral tool schemas to the vendor's tool format, and the vendor's
    response back to a `ModelResponse`. Everything else in the package is
    written against this interface.
    """

    #: Short identifier used by `providers.load()`, e.g. "anthropic".
    name: str = "unset"

    @abc.abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[Tool] | None = None,
        system: str | None = None,
        max_tokens: int = 16000,
    ) -> ModelResponse:
        """Send one request and return the model's reply."""

    @property
    @abc.abstractmethod
    def model(self) -> str:
        """The model identifier this adapter is configured for."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} model={self.model!r}>"


def with_retries(
    call: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
) -> T:
    """Retry `call` on retryable provider errors with exponential backoff.

    Only errors flagged `retryable` are retried — a 400 will fail the same way
    every time, and retrying it just multiplies the latency of a broken request.
    `sleep` and `jitter` are injectable so tests run without real delays.
    """
    last: ProviderError | None = None
    for attempt in range(attempts):
        try:
            return call()
        except ProviderError as exc:
            if not exc.retryable or attempt == attempts - 1:
                raise
            last = exc
            sleep(base_delay * (2**attempt) * (0.5 + jitter()))
    raise last  # pragma: no cover - loop always returns or raises above
