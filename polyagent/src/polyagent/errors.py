"""Exception hierarchy.

Adapters translate vendor exceptions into these so calling code can handle
failures without importing every provider SDK.
"""

from __future__ import annotations


class PolyAgentError(Exception):
    """Base class for every error raised by this package."""


class ProviderError(PolyAgentError):
    """A provider call failed."""

    def __init__(
        self, message: str, *, status_code: int | None = None, retryable: bool = False
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class ConfigurationError(PolyAgentError):
    """A provider is missing a key, package, or setting it needs to run."""


class ToolExecutionError(PolyAgentError):
    """A tool raised while executing. Reported back to the model as a tool error."""


class AgentLimitError(PolyAgentError):
    """The agent hit its step ceiling before the model finished."""
