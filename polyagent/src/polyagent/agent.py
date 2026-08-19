"""The provider-neutral agent loop."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from .errors import AgentLimitError, ToolExecutionError
from .providers.base import LLMProvider
from .tools import Tool, Toolbox
from .types import Message, ModelResponse, StopReason, ToolCall, ToolResult, Usage

logger = logging.getLogger(__name__)

#: Return False from an approval hook to block a call. The model is told the
#: call was declined and can choose a different action.
ApprovalHook = Callable[[ToolCall], bool]


@dataclass
class Step:
    """One model turn plus any tool results it produced."""

    index: int
    response: ModelResponse
    results: list[ToolResult] = field(default_factory=list)


@dataclass
class Run:
    """The outcome of `Agent.run`."""

    output: str
    steps: list[Step]
    usage: Usage
    messages: list[Message]

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [call for step in self.steps for call in step.response.message.tool_calls]


class Agent:
    """Runs a model in a tool-use loop until it produces a final answer.

    The loop is written entirely against `LLMProvider` and the neutral types,
    which is what lets the same agent run on Claude, an OpenAI-compatible
    endpoint, or a local Ollama model without changes.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        tools: list[Tool] | None = None,
        system: str | None = None,
        max_steps: int = 10,
        max_tokens: int = 16000,
        approve: ApprovalHook | None = None,
    ) -> None:
        self.provider = provider
        self.toolbox = Toolbox(tools)
        self.system = system
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.approve = approve

    def run(self, prompt: str, *, history: list[Message] | None = None) -> Run:
        """Run to completion and return the final answer.

        Raises AgentLimitError if the model is still calling tools after
        `max_steps` turns — an unbounded loop against a paid API is a bug, not
        a feature.
        """
        steps: list[Step] = []
        for step in self.iter_steps(prompt, history=history):
            steps.append(step)

        last = steps[-1]
        if last.response.wants_tools:
            raise AgentLimitError(
                f"still calling tools after {self.max_steps} steps; raise max_steps "
                f"or narrow the task"
            )

        total = Usage()
        for step in steps:
            total = total + step.response.usage

        return Run(
            output=last.response.message.content,
            steps=steps,
            usage=total,
            messages=list(self._messages),
        )

    def iter_steps(self, prompt: str, *, history: list[Message] | None = None) -> Iterator[Step]:
        """Yield each step as it completes, for streaming progress to a UI."""
        self._messages: list[Message] = list(history or []) + [Message.user(prompt)]
        tools = list(self.toolbox) or None

        for index in range(self.max_steps):
            response = self.provider.complete(
                self._messages,
                tools=tools,
                system=self.system,
                max_tokens=self.max_tokens,
            )
            self._messages.append(response.message)
            step = Step(index=index, response=response)

            if not response.wants_tools:
                if response.stop_reason is StopReason.MAX_TOKENS:
                    logger.warning("response truncated at max_tokens=%d", self.max_tokens)
                yield step
                return

            step.results = [self._invoke(call) for call in response.message.tool_calls]
            # All results go back in one message; splitting them across turns
            # discourages the model from issuing parallel calls again.
            self._messages.append(Message.tool_output(step.results))
            yield step

    def _invoke(self, call: ToolCall) -> ToolResult:
        """Execute one tool call, converting every failure into a tool result.

        A tool that raises should not kill the run: the model gets the error
        text and usually recovers by fixing its arguments or trying another
        tool.
        """
        tool = self.toolbox.get(call.name)
        if tool is None:
            known = ", ".join(t.name for t in self.toolbox) or "none"
            return ToolResult(
                id=call.id,
                name=call.name,
                content=f"No tool named {call.name!r}. Available tools: {known}.",
                is_error=True,
            )

        if self.approve is not None and not self.approve(call):
            return ToolResult(
                id=call.id,
                name=call.name,
                content="The user declined this tool call. Do not retry it.",
                is_error=True,
            )

        logger.debug("calling %s(%s)", call.name, call.arguments)
        try:
            return ToolResult(id=call.id, name=call.name, content=tool.call(call.arguments))
        except ToolExecutionError as exc:
            logger.info("tool %s failed: %s", call.name, exc)
            return ToolResult(id=call.id, name=call.name, content=str(exc), is_error=True)
