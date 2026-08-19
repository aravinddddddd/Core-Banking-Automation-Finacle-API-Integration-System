from __future__ import annotations

import pytest
from conftest import ScriptedProvider, text_response, tool_response

from polyagent.agent import Agent
from polyagent.errors import AgentLimitError
from polyagent.types import Message, ToolCall


def test_returns_text_without_calling_tools(echo_tool):
    provider = ScriptedProvider([text_response("done")])
    run = Agent(provider, tools=[echo_tool]).run("hi")

    assert run.output == "done"
    assert len(run.steps) == 1
    assert run.usage.input_tokens == 10


def test_executes_tool_then_answers(echo_tool):
    provider = ScriptedProvider(
        [
            tool_response(ToolCall(id="t1", name="echo", arguments={"text": "hey"})),
            text_response("said hey"),
        ]
    )
    run = Agent(provider, tools=[echo_tool]).run("say hey")

    assert run.output == "said hey"
    assert [c.name for c in run.tool_calls] == ["echo"]
    assert run.steps[0].results[0].content == "echo:hey"
    assert run.usage.output_tokens == 10  # summed across both turns


def test_parallel_results_go_back_in_one_message(echo_tool):
    provider = ScriptedProvider(
        [
            tool_response(
                ToolCall(id="a", name="echo", arguments={"text": "1"}),
                ToolCall(id="b", name="echo", arguments={"text": "2"}),
            ),
            text_response("ok"),
        ]
    )
    Agent(provider, tools=[echo_tool]).run("both")

    # Second request: [user, assistant, tool-results] — one message for both.
    sent = provider.calls[1]["messages"]
    result_messages = [m for m in sent if m.tool_results]
    assert len(result_messages) == 1
    assert [r.id for r in result_messages[0].tool_results] == ["a", "b"]


def test_unknown_tool_is_reported_not_raised(echo_tool):
    provider = ScriptedProvider(
        [
            tool_response(ToolCall(id="t1", name="ghost", arguments={})),
            text_response("recovered"),
        ]
    )
    run = Agent(provider, tools=[echo_tool]).run("go")

    result = run.steps[0].results[0]
    assert result.is_error and "No tool named 'ghost'" in result.content
    assert "echo" in result.content  # tells the model what it may call
    assert run.output == "recovered"


def test_tool_exception_becomes_an_error_result(exploding_tool):
    provider = ScriptedProvider(
        [
            tool_response(ToolCall(id="t1", name="explode", arguments={})),
            text_response("handled"),
        ]
    )
    run = Agent(provider, tools=[exploding_tool]).run("go")

    assert run.steps[0].results[0].is_error
    assert "boom" in run.steps[0].results[0].content
    assert run.output == "handled"


def test_approval_hook_can_block_a_call(echo_tool):
    provider = ScriptedProvider(
        [
            tool_response(ToolCall(id="t1", name="echo", arguments={"text": "x"})),
            text_response("stopped"),
        ]
    )
    seen: list[ToolCall] = []

    def deny(call: ToolCall) -> bool:
        seen.append(call)
        return False

    run = Agent(provider, tools=[echo_tool], approve=deny).run("go")

    assert [c.name for c in seen] == ["echo"]
    assert run.steps[0].results[0].is_error
    assert "declined" in run.steps[0].results[0].content


def test_step_limit_raises(echo_tool):
    call = ToolCall(id="t", name="echo", arguments={"text": "loop"})
    provider = ScriptedProvider([tool_response(call) for _ in range(3)])

    with pytest.raises(AgentLimitError, match="after 3 steps"):
        Agent(provider, tools=[echo_tool], max_steps=3).run("loop forever")


def test_history_and_system_prompt_are_forwarded(echo_tool):
    provider = ScriptedProvider([text_response("ok")])
    history = [Message.user("earlier"), Message.assistant("noted")]

    Agent(provider, tools=[echo_tool], system="be terse").run("now", history=history)

    call = provider.calls[0]
    assert call["system"] == "be terse"
    assert [m.content for m in call["messages"]] == ["earlier", "noted", "now"]
    assert call["tools"][0].name == "echo"


def test_iter_steps_yields_progressively(echo_tool):
    provider = ScriptedProvider(
        [
            tool_response(ToolCall(id="t1", name="echo", arguments={"text": "a"})),
            text_response("fin"),
        ]
    )
    steps = list(Agent(provider, tools=[echo_tool]).iter_steps("go"))

    assert [s.index for s in steps] == [0, 1]
    assert steps[0].results[0].content == "echo:a"
    assert steps[1].response.message.content == "fin"


def test_agent_runs_with_no_tools():
    provider = ScriptedProvider([text_response("plain")])
    assert Agent(provider).run("hi").output == "plain"
    assert provider.calls[0]["tools"] == []
