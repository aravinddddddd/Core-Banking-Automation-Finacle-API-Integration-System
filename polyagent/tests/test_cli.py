from __future__ import annotations

import pytest
from conftest import ScriptedProvider, text_response, tool_response

from polyagent import cli
from polyagent.errors import ConfigurationError
from polyagent.types import ToolCall


@pytest.fixture
def scripted(monkeypatch):
    """Replace provider construction so the CLI never touches the network."""

    def install(*responses):
        provider = ScriptedProvider(list(responses))
        monkeypatch.setattr(cli, "load", lambda spec, **kw: provider)
        return provider

    return install


def test_prints_answer_and_returns_zero(scripted, capsys):
    scripted(text_response("42"))
    assert cli.main(["-q", "what is 42"]) == 0
    assert capsys.readouterr().out.strip() == "42"


def test_reports_tool_calls_on_stderr(scripted, capsys, tmp_path):
    scripted(
        tool_response(ToolCall(id="1", name="calculate", arguments={"expression": "2+2"})),
        text_response("4"),
    )
    assert cli.main(["--root", str(tmp_path), "compute"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "4"
    assert "-> calculate" in captured.err
    assert "tokens]" in captured.err


def test_no_tools_flag_disables_the_toolbox(scripted, capsys):
    provider = scripted(text_response("plain"))
    assert cli.main(["--no-tools", "-q", "hi"]) == 0
    assert provider.calls[0]["tools"] == []


def test_step_limit_exits_nonzero(scripted, capsys):
    call = ToolCall(id="1", name="calculate", arguments={"expression": "1+1"})
    scripted(tool_response(call), tool_response(call))
    assert cli.main(["--max-steps", "2", "loop"]) == 1
    assert "still calling tools" in capsys.readouterr().err


def test_missing_prompt_exits_two(capsys, monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    assert cli.main([]) == 2
    assert "no prompt" in capsys.readouterr().err


def test_configuration_error_is_reported_not_raised(monkeypatch, capsys):
    def explode(spec, **kw):
        raise ConfigurationError("no key configured")

    monkeypatch.setattr(cli, "load", explode)
    assert cli.main(["hello"]) == 1
    assert "error: no key configured" in capsys.readouterr().err


def test_approval_declines_without_a_tty(scripted, monkeypatch, capsys):
    scripted(
        tool_response(ToolCall(id="1", name="calculate", arguments={"expression": "2+2"})),
        text_response("stopped"),
    )
    monkeypatch.setattr("builtins.input", lambda: (_ for _ in ()).throw(EOFError()))
    assert cli.main(["--approve", "-q", "compute"]) == 0
    assert capsys.readouterr().out.strip() == "stopped"


def test_parser_exposes_documented_defaults():
    args = cli.build_parser().parse_args(["hi"])
    assert args.provider == "anthropic"
    assert args.max_steps == 10 and args.max_tokens == 16000
    assert args.allow_write is False  # writes are opt-in
