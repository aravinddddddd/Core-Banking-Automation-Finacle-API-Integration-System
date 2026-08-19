from __future__ import annotations

import pytest

from polyagent.builtins import calculate, filesystem_tools
from polyagent.errors import ToolExecutionError


@pytest.mark.parametrize(
    "expression,expected",
    [("2 + 3", "5"), ("2 * (3 + 4)", "14"), ("-5 + 1", "-4"), ("7 / 2", "3.5"), ("2 ** 8", "256")],
)
def test_calculate_arithmetic(expression, expected):
    assert calculate.call({"expression": expression}) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",  # code execution attempt
        "open('/etc/passwd').read()",
        "9 ** 9999",  # resource exhaustion
        "1 +",  # syntax error
        "'a' * 3",  # non-numeric constant
    ],
)
def test_calculate_rejects_unsafe_input(expression):
    with pytest.raises(ToolExecutionError):
        calculate.call({"expression": expression})


def test_filesystem_tools_read_and_list(tmp_path):
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    read_file, list_files = filesystem_tools(tmp_path)

    assert read_file.call({"path": "note.txt"}) == "hello"
    assert list_files.call({"path": "."}) == "note.txt\nsub/"


def test_filesystem_tools_are_read_only_by_default(tmp_path):
    assert [t.name for t in filesystem_tools(tmp_path)] == ["read_file", "list_files"]
    assert "write_file" in [t.name for t in filesystem_tools(tmp_path, writable=True)]


def test_write_file_round_trips(tmp_path):
    tools = {t.name: t for t in filesystem_tools(tmp_path, writable=True)}
    tools["write_file"].call({"path": "out/data.txt", "content": "abc"})
    assert tools["read_file"].call({"path": "out/data.txt"}) == "abc"


@pytest.mark.parametrize("escape", ["../secret.txt", "../../etc/passwd", "sub/../../outside.txt"])
def test_path_traversal_is_blocked(tmp_path, escape):
    (tmp_path.parent / "secret.txt").write_text("classified", encoding="utf-8")
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "sub").mkdir()
    read_file, _ = filesystem_tools(sandbox)

    with pytest.raises(ToolExecutionError, match="outside the allowed directory"):
        read_file.call({"path": escape})


def test_symlink_escaping_the_sandbox_is_blocked(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("classified", encoding="utf-8")
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "link.txt").symlink_to(outside)
    read_file, _ = filesystem_tools(sandbox)

    with pytest.raises(ToolExecutionError, match="outside the allowed directory"):
        read_file.call({"path": "link.txt"})


def test_missing_file_reports_clearly(tmp_path):
    read_file, _ = filesystem_tools(tmp_path)
    with pytest.raises(ToolExecutionError, match="no such file"):
        read_file.call({"path": "ghost.txt"})
