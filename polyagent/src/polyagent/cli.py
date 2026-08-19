"""Command-line entry point: `polyagent --provider ollama "your question"`."""

from __future__ import annotations

import argparse
import logging
import sys

from . import __version__
from .agent import Agent
from .builtins import calculate, filesystem_tools
from .errors import PolyAgentError
from .providers import available, load
from .types import ToolCall

DEFAULT_SYSTEM = (
    "You are a concise assistant with access to tools. "
    "Use a tool when it gives you a fact you do not already have, "
    "and answer directly when it does not."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polyagent",
        description="Run a tool-using agent against any supported LLM backend.",
        epilog=f"Providers: {', '.join(available())}",
    )
    parser.add_argument("prompt", nargs="*", help="the task for the agent")
    parser.add_argument(
        "-p",
        "--provider",
        default="anthropic",
        help="provider, optionally with a model: 'anthropic:claude-opus-5', 'ollama:llama3.1'",
    )
    parser.add_argument("--system", default=DEFAULT_SYSTEM, help="system prompt")
    parser.add_argument("--max-steps", type=int, default=10, help="tool-use turns before giving up")
    parser.add_argument("--max-tokens", type=int, default=16000, help="max tokens per response")
    parser.add_argument("--root", default=".", help="directory the file tools may access")
    parser.add_argument("--allow-write", action="store_true", help="also expose write_file")
    parser.add_argument("--no-tools", action="store_true", help="run without any tools")
    parser.add_argument("--approve", action="store_true", help="confirm each tool call")
    parser.add_argument("-q", "--quiet", action="store_true", help="print only the final answer")
    parser.add_argument("-v", "--verbose", action="store_true", help="log each request")
    parser.add_argument("--version", action="version", version=f"polyagent {__version__}")
    return parser


def _confirm(call: ToolCall) -> bool:
    print(f"\n  run {call.name}({call.arguments})? [y/N] ", end="", file=sys.stderr, flush=True)
    try:
        return input().strip().lower() in ("y", "yes")
    except EOFError:
        # Non-interactive stdin must not silently approve tool calls.
        print("no tty; declining", file=sys.stderr)
        return False


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    prompt = " ".join(args.prompt).strip()
    if not prompt:
        # Allows `echo "task" | polyagent -p ollama`.
        prompt = "" if sys.stdin.isatty() else sys.stdin.read().strip()
    if not prompt:
        print("error: no prompt given", file=sys.stderr)
        return 2

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        provider = load(args.provider)
        tools = (
            []
            if args.no_tools
            else [calculate, *filesystem_tools(args.root, writable=args.allow_write)]
        )
        agent = Agent(
            provider,
            tools=tools,
            system=args.system,
            max_steps=args.max_steps,
            max_tokens=args.max_tokens,
            approve=_confirm if args.approve else None,
        )

        if not args.quiet:
            print(f"[{provider.name}:{provider.model}] {len(tools)} tools", file=sys.stderr)

        last = None
        for step in agent.iter_steps(prompt):
            last = step
            if not args.quiet:
                for call in step.response.message.tool_calls:
                    print(f"  -> {call.name}({call.arguments})", file=sys.stderr)

        if last is None:
            print("error: agent produced no steps", file=sys.stderr)
            return 1
        if last.response.wants_tools:
            print(
                f"error: still calling tools after {args.max_steps} steps; raise --max-steps",
                file=sys.stderr,
            )
            return 1

        print(last.response.message.content)
        if not args.quiet:
            usage = last.response.usage
            print(f"[in {usage.input_tokens} / out {usage.output_tokens} tokens]", file=sys.stderr)
        return 0

    except PolyAgentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
