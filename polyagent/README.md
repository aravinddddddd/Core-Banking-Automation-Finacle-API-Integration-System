# polyagent

One tool-using agent loop that runs on Claude, any OpenAI-compatible API, or a
local Ollama model. Switching backend is a string, not a rewrite:

```python
agent = Agent(load("anthropic:claude-opus-5"), tools=[calculate])
agent = Agent(load("groq:llama-3.3-70b-versatile"), tools=[calculate])
agent = Agent(load("ollama:llama3.1"), tools=[calculate])
```

The agent loop, the tool definitions, and your application code are identical in
all three cases.

## Why this is not a wrapper around one SDK

The three providers disagree about almost everything in the tool-calling
protocol, and the adapters exist to absorb that:

| | Claude | OpenAI-compatible | Ollama |
|---|---|---|---|
| Tool schema key | `input_schema` | `function.parameters` | `function.parameters` |
| Tool arguments | decoded object | **JSON string** | decoded object |
| Tool call id | `tool_use.id` | `tool_calls[].id` | **none — synthesised** |
| Results travel as | `tool_result` blocks in one user turn | one `tool` message each | one `tool` message each, correlated by order |
| Extra turn state | `thinking` blocks that must be replayed verbatim | — | — |

That last row is the subtle one. Claude returns `thinking` blocks carrying a
signature, and the next request has to send them back unchanged. Flattening an
assistant turn to plain text loses them, so `Message.provider_state` keeps each
adapter's original payload and replays it. Adapters ignore each other's keys, so
a transcript stays portable.

## Install

```bash
git clone https://github.com/aravinddddddd/polyagent
cd polyagent
pip install -e ".[dev]"
```

The core package and the OpenAI-compatible and Ollama adapters depend on nothing
outside the standard library. Only the Claude adapter needs a third-party
package (`pip install -e ".[anthropic]"`).

## Run it

No API key, fully offline, using Ollama:

```bash
ollama serve && ollama pull llama3.1
polyagent --provider ollama "what is 19 * 47, and what files are in this folder?"
```

Against Claude:

```bash
export ANTHROPIC_API_KEY=sk-...
polyagent --provider anthropic:claude-opus-5 "summarise README.md in one line"
```

```
[anthropic:claude-opus-5] 3 tools
  -> read_file({'path': 'README.md'})
A provider-agnostic agent loop with adapters for Claude, OpenAI-compatible APIs, and Ollama.
[in 1204 / out 88 tokens]
```

Useful flags: `--approve` confirms each tool call before it runs, `--allow-write`
adds `write_file` (off by default), `--root` bounds which directory the file
tools may touch, and `--max-steps` caps the loop.

## Library use

A tool is a plain function. The JSON Schema comes from its type hints and its
docstring, so there is no second place to keep in sync:

```python
from polyagent import Agent, load, tool

@tool
def city_population(city: str) -> int:
    """Look up the population of a city.

    Args:
        city: name of the city
    """
    return {"pune": 7_400_000, "chennai": 11_500_000}[city.lower()]

agent = Agent(load("ollama:llama3.1"), tools=[city_population])
run = agent.run("Which is bigger, Pune or Chennai?")

print(run.output)
print(f"{len(run.steps)} steps, {run.usage.output_tokens} output tokens")
```

Stream progress instead of blocking with `agent.iter_steps(prompt)`, which
yields each turn as it completes.

### Approval gate

`approve` receives every tool call before it runs. Returning `False` tells the
model the call was declined so it can choose another path, rather than killing
the run:

```python
agent = Agent(provider, tools=tools, approve=lambda call: call.name != "write_file")
```

## Design notes

- **Failures reach the model, not the stack trace.** A tool that raises, or a
  name the model invented, comes back as a tool result with `is_error=True`.
  Models reliably recover from that; they cannot recover from a traceback.
- **Retries are typed, not blanket.** Only errors marked retryable — 429, 5xx,
  connection failures — are retried, with exponential backoff and jitter. A 400
  fails the same way every time, so retrying it just triples the latency of a
  broken request.
- **Parallel tool results go back in one message.** Splitting them across turns
  teaches the model to stop making parallel calls.
- **The loop is bounded.** `max_steps` defaults to 10; an unbounded loop against
  a metered API is a bug.
- **Arithmetic is parsed, not `eval`'d.** `calculate` walks an AST and rejects
  anything that is not arithmetic, because `eval` on model output is arbitrary
  code execution.
- **File tools are sandboxed.** Paths are resolved and checked against the root,
  so `../../etc/passwd` and symlinks pointing outside are both rejected.

## Tests

```bash
pytest --cov=polyagent
```

96 tests, 94% line coverage, no network access and no API key required — every
provider is exercised through an injected transport or a stub client. CI runs
lint and the suite on Python 3.10, 3.11, and 3.12.

The tests worth reading first are in `tests/test_providers.py`: they pin the
wire-format differences in the table above, including that Claude's `thinking`
blocks survive a round trip and that OpenAI's JSON-string arguments decode to
the same dict Claude sends directly.

## Limitations

Known and deliberate, rather than hidden:

- **No streaming.** Responses are awaited in full. Token streaming needs a
  different abstraction per provider and is the next thing worth adding.
- **No async.** The loop is synchronous. Tool calls in one turn run in
  sequence, not concurrently.
- **No token-budget management.** Long conversations will eventually exceed the
  context window; there is no compaction or summarisation.
- **Images and other non-text content are not modelled.** `Message.content` is
  a string.
- **Ollama tool results correlate by order** because Ollama sends no call id.
  This is correct for the sequential loop here but would need revisiting for
  out-of-order execution.

## Licence

MIT.
