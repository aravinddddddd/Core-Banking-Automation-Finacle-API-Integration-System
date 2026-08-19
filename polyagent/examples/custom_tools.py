"""Run the same agent and tools against three different backends.

python examples/custom_tools.py ollama
python examples/custom_tools.py anthropic
python examples/custom_tools.py groq:llama-3.3-70b-versatile
"""

from __future__ import annotations

import sys

from polyagent import Agent, PolyAgentError, load, tool

# A small in-memory table stands in for a database so the example runs anywhere.
ACCOUNTS = {
    "AC1001": {"holder": "R. Nair", "type": "Savings", "balance": 48250.75, "status": "Active"},
    "AC1002": {"holder": "S. Iyer", "type": "Current", "balance": 1200.00, "status": "Active"},
    "AC1003": {"holder": "M. Khan", "type": "Savings", "balance": 0.00, "status": "Dormant"},
}


@tool
def get_account(account_id: str) -> dict:
    """Look up an account by its identifier.

    Args:
        account_id: the account number, for example AC1001
    """
    account = ACCOUNTS.get(account_id.upper())
    if account is None:
        raise ValueError(f"no account {account_id!r}; known ids: {', '.join(ACCOUNTS)}")
    return account


@tool
def can_withdraw(account_id: str, amount: float) -> str:
    """Check whether an account can cover a withdrawal.

    Args:
        account_id: the account number
        amount: the amount to withdraw
    """
    account = ACCOUNTS.get(account_id.upper())
    if account is None:
        raise ValueError(f"no account {account_id!r}")
    if account["status"] != "Active":
        return f"declined: account is {account['status']}"
    if account["balance"] < amount:
        return f"declined: balance {account['balance']} is below {amount}"
    return f"approved: {account['balance'] - amount} would remain"


def main() -> int:
    spec = sys.argv[1] if len(sys.argv) > 1 else "ollama"
    question = "Can AC1003 withdraw 500? What about AC1001 withdrawing 10000?"

    try:
        agent = Agent(
            load(spec),
            tools=[get_account, can_withdraw],
            system="You are a careful banking assistant. Check every account before answering.",
        )
        run = agent.run(question)
    except PolyAgentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for call in run.tool_calls:
        print(f"  -> {call.name}({call.arguments})")
    print(f"\n{run.output}")
    print(f"\n[{len(run.steps)} steps, {run.usage.output_tokens} output tokens]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
