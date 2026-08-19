"""Minimal JSON-over-HTTP helper.

Uses the standard library so the OpenAI-compatible and Ollama adapters add no
dependencies — the package installs and its tests run with nothing but Python.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from ..errors import ProviderError

#: Statuses worth retrying: rate limits, request timeouts, and server faults.
RETRYABLE_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise ProviderError(
            f"HTTP {exc.code} from {url}: {detail}",
            status_code=exc.code,
            retryable=exc.code in RETRYABLE_STATUSES,
        ) from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"could not reach {url}: {exc.reason}", retryable=True) from exc
    except json.JSONDecodeError as exc:
        raise ProviderError(f"{url} returned a non-JSON body") from exc
