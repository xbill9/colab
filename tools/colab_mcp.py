#!/usr/bin/env python3
"""A minimal client for Colab's hosted MCP endpoint, https://colaboratory.googleapis.com/mcp.

JSON-RPC over HTTP: `tools/list` and `tools/call`. The endpoint may answer a call with plain JSON or
with a server-sent event stream; both are handled. The access token comes from `colab_token.py`.

    python3 tools/colab_mcp.py tools                     # tool names
    python3 tools/colab_mcp.py call list_runtime_specs   # call a tool, arguments as JSON after the name
    python3 tools/colab_mcp.py call get_runtime '{"name": "runtimes/abc"}'
"""

from __future__ import annotations

import itertools
import json
import sys

import requests

ENDPOINT = "https://colaboratory.googleapis.com/mcp"

# Errors the endpoint documents as "repeat the identical call": the work goes on server-side and
# the repeat reattaches to it (create_runtime with the same requestId, execute_code with the same
# session and executionId).
REATTACH = ("DEADLINE_EXCEEDED", "UNAVAILABLE")


class ToolError(RuntimeError):
    """A tool call that came back with isError, or a JSON-RPC error."""

    def __init__(self, text: str):
        super().__init__(text)
        self.reattach = any(code in text for code in REATTACH)


class ColabMCP:
    def __init__(self, timeout: float = 700.0):
        from colab_token import headers

        self.session = requests.Session()
        self.session.headers.update(
            {**headers(), "content-type": "application/json", "accept": "application/json, text/event-stream"}
        )
        self.timeout = timeout
        self._ids = itertools.count(1)

    def _rpc(self, method: str, params: dict | None = None) -> dict:
        body = {"jsonrpc": "2.0", "id": next(self._ids), "method": method}
        if params is not None:
            body["params"] = params
        r = self.session.post(ENDPOINT, json=body, timeout=self.timeout)
        if r.status_code >= 400:
            raise ToolError(f"HTTP {r.status_code}: {r.text[:1000]}")
        msg = None
        if r.headers.get("content-type", "").startswith("text/event-stream"):
            for line in r.text.splitlines():  # the last data event carries the result
                if line.startswith("data:"):
                    event = json.loads(line[5:])
                    if "result" in event or "error" in event:
                        msg = event
        else:
            msg = r.json()
        if msg is None:
            raise ToolError(f"no JSON-RPC result in response: {r.text[:500]}")
        if "error" in msg:
            raise ToolError(json.dumps(msg["error"]))
        return msg["result"]

    def tools(self) -> list[dict]:
        return self._rpc("tools/list")["tools"]

    def call(self, name: str, arguments: dict | None = None) -> dict:
        result = self._rpc("tools/call", {"name": name, "arguments": arguments or {}})
        text = "".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
        if result.get("isError"):
            raise ToolError(text or json.dumps(result))
        if "structuredContent" in result:
            return result["structuredContent"]
        try:
            return json.loads(text) if text else {}
        except json.JSONDecodeError:
            return {"text": text}

    def call_reattaching(self, name: str, arguments: dict, attempts: int = 20) -> dict:
        """Repeat an identical call while the endpoint says the work is still in progress."""
        for attempt in range(1, attempts + 1):
            try:
                return self.call(name, arguments)
            except ToolError as e:
                if not e.reattach or attempt == attempts:
                    raise
                print(f"  {name}: {str(e)[:80]} -- reattaching ({attempt})", file=sys.stderr, flush=True)
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError,
                    requests.exceptions.ChunkedEncodingError) as e:  # fmt: skip
                # The stream dropped before the result arrived. Repeat the identical call, as the endpoint
                # documents. MEASURED 2026-11-17: for execute_code the stream drops at 300 s and the kernel
                # is interrupted then, so the reattach returns KeyboardInterrupt rather than the result.
                if attempt == attempts:
                    raise
                print(f"  {name}: {type(e).__name__} -- reattaching ({attempt})", file=sys.stderr, flush=True)
        raise AssertionError("unreachable")


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("tools", "call"):
        print(__doc__, file=sys.stderr)
        return 2
    mcp = ColabMCP()
    if sys.argv[1] == "tools":
        for t in mcp.tools():
            print(f"{t['name']:20s} {t.get('description', '').splitlines()[0]}")
        return 0
    args = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    print(json.dumps(mcp.call(sys.argv[2], args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
