#!/usr/bin/env python3
"""Measure how long one `execute_code` call can stay open, and what the documented reattach returns.

Creates a CPU runtime, runs a cell that prints a timestamp every 30 s for SECONDS, times the call on
the client, repeats the identical call (same session and executionId) if the stream drops, prints
what came back, and deletes the runtime.

    COLAB_QUOTA_PROJECT=<project> python3 tools/check_long_call.py [SECONDS]
"""

from __future__ import annotations

import sys
import time
import uuid

import requests

from colab_mcp import ColabMCP

SECONDS = int(sys.argv[1]) if len(sys.argv) > 1 else 420
CODE = f"""
import time
t0 = time.time()
for i in range({SECONDS} // 30):
    time.sleep(30)
    print(f"{{time.time() - t0:.0f}} s", flush=True)
"done"
"""

mcp = ColabMCP()
cpu = next(s["key"] for s in mcp.call("list_runtime_specs")["runtimeSpecs"]
           if s["key"]["variant"] == "VARIANT_CPU" and s.get("eligible"))  # fmt: skip
runtime = mcp.call_reattaching("create_runtime", {"requestId": str(uuid.uuid4()), "runtime": {"runtimeSpec": cpu}})
name = runtime["name"]
print(f"runtime {name} ({cpu['accelerator']}, {cpu['shape']})", flush=True)
try:
    session = mcp.call("create_session", {"parent": name, "sessionId": "longcall"})["name"]
    args = {"session": session, "code": CODE, "executionId": str(uuid.uuid4())}
    for attempt in (1, 2):
        t0 = time.perf_counter()
        try:
            res = mcp.call("execute_code", args)
            print(f"call {attempt}: returned after {time.perf_counter() - t0:.0f} s")
            print(f"  stdout: {res.get('stdout', '').split()}")
            print(f"  result: {res.get('result')!r}  error: {res.get('executionError', {}).get('name')}")
            break
        except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError) as e:
            print(f"call {attempt}: stream dropped after {time.perf_counter() - t0:.0f} s ({type(e).__name__})")
        except Exception as e:
            print(f"call {attempt}: {type(e).__name__} after {time.perf_counter() - t0:.0f} s: {str(e)[:300]}")
finally:
    mcp.call("delete_runtime", {"name": name, "allowMissing": True})
    print(f"runtime {name} deleted")
