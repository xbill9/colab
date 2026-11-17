#!/usr/bin/env python3
"""Can a server inside a Colab runtime be reached from outside, through the runtime's proxy URL?

Creates a CPU runtime, starts `python -m http.server 8000` in it, then requests it through
`connectionInfo.url` with the `X-Colab-Runtime-Proxy-Token` header on several port-proxy paths.
The documented file download, `GET {url}/files/{path}`, is the positive check: it proves the URL
and token work, so a failure on the port paths means the proxy does not forward them.

    COLAB_QUOTA_PROJECT=<project> python3 tools/check_port_proxy.py
"""

from __future__ import annotations

import uuid

import requests

from colab_mcp import ColabMCP

mcp = ColabMCP()
cpu = next(s["key"] for s in mcp.call("list_runtime_specs")["runtimeSpecs"]
           if s["key"]["variant"] == "VARIANT_CPU" and s.get("eligible"))  # fmt: skip
runtime = mcp.call_reattaching("create_runtime", {"requestId": str(uuid.uuid4()), "runtime": {"runtimeSpec": cpu}})
name = runtime["name"]
print(f"runtime {name}")
try:
    session = mcp.call("create_session", {"parent": name, "sessionId": "portcheck"})["name"]
    setup = mcp.call(
        "execute_code",
        {
            "session": session,
            "code": """
import pathlib, subprocess, sys, time, urllib.request, importlib.util
pathlib.Path('/content/probe').mkdir(exist_ok=True)
pathlib.Path('/content/probe/hello.txt').write_text('hello from inside the runtime\\n')
srv = subprocess.Popen([sys.executable, '-m', 'http.server', '8000', '--bind', '127.0.0.1'], cwd='/content/probe')
time.sleep(2)
print('inside:', urllib.request.urlopen('http://127.0.0.1:8000/hello.txt').read().decode().strip())
print('jupyter_server_proxy installed:', importlib.util.find_spec('jupyter_server_proxy') is not None)
""",
        },
    )
    print(setup.get("stdout", "").strip(), setup.get("executionError") or "")

    info = mcp.call("get_runtime", {"name": name})["connectionInfo"]
    base, hdr = info["url"].rstrip("/"), {"X-Colab-Runtime-Proxy-Token": info["token"]}
    print(f"proxy host: {requests.utils.urlparse(base).netloc.split('.', 1)[-1]}")
    checks = [
        ("documented file download (positive check)", "/files/content/probe/hello.txt"),
        ("jupyter-server-proxy path", "/proxy/8000/hello.txt"),
        ("jupyter-server-proxy absolute path", "/proxy/absolute/8000/hello.txt"),
        ("port in the path", "/8000/hello.txt"),
    ]
    for label, path in checks:
        r = requests.get(base + path, headers=hdr, timeout=30)
        body = r.text.strip()[:60].replace("\n", " ")
        print(f"  {r.status_code}  {label:42s} {path:34s} {body!r}")
    r = requests.get(base + "/files/content/probe/hello.txt", timeout=30)
    print(f"  {r.status_code}  {'file download without the token':42s}")
finally:
    mcp.call("delete_runtime", {"name": name, "allowMissing": True})
    print(f"runtime {name} deleted")
