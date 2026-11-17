#!/usr/bin/env python3
"""Run a built notebook on a fresh Colab runtime through the Colab MCP endpoint, and fail on any error.

This is the same path an agent uses in the demo: `list_runtime_specs`, `create_runtime`,
`create_session`, one `execute_code` per code cell, then `delete_runtime`. The executed copy is
written to `runs/<date>/<stem>_mcp_output.ipynb` and scanned the same way as `verify_on_tpu.py`, which
drives the older Colab CLI instead.

    python3 tools/run_on_colab.py                          # every notebook in notebooks.json
    python3 tools/run_on_colab.py gemma4_e2b_serving_v5e1
    python3 tools/run_on_colab.py --specs                  # list runtime specs and eligibility, then stop
    python3 tools/run_on_colab.py --keep ...               # leave the runtime up afterwards

A runtime spends compute units while it exists. The runtime is deleted in a `finally`, so an error
or Ctrl-C between creation and teardown still releases it; `--keep` is the only way to leave one up.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys
import time
import uuid

import nbformat

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from colab_mcp import ColabMCP, ToolError  # noqa: E402
from verify_on_tpu import scan  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks"
RUNS = ROOT / "runs"
CONFIG = ROOT / "notebooks.json"

# MEASURED 2026-11-17 with tools/check_long_call.py on a CPU runtime: an execute_code call's stream
# closes after 300 s (the tool description says 600), the kernel is interrupted at that moment, and
# the documented reattach (same session and executionId) returns KeyboardInterrupt. So every cell
# must finish inside 300 s; a notebook with a longer cell has to be split.
CALL_LIMIT_S = 300


def spec_matches(key: dict, want: str) -> bool:
    """`want` is notebooks.json's form, e.g. "v5e1"; the API spells accelerators its own way."""
    acc = "".join(ch for ch in key.get("accelerator", "").lower() if ch.isalnum())
    return acc == want.lower() or acc.replace("tpu", "") == want.lower()


def pick_spec(mcp: ColabMCP, want: str) -> dict:
    specs = mcp.call("list_runtime_specs").get("runtimeSpecs", [])
    for s in specs:
        if spec_matches(s["key"], want) and s.get("eligible"):
            return s["key"]
    have = ", ".join(f"{s['key'].get('accelerator')}{'' if s.get('eligible') else ' (not eligible)'}" for s in specs)
    raise SystemExit(f"no eligible runtime spec for {want!r}; specs: {have}")


def outputs_of(res: dict) -> list:
    outs = []
    for name in ("stdout", "stderr"):
        if res.get(name):
            outs.append(nbformat.v4.new_output("stream", name=name, text=res[name]))
    if res.get("result"):
        outs.append(nbformat.v4.new_output("execute_result", data={"text/plain": res["result"]}, execution_count=None))
    if err := res.get("executionError"):
        outs.append(
            nbformat.v4.new_output(
                "error", ename=err.get("name", ""), evalue=err.get("value", ""), traceback=err.get("traceback", [])
            )
        )
    if res.get("outputTruncated"):
        outs.append(nbformat.v4.new_output("stream", name="stdout", text="[output truncated by the endpoint]\n"))
    return outs


def run(stem: str, tpu: str, keep: bool) -> bool:
    notebook = OUT / f"{stem}.ipynb"
    if not notebook.exists():
        print(f"not built: {notebook} -- run `make build` first", file=sys.stderr)
        return False
    nb = nbformat.read(notebook, as_version=4)
    mcp = ColabMCP()
    key = pick_spec(mcp, tpu)
    print(f"\n=== {stem} on {key} ===", flush=True)

    t_start = t0 = time.perf_counter()
    runtime = mcp.call_reattaching("create_runtime", {"requestId": str(uuid.uuid4()), "runtime": {"runtimeSpec": key}})
    name = runtime["name"]
    print(f"  runtime {name} ready in {time.perf_counter() - t0:.0f} s", flush=True)
    failed_at, crash = None, None
    try:
        session = mcp.call("create_session", {"parent": name, "sessionId": "verify"})["name"]
        cells = [c for c in nb.cells if c.cell_type == "code"]
        for i, cell in enumerate(cells, 1):
            first = cell.source.splitlines()[0] if cell.source else ""
            print(f"  [{i}/{len(cells)}] {first[:70]}", flush=True)
            t = time.perf_counter()
            res = mcp.call_reattaching(
                "execute_code", {"session": session, "code": cell.source, "executionId": str(uuid.uuid4())}
            )
            cell.outputs = outputs_of(res)
            cell.execution_count = res.get("executionCount")
            print(f"        {time.perf_counter() - t:.0f} s", flush=True)
            if res.get("executionError"):
                failed_at = i
                print(f"        {res['executionError'].get('name')}: {res['executionError'].get('value')}", flush=True)
                if res["executionError"].get("name") == "KeyboardInterrupt" and time.perf_counter() - t >= CALL_LIMIT_S:
                    print(f"        the call outlived the endpoint's {CALL_LIMIT_S} s stream, which interrupts the "
                          "kernel; split this cell", flush=True)  # fmt: skip
                break
    except Exception as e:  # keep the partial executed copy; the runtime is still deleted below
        crash = f"{type(e).__name__}: {e}"
        print(f"        client error: {crash}", file=sys.stderr, flush=True)
    finally:
        if keep:
            print(f"  --keep: runtime {name} left running; delete it with:\n"
                  f"      python3 tools/colab_mcp.py call delete_runtime '{{\"name\": \"{name}\"}}'")  # fmt: skip
        else:
            try:
                mcp.call("delete_runtime", {"name": name, "allowMissing": True})
                print(f"  runtime {name} deleted")
            except ToolError as e:
                print(f"  could not delete {name}: {e}\n  delete it at colab.research.google.com", file=sys.stderr)

    archive = RUNS / datetime.date.today().isoformat()
    archive.mkdir(parents=True, exist_ok=True)
    executed = archive / f"{stem}_mcp_output.ipynb"
    nbformat.write(nb, executed)
    print(f"  total {time.perf_counter() - t_start:.0f} s from create_runtime to delete_runtime", flush=True)
    problems = scan(executed)
    if crash:
        problems.append(f"client error: {crash}")
    if problems or failed_at:
        print(f"\nFAILED: {stem}", file=sys.stderr)
        for p in problems:
            print(f"\n  {p}", file=sys.stderr)
        print(f"\nexecuted copy: {executed.relative_to(ROOT)}", file=sys.stderr)
        return False
    print(f"OK: {stem} ran clean on {tpu} -> {executed.relative_to(ROOT)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stems", nargs="*")
    parser.add_argument("--tpu", help="accelerator, default from notebooks.json (e.g. v5e1)")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--specs", action="store_true", help="list runtime specs and the subscription tier, then exit")
    args = parser.parse_args()

    if args.specs:
        mcp = ColabMCP()
        print("subscription:", mcp.call("get_subscription").get("tier"))
        for s in mcp.call("list_runtime_specs").get("runtimeSpecs", []):
            k = s["key"]
            print(f"  {k.get('variant', ''):14s} {k.get('accelerator', ''):10s} {k.get('shape', ''):15s} "
                  f"{'eligible' if s.get('eligible') else 'not eligible'}")  # fmt: skip
        return 0

    cfg = json.loads(CONFIG.read_text())
    entries = {pathlib.Path(n).stem: m for n, m in cfg["notebooks"].items()}
    stems = args.stems or list(entries)
    failed = [s for s in stems if not run(s, args.tpu or entries[s].get("tpu", "v5e1"), args.keep)]
    print(f"\n{len(stems) - len(failed)}/{len(stems)} notebooks verified")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
