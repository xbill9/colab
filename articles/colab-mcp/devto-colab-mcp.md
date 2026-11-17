---
title: "The Colab MCP Endpoint From Claude Code: Set It Up, Test It, Run a Notebook on a TPU"
published: false
series: Colab
description: "Google's hosted Colab MCP endpoint, step by step: a token with the colaboratory scope, a quota project, registration in Claude Code with a headers helper, and a Python runner that executes a notebook on a fresh TPU v5e-1 and deletes it."
tags: colab, mcp, tpu, claude
cover_image: https://raw.githubusercontent.com/xbill9/colab/main/articles/colab-mcp/devto-cover.7a27a6a0.jpg
---

This article provides a step by step guide to Google's new Colab API and hosted MCP endpoint: getting a credential it accepts, registering it in Claude Code, testing it, and building a tool that runs a notebook on a fresh Colab TPU v5e-1 through it and deletes the runtime afterwards.

https://github.com/xbill9/colab

**A fresh Colab TPU v5e-1 was ready 4 seconds after `create_runtime`, and a 26-cell notebook ran clean on it through `execute_code` in 901 seconds, with the runtime deleted at the end.** The notebook serves Gemma 4 E2B on the TPU. The same notebook run through the Colab CLI gives the same answers:

| Gemma 4 E2B serving notebook, Colab v5e-1, 2026-11-17 | Colab CLI | MCP endpoint |
| :--- | ---: | ---: |
| Server and chat page up, seconds after the first cell | 117 | 123 |
| Thinking off / on, completion tokens | 391 / 857 | 391 / 857 |
| Greedy decode, 256-token request, tok/s | 124.6 | 122.0 |
| llama.cpp on the host CPU, tok/s | 4.0 | 4.3 |

---

#### What Is the Colab API?

Google [announced](https://medium.com/google-colab/the-google-colab-api-and-mcp-endpoint-for-developers-and-agents-65cfef497056) the Colab API and a hosted MCP endpoint for creating, inspecting and deleting Colab runtimes from code. The REST API is `v1beta`, and its discovery document lists six methods:

```text
rootUrl https://colaboratory.googleapis.com/ version v1beta revision 20261116
scope https://www.googleapis.com/auth/colaboratory - Read, write, and delete your Colab resources
scope https://www.googleapis.com/auth/colaboratory.readonly - Read-only access to your Colab resources
method GET v1beta/{+name} -> v1beta.getSubscription
method GET v1beta/runtimespecs -> runtimespecs.list
method GET v1beta/{+name} -> runtimes.get
method DELETE v1beta/{+name} -> runtimes.delete
method GET v1beta/runtimes -> runtimes.list
method POST v1beta/runtimes -> runtimes.create
ConnectionInfo fields: ['url', 'token', 'expireTime']
```

A runtime is a Jupyter server. `ConnectionInfo` carries its URL and a short-lived proxy token. The MCP endpoint, `https://colaboratory.googleapis.com/mcp`, wraps those methods as tools and adds sessions and code execution.

---

#### At This Point You Should Have…

- A Google account with Colab access.
- A Google Cloud project you can enable an API in.
- Python 3 with `google-auth` and `requests`, and `pip install google-colab-cli` for the login in Step 2.
- Claude Code, for Steps 5 and 6.

---

#### Step 1 — List the Tools

`tools/list` answers without a token, so it is the quickest check that the endpoint is reachable:

```shell
curl -s -X POST https://colaboratory.googleapis.com/mcp \
  -H 'content-type: application/json' -H 'accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

```text
create_runtime       Creates a Colab runtime and waits until it's ready.
create_session       Opens a Jupyter kernel session on a runtime (`runtimes/{runtime_id}`) and returns its name (`runtimes/{runtime_id}/sessions/{session_id}`).
delete_session       Closes a session (`runtimes/{runtime_id}/sessions/{session_id}`), discarding its kernel state and aborting any `execute_code` call running in it. Files it wrote stay on the runtime, and the runtime keeps running.
execute_code         Runs Python in a Colab runtime's Jupyter kernel. Returns stdout, stderr, the value of the last expression (`result`), and any exception (`executionError`). Only text is returned, not images or plots. Output over the size limit is truncated and flagged with `outputTruncated`.
get_subscription     Gets the user's Colab subscription tier: free, Pro or Pro+. https://colab.research.google.com/signup compares the tiers.
list_sessions        Lists the open sessions on a runtime (`runtimes/{runtime_id}`), including ones you didn't open, such as the user's notebooks. Use it to find a session of yours to reuse or close. Close only sessions you opened. It returns names only, not what is running in them.
delete_runtime       Deletes a runtime (`runtimes/{runtime_id}`) and everything on it: files, sessions and their state. This can't be undone.
get_runtime          Gets a runtime (`runtimes/{runtime_id}`), including its spec and `connectionInfo`.
list_runtimes        Lists the user's Colab runtimes, with their specs and `connectionInfo`. Check here for a runtime to reuse before calling `create_runtime`.
list_runtime_specs   Lists the runtime specs (variant, accelerator and shape) and whether the user is eligible for each. Call it before `create_runtime` to pick a spec.
10 tools
```

The descriptions are written for an agent: they say when to reuse a runtime, when to retry, and which calls cannot be undone.

---

#### Step 2 — Get a Token With the `colaboratory` Scope

Every tool call needs an access token carrying `https://www.googleapis.com/auth/colaboratory`. A plain `gcloud auth application-default login` token lacks it:

```text
token scopes: email https://www.googleapis.com/auth/cloud-platform https://www.googleapis.com/auth/sqlservice.login https://www.googleapis.com/auth/userinfo.email openid
403 PERMISSION_DENIED Request had insufficient authentication scopes. ACCESS_TOKEN_SCOPE_INSUFFICIENT
```

The [Colab CLI](https://github.com/googlecolab/google-colab-cli)'s own login asks for the scope. Sign in once:

```shell
pip install google-colab-cli
colab --auth oauth2 sessions
```

It saves a refreshable token in `~/.config/colab-cli/token.json`. `tools/colab_token.py` refreshes it in memory and prints a fresh access token, never rewriting the file:

```python
def access_token() -> str:
    creds = Credentials.from_authorized_user_file(str(TOKEN_FILE))
    if SCOPE not in (creds.scopes or []):
        sys.exit(f"{TOKEN_FILE} lacks the {SCOPE} scope; run: colab --auth oauth2 sessions")
    creds.refresh(google.auth.transport.requests.Request())
    return creds.token
```

---

#### 🔎 Tip: Read a Token's Scopes Before Debugging a 403

Google's `tokeninfo` endpoint lists the scopes a token carries, which tells a scope problem apart from a permission problem in one call:

```shell
curl -s "https://oauth2.googleapis.com/tokeninfo?access_token=$(python3 tools/colab_token.py)"
```

Asking `google.auth.default()` for the `colaboratory` scope does not add it: an ADC refresh token returns the scopes it was granted at login.

---

#### Step 3 — Name a Quota Project

With the right scope, the next call names the second requirement:

```text
ToolError: HTTP 403: {"id":1,"jsonrpc":"2.0","result":{"content":[{"text":"Your application is authenticating by using local Application Default Credentials. The colaboratory.googleapis.com API requires a quota project, which is not set by default. To learn how to set your quota project, see https://cloud.google.com/docs/authentication/adc-troubleshooting/user-creds .","type":"text"}],"isError":tr
```

Enable the API in a project you own and send that project as `x-goog-user-project`:

```shell
gcloud services enable colaboratory.googleapis.com --project <your-project>
export COLAB_QUOTA_PROJECT=<your-project>
```

`colab_token.py --header` prints both headers as JSON:

```python
def headers() -> dict:
    """Authorization plus, when COLAB_QUOTA_PROJECT is set, the quota project the endpoint requires."""
    h = {"Authorization": f"Bearer {access_token()}"}
    if QUOTA_PROJECT:
        h["x-goog-user-project"] = QUOTA_PROJECT
    return h
```

---

#### Step 4 — Test It: Your Tier and the Runtimes You Can Create

`tools/colab_mcp.py` is a small JSON-RPC client for the endpoint. `make specs` calls `get_subscription` and `list_runtime_specs`:

```shell
COLAB_QUOTA_PROJECT=<your-project> make specs
```

```text
subscription: SUBSCRIPTION_TIER_FREE
  VARIANT_CPU    NONE       SHAPE_HIGHMEM   not eligible
  VARIANT_CPU    NONE       SHAPE_STANDARD  eligible
  VARIANT_GPU    A100       SHAPE_HIGHMEM   not eligible
  VARIANT_GPU    T4         SHAPE_HIGHMEM   not eligible
  VARIANT_GPU    A100       SHAPE_STANDARD  eligible
  VARIANT_GPU    G4         SHAPE_STANDARD  eligible
  VARIANT_GPU    H100       SHAPE_STANDARD  not eligible
  VARIANT_GPU    L4         SHAPE_STANDARD  eligible
  VARIANT_GPU    T4         SHAPE_STANDARD  eligible
  VARIANT_TPU    V5E1       SHAPE_STANDARD  eligible
  VARIANT_TPU    V6E1       SHAPE_STANDARD  eligible
```

Eligibility says the account may ask for a spec. Capacity is decided at `create_runtime`, which returns `FAILED_PRECONDITION` when there is none.

---

#### Step 5 — Register the Endpoint in Claude Code

Claude Code's built-in OAuth for remote MCP servers registers a client on the fly, and Google's authorization server does not support that:

```text
plugin:google-cloud-developer:developer-knowledge: https://developerknowledge.googleapis.com/mcp (HTTP) - ✘ Failed to connect — Incompatible auth server: does not support dynamic client registration
```

A `headersHelper` sidesteps it: Claude Code runs a command when it connects and sends the JSON it prints as request headers. `colab_token.py --header` is that command:

```shell
claude mcp add-json colab --scope user '{"type": "http",
  "url": "https://colaboratory.googleapis.com/mcp",
  "headersHelper": "COLAB_QUOTA_PROJECT=<your-project> python3 /path/to/colab/tools/colab_token.py --header",
  "timeout": 700000}'
claude mcp get colab
```

```text
colab:
  Scope: User config (available in all your projects)
  Status: ✔ Connected
  Type: http
  URL: https://colaboratory.googleapis.com/mcp
  Timeout: 700000ms
```

The tool description says an `execute_code` call is held open for up to 600 seconds, so the timeout is set above that. Step 8's tip has the measured limit.

---

#### Step 6 — Ask Claude Code

A new session picks the tools up. With only the read-only tools allowed:

```shell
claude -p "Using the colab MCP server, tell me my Colab subscription tier and which TPU runtime specs I am eligible for. Call the tools; do not guess." \
  --allowedTools mcp__colab__get_subscription mcp__colab__list_runtime_specs mcp__colab__list_runtimes
```

```text
**Subscription tier:** Free (`SUBSCRIPTION_TIER_FREE`).

**TPU runtime specs:** the list has two, and you're eligible for both.

| Variant | Accelerator | Shape | Eligible |
|---|---|---|---|
| TPU | V5E1 (v5e, 1 chip) | Standard | ✅ |
| TPU | V6E1 (v6e, 1 chip) | Standard | ✅ |
```

`create_runtime`, `execute_code` and `delete_runtime` are left out of `--allowedTools`, so Claude Code asks before it spends compute or deletes anything.

---

#### Step 7 — Build: A Client That Survives Long Calls

The endpoint answers a call with plain JSON or with a server-sent event stream, and the client reads the last data event that carries a result:

```python
if r.headers.get("content-type", "").startswith("text/event-stream"):
    for line in r.text.splitlines():  # the last data event carries the result
        if line.startswith("data:"):
            event = json.loads(line[5:])
            if "result" in event or "error" in event:
                msg = event
```

Two tools document a retry that reattaches instead of repeating work: `create_runtime` with the same `requestId`, and `execute_code` with the same `session` and `executionId`, after `DEADLINE_EXCEEDED` or `UNAVAILABLE`. The client repeats the identical call in those cases only:

```python
REATTACH = ("DEADLINE_EXCEEDED", "UNAVAILABLE")

def call_reattaching(self, name: str, arguments: dict, attempts: int = 20) -> dict:
    """Repeat an identical call while the endpoint says the work is still in progress."""
    for attempt in range(1, attempts + 1):
        try:
            return self.call(name, arguments)
        except ToolError as e:
            if not e.reattach or attempt == attempts:
                raise
```

---

#### Step 8 — Build: Run a Notebook on a Fresh v5e-1

`tools/run_on_colab.py` runs a notebook through the endpoint: pick an eligible spec, create the runtime, open one session, send each code cell to `execute_code`, record the outputs, and delete the runtime in a `finally`:

```python
runtime = mcp.call_reattaching("create_runtime", {"requestId": str(uuid.uuid4()), "runtime": {"runtimeSpec": key}})
name = runtime["name"]
try:
    session = mcp.call("create_session", {"parent": name, "sessionId": "verify"})["name"]
    for cell in cells:
        res = mcp.call_reattaching(
            "execute_code", {"session": session, "code": cell.source, "executionId": str(uuid.uuid4())}
        )
        cell.outputs = outputs_of(res)
finally:
    mcp.call("delete_runtime", {"name": name, "allowMissing": True})
```

The notebook is the Gemma 4 E2B serving demo from the same repository: a 6.56 GB download, a JAX engine, an OpenAI-compatible server, samples and a llama.cpp comparison. `make verify` runs it:

```shell
COLAB_QUOTA_PROJECT=<your-project> make verify
```

```text
=== gemma4_e2b_serving_v5e1 on {'accelerator': 'V5E1', 'shape': 'SHAPE_STANDARD', 'variant': 'VARIANT_TPU'} ===
  runtime runtimes/r-508e423c-84f3-4398-9cc3-44216293a093 ready in 4 s
  [1/26] # Cell 1.1: start the checkpoint download in the background
        1 s
  [4/26] # Cell 2.1: the engine, pinned, and the checkpoint on the chip
        85 s
  [9/26] # Cell 3.1: the client and the measuring helper
        177 s
  [17/26] # Cell 3.9: first-token time against prompt length
        159 s
  [23/26] # Cell 5.3: thinking on (about four minutes at CPU speed)
        211 s
  [26/26] # Cell 7.1: stop the servers and free the chip
        2 s
  runtime runtimes/r-508e423c-84f3-4398-9cc3-44216293a093 deleted
  total 901 s from create_runtime to delete_runtime
OK: gemma4_e2b_serving_v5e1 ran clean on v5e1 -> runs/2026-11-17/gemma4_e2b_serving_v5e1_mcp_output.ipynb
```

The excerpt keeps the longest cells; the full log is in the repository's evidence folder. The executed notebook is written back with each cell's stdout, stderr and errors, and scanned for errors and unexpected stderr.

---

#### 🔎 Tip: Keep Every Cell Under 300 Seconds

The `execute_code` description gives a 600-second cutoff after which the code keeps running and the identical call reattaches. Measured from this client on a CPU runtime, one call printing a line every 30 seconds:

```shell
COLAB_QUOTA_PROJECT=<your-project> python3 tools/check_long_call.py 420
```

```text
call 1: stream dropped after 300 s (ChunkedEncodingError)
call 2: returned after 2 s
  stdout: ['30', 's', '60', 's', '90', 's', '120', 's', '150', 's', '180', 's', '210', 's', '240', 's', '270', 's', '300', 's']
  result: None  error: KeyboardInterrupt
```

The stream closed at 300 seconds, the kernel was interrupted at that moment, and the reattach returned the interrupted result. A notebook driven through the endpoint needs every cell to finish inside 300 seconds. The demo notebook's CPU comparison runs as three cells for this reason, the longest about four minutes, and `run_on_colab.py` names the cause when a cell hits the limit.

---

#### 🔎 Tip: One Session Holds the Notebook's State

Every `execute_code` call that names the same session shares one kernel, so variables, imports and background threads from one cell are there for the next, as in Colab's own notebook view. A server a cell starts in a thread keeps running between calls, and a later cell reaches it on `127.0.0.1`. Output comes back as text only: a cell that displays HTML, such as the notebook's chat page, returns nothing for it.

---

#### Step 9 — Clean Up

`delete_runtime` runs in the `finally`, so an exception or Ctrl-C between creation and teardown still releases the chip. `list_runtimes` confirms it:

```shell
COLAB_QUOTA_PROJECT=<your-project> python3 tools/colab_mcp.py call list_runtimes
```

```text
{}
```

An empty object means the account holds no runtimes.

---

#### Compare and Contrast

| | Colab CLI (`colab exec`) | Colab MCP endpoint |
| :--- | :--- | :--- |
| Runs a notebook | whole file, one command | one `execute_code` per cell |
| Long cells | per-cell `--timeout` | 300 s per call, measured; split longer cells |
| Auth | CLI login or ADC | token with `colaboratory` scope plus a quota project |
| Finds every runtime you own | sessions it created | `list_runtimes` 🥇 |
| An agent can call it | through a shell | as MCP tools 🥇 |

---

#### So, Which One?

For an agent, use the MCP endpoint: the tools are described for agents, a runtime is one call away, and `list_runtimes` and `delete_runtime` reach every runtime the account owns.

For a one-off script that runs a whole notebook file, the Colab CLI is one command. The CLI's login is also the quickest way to get a token with the `colaboratory` scope.

---

#### Summary

The goal of this article was to set up Google's Colab MCP endpoint, test it from Claude Code, and build a tool that runs a notebook on a hosted TPU through it. The key to the solution was a token with the `colaboratory` scope from the Colab CLI's login, a quota project with the API enabled, and a `headersHelper` that hands both to Claude Code. The results were:

- 🟢 `tools/list` answers without a token; every tool call needs the `colaboratory` scope and a quota project
- 🟢 The Colab CLI's login supplies the scope, and a `headersHelper` hands the token to Claude Code
- 🟢 Claude Code lists the tier and the eligible TPU specs from a plain-language prompt
- 🟢 A v5e-1 was ready 4 seconds after `create_runtime`
- 🟢 A 26-cell Gemma 4 notebook ran clean through `execute_code` in 901 seconds, with the same answers as the Colab CLI run
- 🟢 `delete_runtime` in a `finally` left no runtimes behind
- ⚠️ An `execute_code` stream closes at 300 seconds and interrupts the kernel, so cells must stay under it
- ⚠️ `execute_code` returns text only; HTML output such as a chat page does not come back
- ❌ A `gcloud auth application-default login` token is refused for lack of the scope

Scope: one Google account on the free tier, quota project in one Google Cloud project, Claude Code 2.1.326, Colab API `v1beta` revision 20261116, one notebook run through the endpoint and one through the Colab CLI on 2026-11-17, the 300-second measurement on one CPU runtime. Runtime capacity and the endpoint's rate limits were not tested.

The strategy for using MCP for hosted Colab runtimes was validated with an incremental step by step approach.

---

#### References

- Repository: https://github.com/xbill9/colab
- Announcement: https://medium.com/google-colab/the-google-colab-api-and-mcp-endpoint-for-developers-and-agents-65cfef497056
- MCP reference: https://developers.google.com/colab/api/reference/mcp
- REST reference: https://developers.google.com/colab/api/reference/rest
- Colab CLI: https://github.com/googlecolab/google-colab-cli
- The notebook it runs: https://github.com/xbill9/colab/blob/main/notebooks/gemma4_e2b_advanced_v5e1.ipynb
