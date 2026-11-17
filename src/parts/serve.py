# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# ## 1. Start the download, then read the chip
#
# The first cell starts the checkpoint download in its own process and returns immediately. The
# download runs while the cells after it read the chip, so it costs no waiting time of its own.
# Every checkpoint and the engine are pinned to a fixed revision, so the bytes you run are the bytes
# this notebook was written against.

# %%
# Cell 1.1: start the checkpoint download in the background, with a live progress bar
import os
import pathlib
import subprocess
import sys
import threading
import time

from IPython.display import HTML, display

T_START = time.perf_counter()
STAGES = {}  # stage name -> seconds since T_START; the summary at the end reads this

MODEL_ID = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text"
MODEL_REV = "315aec06cff196fd6479aa2669bf9379b0d0268a"
WORK = pathlib.Path("/content" if os.path.isdir("/content") else os.path.expanduser("~/gemma4-colab"))
WORK.mkdir(parents=True, exist_ok=True)

# What to expect: the size comes from the Hub, the time from the runs this notebook was tested with.
from huggingface_hub import HfApi

files = HfApi(token=False).model_info(MODEL_ID, revision=MODEL_REV, files_metadata=True).siblings  # public: no token
TOTAL_BYTES = sum(f.size or 0 for f in files)
print(f"model:    {MODEL_ID}")
print(f"revision: {MODEL_REV[:8]}, {len(files)} files, {TOTAL_BYTES / 1e9:.2f} GB")
print("estimate: about a minute. On a Colab v5e-1 on 2026-11-17 the download finished 63 and 66 s")
print("          after this cell started. The bar below measures this runtime's own rate.")
print("          The next cells run while it downloads; there is no need to wait here.\n", flush=True)

DOWNLOAD = """
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2]))
"""
download_log = open(WORK / "download.log", "w")
download = subprocess.Popen(
    [sys.executable, "-c", DOWNLOAD, MODEL_ID, MODEL_REV],
    stdout=subprocess.PIPE,
    stderr=download_log,
    text=True,
)


def bytes_received():
    """Bytes this VM has received on its network interfaces, from /proc/net/dev."""
    total = 0
    for line in open("/proc/net/dev").read().splitlines()[2:]:
        name, data = line.split(":", 1)
        if name.strip() != "lo":
            total += int(data.split()[0])
    return total


def bar(frac, text):
    return HTML(
        f'<progress value="{frac * 100:.1f}" max="100" style="width:55%;height:18px"></progress>'
        f'&nbsp;<code style="font-size:13px">{text}</code>'
    )


DL = {"updates": 0}  # filled by the watcher: "dir", "seconds", "error", and how often it redrew the bar


def watch_download(handle):
    """Update the bar once a second from network bytes, then collect the download's result.

    Progress is read from bytes received rather than from files in the cache, because the
    downloader may write each file only when it is complete. Other traffic in the meantime
    (pip, git) counts too, so the bar is capped at 99% until the download process exits.
    """
    rx0, t0 = bytes_received(), time.perf_counter()
    while download.poll() is None:
        got, el = bytes_received() - rx0, time.perf_counter() - t0
        rate = got / el if el > 0 else 0
        if rate < 1e6:
            left = "measuring the rate"
        elif got < TOTAL_BYTES:
            left = f"about {(TOTAL_BYTES - got) / rate:.0f} s left"
        else:
            left = "finishing"
        handle.update(bar(min(got / TOTAL_BYTES, 0.99),
                          f"{got / 1e9:5.2f} of {TOTAL_BYTES / 1e9:.2f} GB  {rate / 1e6:5.0f} MB/s  {left}"))  # fmt: skip
        DL["updates"] += 1
        time.sleep(1)
    DL["seconds"] = time.perf_counter() - t0
    out = download.stdout.read()
    download_log.close()
    if download.returncode == 0:
        DL["dir"] = pathlib.Path(out.strip().splitlines()[-1])
        handle.update(bar(1.0, f"downloaded {TOTAL_BYTES / 1e9:.2f} GB in {DL['seconds']:.0f} s"))
    else:
        DL["error"] = (WORK / "download.log").read_text()[-2000:]
        handle.update(bar(0, "download failed: see download.log"))


progress = display(bar(0, "starting"), display_id=True)
DOWNLOAD_THREAD = threading.Thread(target=watch_download, args=(progress,), daemon=True)
DOWNLOAD_THREAD.start()

# %%
# Cell 1.2: the chip, the host and the software versions
import importlib.metadata as md
import platform
import shutil

import jax
import jax.numpy as jnp


def version(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "not installed"


devices = jax.devices()
dev = devices[0]
if dev.platform != "tpu":
    raise RuntimeError("This notebook needs a TPU runtime: Runtime > Change runtime type > v5e-1 TPU")

mem = dev.memory_stats()
HBM_LIMIT = mem["bytes_limit"]
host_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
cpu_model = next(
    (line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name")), "unknown"
)

print("software")
print(f"  Python        {platform.python_version()}")
for pkg in ("jax", "jaxlib", "libtpu", "transformers", "huggingface_hub"):
    print(f"  {pkg:13s} {version(pkg)}")
print("\nTPU")
print(f"  devices       {len(devices)} x {dev.device_kind}")
print(f"  HBM           {HBM_LIMIT / 1e9:.2f} GB, {mem['bytes_in_use'] / 1e9:.2f} GB in use")
for attr in ("coords", "core_on_chip"):
    if hasattr(dev, attr):
        print(f"  {attr:13s} {getattr(dev, attr)}")
for key in sorted(k for k in os.environ if k.startswith(("TPU_", "COLAB_TPU"))):
    print(f"  ${key} = {os.environ[key]}")
print("\nhost")
print(f"  CPU           {os.cpu_count()} x {cpu_model}")
print(f"  RAM           {host_ram / 1e9:.1f} GB")
print(f"  free disk     {shutil.disk_usage(WORK).free / 1e9:.0f} GB")
STAGES["chip info"] = time.perf_counter() - T_START

# %% [markdown]
# **What you should see.** One device of kind `TPU v5 lite` with about 16.9 GB of HBM. The host CPU
# count matters again in section 5, where llama.cpp runs on it.

# %%
# Cell 1.3: what the chip does on two simple kernels: a bf16 matmul and a large copy
import timeit


def bench(fn, *args, reps=20):
    jax.block_until_ready(fn(*args))  # compile and warm up
    t = timeit.timeit(lambda: jax.block_until_ready(fn(*args)), number=reps)
    return t / reps


N = 8192
a = jnp.ones((N, N), jnp.bfloat16)
b = jnp.ones((N, N), jnp.bfloat16)
t_mm = bench(jax.jit(jnp.matmul), a, b)
print(f"bf16 matmul {N}x{N}: {t_mm * 1e3:.2f} ms  ->  {2 * N**3 / t_mm / 1e12:.0f} TFLOP/s")

x = jnp.ones((1024, 1024, 512), jnp.bfloat16)  # 1 GiB
t_cp = bench(jax.jit(lambda v: v + 1), x)
moved = 2 * x.size * x.dtype.itemsize  # read once, write once
print(f"1 GiB elementwise:     {t_cp * 1e3:.2f} ms  ->  {moved / t_cp / 1e9:.0f} GB/s of HBM traffic")
del a, b, x

# %% [markdown]
# **What these numbers mean.** The matmul figure is the chip's dense bf16 compute; the copy figure is
# its memory bandwidth. Compare both with Google's published
# [v5e specification](https://cloud.google.com/tpu/docs/v5e). Decoding one token at a time is
# bandwidth-bound: every token reads all the weights from HBM, which is why a 4-bit checkpoint
# decodes faster than a 16-bit one of the same model.

# %% [markdown]
# ## 2. Load and serve
#
# Three steps, timed separately:
#
# 1. **Engine.** Clone `tpu-jax` at a fixed commit, wait for the download, and load the weights onto
#    the chip. `DEQUANT_AT_LOAD` picks how: `True` unpacks the 4-bit weights to bf16 once, at load;
#    `False` keeps them packed and unpacks them at every step. Both give the same weights and the same
#    tokens. Measured on a Colab v5e-1 on 2026-10-21, unpacking once decoded at 131 tokens per second
#    against 103 and used 9.3 GB of HBM against 6.6 GB. The chip has room, so the default is `True`.
# 2. **One compile.** XLA compiles a program for each request shape. One short request compiles the
#    shape most chat turns use. Other shapes (long prompts, long answers) compile the first time
#    they are used, and the samples below show that cost where it happens.
# 3. **Server and chat page.** An OpenAI-compatible HTTP server over the engine, and this repo's
#    [`ui/chat_ui.py`](https://github.com/xbill9/colab/blob/main/ui/chat_ui.py), a chat page that
#    shows what is running and lets you chat in the browser.

# %%
# Cell 2.1: the engine, pinned, and the checkpoint on the chip
import importlib.util
import json

missing = [p for p in ("safetensors", "tokenizers", "openai") if importlib.util.find_spec(p) is None]
if missing:
    print(f"installing {', '.join(missing)} ...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)

TPU_JAX_COMMIT = "4b9f8e94619f1995206617e07e3c4ac62685d47b"
ENGINE_DIR = WORK / "tpu-jax"
if not (ENGINE_DIR / ".git").exists():
    print("cloning the tpu-jax engine ...", flush=True)
    subprocess.run(["git", "clone", "-q", "https://github.com/xbill9/tpu-jax", str(ENGINE_DIR)], check=True)
subprocess.run(["git", "-C", str(ENGINE_DIR), "checkout", "-q", TPU_JAX_COMMIT], check=True)
sys.path.insert(0, str(ENGINE_DIR))
from jax_engine import JaxGemmaEngine  # noqa: E402

t0 = time.perf_counter()
if DOWNLOAD_THREAD.is_alive():
    print("waiting for the checkpoint download (progress bar in cell 1.1) ...", flush=True)
DOWNLOAD_THREAD.join()
if "error" in DL:
    raise RuntimeError(f"download failed:\n{DL['error']}")
MODEL_DIR = DL["dir"]
size = sum(f.stat().st_size for f in MODEL_DIR.glob("*.safetensors"))
STAGES["download finished"] = time.perf_counter() - T_START
print(
    f"checkpoint: {size / 1e9:.2f} GB, downloaded in {DL['seconds']:.0f} s, done {STAGES['download finished']:.0f} s "
    f"after the first cell (this cell waited {time.perf_counter() - t0:.0f} s for it; the bar redrew {DL['updates']} times)"
)

from transformers import AutoTokenizer  # noqa: E402

TOK = AutoTokenizer.from_pretrained(MODEL_DIR)
GEN_CFG = json.loads((MODEL_DIR / "generation_config.json").read_text())
EOS_IDS = GEN_CFG["eos_token_id"]
# Prompt plus answer. At 8,192 the prefill of a long prompt needs more HBM than this chip has left
# beside the unpacked weights (measured on a Colab v5e-1: 16.77 GB asked, 15.75 GB free), so the
# server stops at 4,096 and refuses longer prompts with a clear error.
MAX_MODEL_LEN = 4096
DEQUANT_AT_LOAD = True  # False keeps the weights packed in HBM: less memory, slower decode

print("loading the weights onto the chip; this took 17 s on a Colab v5e-1 on 2026-11-17 ...", flush=True)
t0 = time.perf_counter()
ENGINE = JaxGemmaEngine(MODEL_ID, quant_mode="w4a16", max_model_len=MAX_MODEL_LEN, dequant_at_load=DEQUANT_AT_LOAD)
ENGINE.load(local_dir=str(MODEL_DIR))
ENGINE.bos_token_id = TOK.bos_token_id
STAGES["weights on chip"] = time.perf_counter() - T_START
print(
    f"engine: tpu-jax {TPU_JAX_COMMIT[:7]}, dequant_at_load={DEQUANT_AT_LOAD}, "
    f"{ENGINE.weight_bytes / 1e9:.2f} GB of weights on the chip, "
    f"loaded in {time.perf_counter() - t0:.0f} s"
)
print(f"HBM in use: {dev.memory_stats()['bytes_in_use'] / 1e9:.2f} of {HBM_LIMIT / 1e9:.2f} GB")

# %% [markdown]
# ### How a request becomes tokens
#
# The next cell holds everything between an OpenAI request and the engine:
#
# * **The chat template.** The checkpoint's own template turns messages into tokens. Thinking is a
#   template switch, `enable_thinking`, passed the way vLLM takes it:
#   `chat_template_kwargs: {"enable_thinking": true}`. With it on, the template puts a `<|think|>`
#   marker in the system turn and the model answers with its reasoning first.
# * **Splitting reasoning from the answer.** The model writes its reasoning between two special
#   tokens, `<|channel>` and `<channel|>`, with the word `thought` as a label. The server returns that
#   text as `reasoning_content` and the rest as `content`, the field names vLLM and llama.cpp use.
# * **Shapes and compiles.** The engine compiles one program per (prompt bucket, answer budget) pair.
#   Prompts are padded to 64, 128, 256, ... tokens. Answer budgets are rounded up to 512, 2048 or
#   4096 tokens and the stream is cut at the requested `max_tokens`, so different `max_tokens` values
#   reuse one program instead of compiling a new one each. Every response carries `x_new_shape`,
#   true when serving it compiled a new program, so a slow first token can be told apart from a slow
#   model.
# * **Stop sequences.** Text that could be the start of a stop sequence is held back until the next
#   tokens show whether it is, so a stop string never leaks out half-written.
# * **One request at a time.** This engine decodes a single sequence, so the server runs requests in
#   turn and reports the queue on `/metrics`.
#
# Supported request fields: `messages`, `max_tokens` (or `max_completion_tokens`), `temperature`,
# `top_k`, `seed`, `stop`, `stream`, `stream_options.include_usage` and
# `chat_template_kwargs.enable_thinking`. Other fields, such as `top_p`, are accepted and ignored,
# and the response says so in a `warnings` field.

# %%
# Cell 2.2: the request path: template, budget, generation, reasoning split, stop sequences
import random
import threading

CH_OPEN = TOK.convert_tokens_to_ids("<|channel>")
CH_CLOSE = TOK.convert_tokens_to_ids("<channel|>")
ANSWER_BUDGETS = (512, 2048, 4096)
# Thinking for requests that do not say. The chat page sends no `enable_thinking`, so this switch is
# how the page gets thinking: set it to True and the page shows the reasoning in a collapsible
# section above each answer. Every sample below sets thinking explicitly, so this does not affect them.
THINKING_DEFAULT = False
SUPPORTED = {
    "model", "messages", "max_tokens", "max_completion_tokens", "temperature", "top_k", "seed", "stop",
    "stream", "stream_options", "chat_template_kwargs", "n",
}  # fmt: skip


class ReasoningSplitter:
    """Turn generated token ids into (kind, text) deltas, kind being 'reasoning' or 'content'."""

    LABEL = "thought\n"

    def __init__(self):
        self.kind, self.ids, self.sent = "content", [], 0

    def feed(self, tid):
        if tid in (CH_OPEN, CH_CLOSE):
            self.kind = "reasoning" if tid == CH_OPEN else "content"
            self.ids, self.sent = [], 0
            return None
        self.ids.append(tid)
        text = TOK.decode(self.ids, skip_special_tokens=True)
        if self.kind == "reasoning":
            if self.LABEL.startswith(text):
                return None  # still inside the "thought" label
            text = text.removeprefix(self.LABEL)
        if text.endswith("�"):
            return None  # half of a multi-byte character; wait for the rest
        delta, self.sent = text[self.sent :], len(text)
        return (self.kind, delta) if delta else None


from ports.gemma4.jax_e_model import TPUv6eHardwareProfile  # noqa: E402

SEEN_SHAPES = set()  # (prompt bucket, answer budget) pairs already compiled


def prompt_ids(req):
    """The request's messages through the chat template; ValueError if they do not fit."""
    thinking = bool((req.get("chat_template_kwargs") or {}).get("enable_thinking", THINKING_DEFAULT))
    enc = TOK.apply_chat_template(req["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=thinking)
    ids = list(enc["input_ids"] if hasattr(enc, "keys") else enc)
    if len(ids) >= MAX_MODEL_LEN:
        raise ValueError(f"prompt is {len(ids)} tokens; this server's limit is {MAX_MODEL_LEN - 1}")
    return ids


def shape_of(n_prompt, budget):
    """The compiled program a request uses: its padded prompt length and its answer budget."""
    return TPUv6eHardwareProfile.get_nearest_bucket(n_prompt), min(budget, MAX_MODEL_LEN - n_prompt)


def stop_holdback(text, stops):
    """Length of the longest tail of `text` that could still grow into one of `stops`."""
    return max((k for s in stops for k in range(1, len(s)) if text.endswith(s[:k])), default=0)


def run_request(req, emit=None, ids=None):
    """Generate for one OpenAI chat request. `emit(kind, text)` receives deltas as they are produced."""
    ids = ids or prompt_ids(req)
    max_tokens = int(req.get("max_completion_tokens") or req.get("max_tokens") or 1024)
    budget = next((b for b in ANSWER_BUDGETS if b >= max_tokens), max_tokens)
    shape = shape_of(len(ids), budget)
    new_shape = shape not in SEEN_SHAPES
    temperature = req.get("temperature")
    temperature = 1.0 if temperature is None else float(temperature)
    top_k = int(req.get("top_k") or -1)
    seed = req.get("seed")
    seed = random.randrange(2**31) if seed is None else int(seed)
    stops = req.get("stop") or []
    stops = [stops] if isinstance(stops, str) else list(stops)

    split, parts, held = ReasoningSplitter(), {"reasoning": "", "content": ""}, ""
    finish, n = "length", 0

    def send(kind, text):
        parts[kind] += text
        if emit and text:
            emit(kind, text)

    stream = ENGINE.generate_stream(ids, budget, temperature, top_k, EOS_IDS, seed)
    try:
        for item in stream:
            if not isinstance(item, int):  # the GenerationStats at the end: the model stopped itself
                finish = "stop"
                break
            n += 1
            out = split.feed(item)
            if out and out[0] == "content" and stops:
                pending = held + out[1]
                full = parts["content"] + pending
                cut = min((i for i in (full.find(s) for s in stops) if i >= 0), default=-1)
                if cut >= 0:
                    send("content", full[len(parts["content"]) : cut])
                    held, finish = "", "stop"
                else:
                    keep = stop_holdback(full, stops)
                    send("content", pending[: len(pending) - keep])
                    held = pending[len(pending) - keep :]
            elif out:
                send(*out)
            if finish == "stop" or n >= max_tokens:
                break
    finally:
        stream.close()
    send("content", held)  # the held-back tail turned out not to be a stop sequence
    SEEN_SHAPES.add(shape)
    return {
        "reasoning": parts["reasoning"].strip(),
        "content": parts["content"].strip(),
        "finish_reason": finish,
        "prompt_tokens": len(ids),
        "completion_tokens": n,
        "new_shape": new_shape,
    }


# One short request compiles the shape most chat turns use (prompt bucket 64, answer budget 512).
print("compiling the first request shape; this took 24 s on a Colab v5e-1 on 2026-11-17 ...", flush=True)
# The other common chat shapes compile in the background once the server is up (cell 2.5).
t0 = time.perf_counter()
warm = run_request({"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 16, "temperature": 0})
assert warm["new_shape"]
STAGES["first compile done"] = time.perf_counter() - T_START
print(f"warm-up: {warm['content']!r} in {time.perf_counter() - t0:.0f} s, including the compile")

# %% [markdown]
# ### The server
#
# A standard-library HTTP server in a background thread. Its routes follow the OpenAI API and
# vLLM's server, so any OpenAI client works against it:
#
# | Route | Returns |
# | --- | --- |
# | `GET /health` | 200 once the model is loaded |
# | `GET /version` | the engine, its commit, and the JAX version |
# | `GET /v1/models` | the model id and `max_model_len` |
# | `GET /metrics` | Prometheus counters: requests running and waiting, prompt and generated tokens |
# | `POST /v1/chat/completions` | a chat completion, streamed as server-sent events or returned whole |
#
# The counters are named `tpu_jax:num_requests_running` and so on. The chat page reads these, and
# vLLM's names as well, so it can also be pointed at a vLLM server.

# %%
# Cell 2.3: the OpenAI-compatible server on 127.0.0.1:8000
import http.server
import uuid

PORT = 8000
LOCK = threading.Lock()
COUNTERS = {"running": 0, "waiting": 0, "requests": 0, "prompt_tokens": 0, "generation_tokens": 0}
COUNTERS_LOCK = threading.Lock()


def bump(**delta):
    with COUNTERS_LOCK:
        for k, v in delta.items():
            COUNTERS[k] += v


class OpenAIHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self._send(200, b"", "text/plain")
        elif self.path == "/version":
            self._json({"version": f"tpu-jax {TPU_JAX_COMMIT[:7]} (JAX {jax.__version__})"})
        elif self.path == "/v1/models":
            model = {
                "id": MODEL_ID, "object": "model", "created": int(T_START), "owned_by": "tpu-jax",
                "root": MODEL_ID, "max_model_len": MAX_MODEL_LEN,
            }  # fmt: skip
            self._json({"object": "list", "data": [model]})
        elif self.path == "/metrics":
            with COUNTERS_LOCK:
                c = dict(COUNTERS)
            body = "".join(
                f'tpu_jax:{name}{{model_name="{MODEL_ID}"}} {value}\n'
                for name, value in (
                    ("num_requests_running", c["running"]),
                    ("num_requests_waiting", c["waiting"]),
                    ("request_success_total", c["requests"]),
                    ("prompt_tokens_total", c["prompt_tokens"]),
                    ("generation_tokens_total", c["generation_tokens"]),
                )
            )
            self._send(200, body.encode(), "text/plain; version=0.0.4")
        else:
            self._json({"error": {"message": f"no route {self.path}"}}, 404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._json({"error": {"message": f"no route {self.path}"}}, 404)
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if int(req.get("n") or 1) != 1:
                raise ValueError("n must be 1: this engine decodes one sequence per request")
            ids = prompt_ids(req)
        except (ValueError, TypeError, KeyError) as e:
            self._json({"error": {"message": str(e), "type": "invalid_request_error"}}, 400)
            return
        ignored = sorted(set(req) - SUPPORTED)
        rid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())

        def chunk(delta, finish=None, usage=None, **extra):
            body = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID}
            body["choices"] = [] if usage else [{"index": 0, "delta": delta, "finish_reason": finish}]
            if usage:
                body["usage"] = usage
            body.update(extra)
            self.wfile.write(f"data: {json.dumps(body)}\n\n".encode())
            self.wfile.flush()

        stream = bool(req.get("stream"))
        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            chunk({"role": "assistant", "content": ""})
        emit = (
            (lambda kind, text: chunk({"reasoning_content" if kind == "reasoning" else "content": text}))
            if stream
            else None
        )

        bump(waiting=1)
        with LOCK:
            bump(waiting=-1, running=1)
            try:
                result = run_request(req, emit, ids)
            except (BrokenPipeError, ConnectionResetError):
                return  # the client hung up; closing the generator stopped decoding
            except Exception as e:  # report it to the client instead of dropping the connection
                if stream:
                    self.wfile.write(f"data: {json.dumps({'error': {'message': str(e)}})}\n\n".encode())
                else:
                    self._json({"error": {"message": str(e)}}, 500)
                return
            finally:
                bump(running=-1)
        bump(requests=1, prompt_tokens=result["prompt_tokens"], generation_tokens=result["completion_tokens"])
        usage = {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
        }
        if stream:
            try:
                chunk({}, finish=result["finish_reason"], x_new_shape=result["new_shape"])
                if (req.get("stream_options") or {}).get("include_usage"):
                    chunk(None, usage=usage)
                self.wfile.write(b"data: [DONE]\n\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        message = {"role": "assistant", "content": result["content"]}
        if result["reasoning"]:
            message["reasoning_content"] = result["reasoning"]
        body = {
            "id": rid, "object": "chat.completion", "created": created, "model": MODEL_ID,
            "choices": [{"index": 0, "message": message, "finish_reason": result["finish_reason"]}],
            "usage": usage, "x_new_shape": result["new_shape"],
        }  # fmt: skip
        if ignored:
            body["warnings"] = [f"ignored unsupported field: {k}" for k in ignored]
        self._json(body)

    def _send(self, code, data, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def log_message(self, fmt, *args):
        pass


SERVER = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), OpenAIHandler)
threading.Thread(target=SERVER.serve_forever, daemon=True).start()
BASE_URL = f"http://127.0.0.1:{PORT}/v1"
print(f"OpenAI-compatible server on {BASE_URL}")

# %%
# Cell 2.4: this repo's chat page, ui/chat_ui.py, pointed at the server above
import urllib.request

UI_DIR = WORK / "chat_ui"
UI_DIR.mkdir(exist_ok=True)
print("starting the chat page ...", flush=True)
urllib.request.urlretrieve("https://raw.githubusercontent.com/xbill9/colab/main/ui/chat_ui.py", UI_DIR / "chat_ui.py")
# The page's "what is running" panel reads these keys from a serving.env beside it.
(UI_DIR / "serving.env").write_text(
    f"MODEL_NAME={MODEL_ID}\nACCELERATOR_TYPE=Colab {dev.device_kind} x{len(devices)}\nTENSOR_PARALLEL_SIZE=1\n"
)
UI_PORT = 8765
ui = subprocess.Popen(
    [sys.executable, str(UI_DIR / "chat_ui.py"), "--upstream", f"http://127.0.0.1:{PORT}", "--port", str(UI_PORT)],
    stdout=subprocess.DEVNULL,
    stderr=open(WORK / "chat_ui.log", "w"),
)
for _ in range(50):
    try:
        info = json.load(urllib.request.urlopen(f"http://127.0.0.1:{UI_PORT}/ui/info", timeout=5))
        break
    except OSError:
        time.sleep(0.2)
else:
    raise RuntimeError(f"chat page did not start:\n{(WORK / 'chat_ui.log').read_text()[-2000:]}")

STAGES["ready to test"] = time.perf_counter() - T_START
print("what the chat page reports about this server:")
print(json.dumps(info, indent=2))
print(f"\nREADY: model served and chat page up {STAGES['ready to test']:.0f} s after the first cell started")

try:
    from google.colab import output

    output.serve_kernel_port_as_iframe(UI_PORT, height=720)
except ImportError:
    print(f"not on Colab: open http://127.0.0.1:{UI_PORT} in a browser on this machine")

# %% [markdown]
# **What you should see.** The download finishing a few minutes after the first cell, the weights
# loading onto the chip, a short compile, then the chat page. The page's side panel shows the model,
# the server version and the request counters, all read from the server above. **You can chat with
# the model now**. Requests from the page and from any later cell take turns on the chip.
#
# The page sends no `enable_thinking`, so the server's `THINKING_DEFAULT` decides. To demo
# thinking in the page, run `THINKING_DEFAULT = True` in a new cell; the next message shows its
# reasoning in a collapsible section. The page also has max tokens, temperature, top-k and a system
# prompt in its side panel, and shows the first-token time and decode rate of each reply.

# %% [markdown]
# ### Warming the other chat shapes in the background
#
# A chat's first few turns move through the 128, 256 and 512-token prompt buckets, and a thinking
# answer needs the 2048-token budget. Each would compile the first time a user hit it. The next cell
# compiles them in a background thread instead, one at a time. A chat message sent meanwhile waits
# for at most the one compile in progress.

# %%
# Cell 2.5: compile the common chat shapes in the background
WARM_SHAPES = [(128, 512), (256, 512), (64, 2048), (128, 2048), (512, 512), (256, 2048)]
WARM_LOG = []
FILLER_ID = TOK.encode("hello", add_special_tokens=False)[0]


def warm_shapes():
    for bucket, budget in WARM_SHAPES:
        ids = [TOK.bos_token_id] + [FILLER_ID] * (bucket - 1)  # exactly fills the bucket
        if shape_of(len(ids), budget) in SEEN_SHAPES:
            continue
        with LOCK:
            t0 = time.perf_counter()
            stream = ENGINE.generate_stream(ids, budget, 0.0, -1, [], 0)
            for _ in range(2):  # the first token compiles the prefill, the second the decode step
                next(stream)
            stream.close()
            SEEN_SHAPES.add(shape_of(len(ids), budget))
        WARM_LOG.append((bucket, budget, time.perf_counter() - t0))


WARM_THREAD = threading.Thread(target=warm_shapes, daemon=True)
WARM_THREAD.start()
print(f"compiling {len(WARM_SHAPES)} chat shapes in the background, about 3 minutes (178 s on 2026-11-17).")
print("The chat page stays usable meanwhile: a message waits for at most the one compile in progress.")
