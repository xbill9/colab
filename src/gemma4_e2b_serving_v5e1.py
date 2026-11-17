# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Gemma 4 E2B on a Colab v5e-1: the QAT repack served, with thinking on and off
#
# **Author:** William McLean, Google Developer Expert
#
# This notebook serves Gemma 4 E2B from one Colab TPU v5e chip and puts it through the requests a
# real deployment sees: thinking on and off, streaming, sampling and seeds, stop sequences, token
# limits, system prompts, multi-turn chat, JSON output and long prompts. Every request goes through
# an OpenAI-compatible endpoint, so the same client code also talks to a vLLM deployment and to
# llama.cpp.
#
# ### The model
#
# [`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text)
# is an exact repack of Google's quantization-aware-trained (QAT) Gemma 4 E2B weights. Google trained
# every weight onto a 4-bit grid; the repack stores those trained 4-bit levels and their trained
# scales in the W4A16 format, with the vision and audio towers removed. It is public and ungated.
#
# | Build | Download |
# | --- | ---: |
# | Google's `google/gemma-4-E2B-it-qat-w4a16-ct` | 8.32 GB |
# | The repack, `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct` | 7.51 GB |
# | **The repack, text only (used here)** | **6.56 GB** |
# | The repack, text only, embeddings in int4 too (`-text-emb4`) | 2.83 GB |
#
# Google's export re-rounds every weight onto a new grid; the repack keeps the trained one. Served by
# vLLM on one v5e chip the two decode at the same speed (136.6 and 136.5 tokens per second) and the
# repack scores 2.4 points higher on a 3,880-record test suite. The `-text-emb4` build is the smallest,
# but the engine used here needs the embedding tables in bf16, so this notebook uses `-text`.
#
# ### The serving engine, and why this one
#
# The goal is a model you can talk to a few minutes after **Run all**. The engine is
# [`tpu-jax`](https://github.com/xbill9/tpu-jax), a Gemma 4 E2B decoder in pure JAX. It reads the
# checkpoint, puts the weights on the chip and compiles only the shapes a request uses, so the first
# answer arrives after one short compile. On a Colab v5e-1 on 2026-10-21 this engine loaded the repack
# in 7 to 16 seconds and compiled its first request in 23 to 40 seconds. vLLM on the same chip took 16 to 17 minutes from TPU grant
# to its first answer when measured on 2026-11-03 (most of it compiling), which is why vLLM appears
# here as an optional remote endpoint rather than as the local server.
#
# ### How the notebook runs
#
# 1. **Start the download, then read the chip.** The 6.6 GB checkpoint downloads in the background
#    while the first cells report on the TPU.
# 2. **Load and serve.** The engine loads, compiles one request shape, and starts an
#    OpenAI-compatible server and a browser chat page. The cell prints how long all of this took.
#    **From here you can test the model yourself.**
# 3. **Samples.** Each common serving option, called through the OpenAI client.
# 4. **A deployed endpoint (optional).** The same calls against a vLLM server you run elsewhere.
# 5. **llama.cpp on the host CPU.** The repack's exact GGUF build, for a start-up and speed
#    comparison.
# 6. **Summary** of every number measured in this run.
#
# ### Before you start
#
# * **Runtime.** Runtime, then Change runtime type, then **v5e-1 TPU**.
# * **No token.** Every download is public. Nothing asks for a Hugging Face token.
# * **Time.** Plan on 30 to 60 minutes for the whole notebook; the chat page is up within the first
#   few minutes.
# * **Safety.** Nothing is uploaded. The servers listen on `127.0.0.1` inside the Colab VM.

# %% [markdown]
# ## 1. Start the download, then read the chip
#
# The first cell starts the checkpoint download in its own process and returns immediately. The
# download runs while the cells after it read the chip, so it costs no waiting time of its own.
# Every checkpoint and the engine are pinned to a fixed revision, so the bytes you run are the bytes
# this notebook was written against.

# %%
# Cell 1.1: start the checkpoint download in the background
import os
import pathlib
import subprocess
import sys
import time

T_START = time.perf_counter()
STAGES = {}  # stage name -> seconds since T_START; the summary at the end reads this

MODEL_ID = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text"
MODEL_REV = "315aec06cff196fd6479aa2669bf9379b0d0268a"
WORK = pathlib.Path("/content" if os.path.isdir("/content") else os.path.expanduser("~/gemma4-colab"))
WORK.mkdir(parents=True, exist_ok=True)

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
print(f"downloading {MODEL_ID} @ {MODEL_REV[:8]} in the background (pid {download.pid})")

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
# 3. **Server and chat page.** An OpenAI-compatible HTTP server over the engine, and the chat page
#    from the [`tpu-vllm-v5e1-2b`](https://github.com/xbill9/tpu-inference/tree/main/tpu-vllm-v5e1-2b)
#    rig, which shows what is running and lets you chat in the browser.

# %%
# Cell 2.1: the engine, pinned, and the checkpoint on the chip
import importlib.util
import json

missing = [p for p in ("safetensors", "tokenizers", "openai") if importlib.util.find_spec(p) is None]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)

TPU_JAX_COMMIT = "4b9f8e94619f1995206617e07e3c4ac62685d47b"
ENGINE_DIR = WORK / "tpu-jax"
if not (ENGINE_DIR / ".git").exists():
    subprocess.run(["git", "clone", "-q", "https://github.com/xbill9/tpu-jax", str(ENGINE_DIR)], check=True)
subprocess.run(["git", "-C", str(ENGINE_DIR), "checkout", "-q", TPU_JAX_COMMIT], check=True)
sys.path.insert(0, str(ENGINE_DIR))
from jax_engine import JaxGemmaEngine  # noqa: E402

t0 = time.perf_counter()
out, _ = download.communicate()
download_log.close()
if download.returncode != 0:
    raise RuntimeError(f"download failed:\n{(WORK / 'download.log').read_text()[-2000:]}")
MODEL_DIR = pathlib.Path(out.strip().splitlines()[-1])
size = sum(f.stat().st_size for f in MODEL_DIR.glob("*.safetensors"))
STAGES["download finished"] = time.perf_counter() - T_START
print(
    f"checkpoint: {size / 1e9:.2f} GB, done {STAGES['download finished']:.0f} s after the first cell "
    f"(this cell waited {time.perf_counter() - t0:.0f} s for it)"
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
# The other common chat shapes compile in the background once the server is up (cell 2.5).
t0 = time.perf_counter()
warm = run_request({"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 16, "temperature": 0})
assert warm["new_shape"]
STAGES["first compile done"] = time.perf_counter() - T_START
print(f"warm-up: {warm['content']!r} in {time.perf_counter() - t0:.0f} s, including the compile")

# %% [markdown]
# ### The server
#
# A standard-library HTTP server in a background thread, shaped like vLLM's so that vLLM clients
# and the chat page work against it unchanged:
#
# | Route | Returns |
# | --- | --- |
# | `GET /health` | 200 once the model is loaded |
# | `GET /version` | the engine, its commit, and the JAX version |
# | `GET /v1/models` | the model id and `max_model_len` |
# | `GET /metrics` | Prometheus counters: requests running and waiting, prompt and generated tokens |
# | `POST /v1/chat/completions` | a chat completion, streamed as server-sent events or returned whole |
#
# The counters use vLLM's metric names (`vllm:num_requests_running` and so on) so that the chat page,
# written for vLLM, reads them without changes.

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
                f'vllm:{name}{{model_name="{MODEL_ID}"}} {value}\n'
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
# Cell 2.4: the chat page from the tpu-vllm-v5e1-2b rig, pointed at this server
import urllib.request

CHAT_UI_COMMIT = "cc99b0f5000fdee92ce622411f6bfae75d43092c"
UI_DIR = WORK / "chat_ui"
UI_DIR.mkdir(exist_ok=True)
urllib.request.urlretrieve(
    f"https://raw.githubusercontent.com/xbill9/tpu-inference/{CHAT_UI_COMMIT}/tpu-vllm-v5e1-2b/chat_ui.py",
    UI_DIR / "chat_ui.py",
)
# The page's "what is running" panel reads these keys from a tpu.env beside it.
(UI_DIR / "tpu.env").write_text(
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
# the model now**, while the rest of the notebook runs; requests take turns on the chip.
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
print(f"compiling {len(WARM_SHAPES)} chat shapes in the background; the chat page stays usable")

# %% [markdown]
# ## 3. Samples through the OpenAI client
#
# Every sample uses the official `openai` Python client against the server above, which is how an
# application would call it. The first cell waits for the background compiles to finish, so that the
# timings below are not measured while a compile holds the chip. One helper streams each request and
# measures it on the client's clock:
#
# * **First token:** time from sending the request to the first streamed text. It includes reading
#   the prompt (prefill), and, the first time a new shape is used, the compile. Those rows are
#   marked `compiled`.
# * **Decode rate:** completion tokens per second after the first token.
#
# The server reports token counts in `usage`. Numbers vary a little between runs; the ones printed
# are this run's.

# %%
# Cell 3.1: the client and the measuring helper
from openai import OpenAI

t0 = time.perf_counter()
WARM_THREAD.join()
STAGES["chat shapes compiled"] = time.perf_counter() - T_START
print(f"background compiles (waited {time.perf_counter() - t0:.0f} s for them here):")
for bucket, budget, secs in WARM_LOG:
    print(f"  prompt bucket {bucket:4d}, answer budget {budget:4d}: {secs:5.1f} s")

client = OpenAI(base_url=BASE_URL, api_key="not-needed")
MODEL = client.models.list().data[0].id
RESULTS = []  # one row per sample, for the summary


def ask(messages, *, thinking=False, show=True, label="", cli=None, model=None, **params):
    """Stream one chat request and return the text, the reasoning and the timings."""
    cli, model = cli or client, model or MODEL
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    extra = {"chat_template_kwargs": {"enable_thinking": thinking}}
    for key in ("top_k",):  # not in the OpenAI schema, so it travels in the request body
        if key in params:
            extra[key] = params.pop(key)
    t0 = time.perf_counter()
    first = None
    content, reasoning, usage, finish, compiled = [], [], None, None, None
    stream = cli.chat.completions.create(
        model=model, messages=messages, stream=True, stream_options={"include_usage": True},
        extra_body=extra, **params,
    )  # fmt: skip
    for ev in stream:
        if ev.usage:
            usage = ev.usage
        if (ev.model_extra or {}).get("x_new_shape") is not None:  # this notebook's server only
            compiled = ev.model_extra["x_new_shape"]
        if not ev.choices:
            continue
        d = ev.choices[0].delta
        r = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None)
        if (r or d.content) and first is None:
            first = time.perf_counter() - t0
        if r:
            reasoning.append(r)
        if d.content:
            content.append(d.content)
        finish = ev.choices[0].finish_reason or finish
    total = time.perf_counter() - t0
    n = usage.completion_tokens if usage else None
    rate = (n - 1) / (total - first) if n and n > 1 and first and total > first else None
    out = {
        "label": label, "content": "".join(content).strip(), "reasoning": "".join(reasoning).strip(),
        "finish": finish, "prompt_tokens": usage.prompt_tokens if usage else None, "completion_tokens": n,
        "first_token_s": first, "total_s": total, "decode_tok_s": rate, "compiled": compiled,
    }  # fmt: skip
    if show:
        if out["reasoning"]:
            print(f"--- reasoning ({len(out['reasoning'])} chars) ---\n{out['reasoning']}\n--- answer ---")
        print(out["content"])
        r = f"{rate:.1f} tok/s" if rate else "n/a"
        f = f"{first:.2f} s" if first is not None else "n/a"
        print(
            f"\n[{finish}] prompt {out['prompt_tokens']} tok, completion {n} tok, first token {f}, "
            f"decode {r}, total {total:.1f} s" + (" (compiled a new shape)" if compiled else "")
        )
    if label:
        RESULTS.append(out)
    return out


print(f"model: {MODEL}")

# %% [markdown]
# ### 3.1 Thinking off
#
# The default. The model answers directly. Temperature 0 picks the most likely token every step, so
# the answer is the same on every run.

# %%
# Cell 3.2: thinking off, greedy
r_plain = ask(
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    temperature=0,
    max_tokens=512,
    label="thinking off",
)

# %% [markdown]
# ### 3.2 Thinking on
#
# The same question with `chat_template_kwargs={"enable_thinking": True}`. The model writes its
# reasoning first, which the server returns as `reasoning_content`, then the answer. Thinking costs
# tokens: compare the completion count with the cell above. The answer budget is larger here
# because the reasoning counts against `max_tokens`.

# %%
# Cell 3.3: thinking on, greedy
r_think = ask(
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    thinking=True,
    temperature=0,
    max_tokens=2048,
    label="thinking on",
)
print(
    f"\nthinking used {r_think['completion_tokens']} tokens against {r_plain['completion_tokens']} without; "
    f"wall time {r_think['total_s']:.1f} s against {r_plain['total_s']:.1f} s"
)

# %% [markdown]
# ### 3.3 Sampling: temperature, top-k and seed
#
# With temperature above 0 the model samples, so answers vary. A fixed `seed` makes a sampled answer
# repeat exactly; different seeds give different answers. `top_k` limits each step to the k most
# likely tokens. Gemma 4's own `generation_config.json` suggests temperature 1.0, top-k 64 and top-p
# 0.95; this engine has no top-p, so it is left out.

# %%
# Cell 3.4: same seed twice, then two other seeds
prompt = "Write a one-sentence slogan for a bakery that only opens at night."
samples = [ask(prompt, temperature=1.0, top_k=64, seed=s, max_tokens=64, show=False) for s in (7, 7, 8, 9)]
for s, r in zip((7, 7, 8, 9), samples):
    print(f"seed {s}: {r['content']}")
print(f"\nseed 7 repeated exactly: {samples[0]['content'] == samples[1]['content']}")
print(f"distinct answers across seeds 7, 8, 9: {len({r['content'] for r in samples[1:]})} of 3")

# %%
# Cell 3.5: what top-k costs per token on this engine
prompt = "Describe the water cycle in about 150 words."
ask(prompt, temperature=0, max_tokens=256, show=False)  # compile this shape once before timing it
for label, params in (
    ("greedy", {"temperature": 0}),
    ("temperature 1, top-k off", {"temperature": 1.0}),
    ("temperature 1, top-k 64", {"temperature": 1.0, "top_k": 64}),
):
    r = ask(prompt, max_tokens=256, seed=0, show=False, label=f"sampling: {label}", **params)
    print(f"{label:26s} {r['decode_tok_s']:6.1f} tok/s  ({r['completion_tokens']} tokens)")

# %% [markdown]
# **What to look for.** On the `tpu-vllm-v5e1-2b` rig, vLLM measured 125 tok/s greedy and 45 tok/s
# with top-k 64 on the same chip type, because its top-k step was slow. The rows above show what
# top-k costs in this engine on your chip.

# %% [markdown]
# ### 3.4 Limits and stops: `max_tokens` and `stop`
#
# `max_tokens` cuts the answer at a token count, and `finish_reason` says `length`. A stop sequence
# ends the answer when it appears, the stop text is left out, and `finish_reason` says `stop`.

# %%
# Cell 3.6: a token limit, then a stop sequence
print("max_tokens=20:")
ask("List the planets of the solar system, one per line, with one fact each.", temperature=0, max_tokens=20)
print("\nstop=['Saturn']:")
_ = ask(
    "List the planets of the solar system in order from the Sun, one per line.",
    temperature=0,
    max_tokens=256,
    stop=["Saturn"],
)

# %% [markdown]
# ### 3.5 System prompt and multi-turn chat
#
# The client keeps the conversation: each turn sends the whole history back. The system prompt sets
# the voice for every turn. The second turn's prompt is longer, because it contains the first.

# %%
# Cell 3.7: a system prompt and two turns
history = [
    {"role": "system", "content": "You are a terse ship's navigator. Answer in at most two sentences."},
    {"role": "user", "content": "We are at 45 N, 30 W. Which ocean are we in?"},
]
r1 = ask(history, temperature=0, max_tokens=128)
history += [{"role": "assistant", "content": r1["content"]}, {"role": "user", "content": "Nearest land?"}]
print()
r2 = ask(history, temperature=0, max_tokens=128, label="multi-turn, turn 2")

# %% [markdown]
# ### 3.6 JSON output
#
# Ask for JSON in the prompt and parse the reply. This server does not constrain decoding to a
# schema, so the cell checks that the reply parses rather than assuming it.

# %%
# Cell 3.8: JSON by instruction, then parsed
r = ask(
    [
        {"role": "system", "content": "Reply with JSON only: no prose, no code fences."},
        {
            "role": "user",
            "content": 'Extract {"name": str, "city": str, "year": int} from: '
            '"Ada moved to Lisbon in 2019 to open a bookshop."',
        },
    ],
    temperature=0,
    max_tokens=128,
    label="JSON extraction",
)
text = r["content"].removeprefix("```json").removeprefix("```").removesuffix("```").strip()
try:
    print("\nparsed:", json.loads(text))
except json.JSONDecodeError as e:
    print(f"\nthe reply is not valid JSON ({e})")

# %% [markdown]
# ### 3.7 Long prompts
#
# Reading a prompt (prefill) processes all of its tokens at once, so first-token time grows with
# prompt length far more slowly than token by token would. Each prompt length here lands in a new
# bucket, so each first call compiles. The cell asks twice per length and reports the second, which
# is the time a user sees once the shape is warm. The longest prompt is close to this server's
# 4,096-token limit; the cell after it shows what a longer one gets.

# %%
# Cell 3.9: first-token time against prompt length
filler = "The quick brown fox jumps over the lazy dog near the riverbank at dawn. "
print(f"{'prompt tokens':>13s} {'first call s':>12s} {'warm first token s':>18s} {'decode tok/s':>12s}")
for reps in (5, 40, 160, 240):
    msg = filler * reps + "\n\nHow many times does the word 'fox' appear above? Answer with a number."
    cold = ask(msg, temperature=0, max_tokens=16, show=False)
    warm = ask(msg, temperature=0, max_tokens=16, show=False, label=f"prompt {cold['prompt_tokens']} tokens")
    print(
        f"{warm['prompt_tokens']:13d} {cold['first_token_s']:12.2f} {warm['first_token_s']:18.3f} "
        f"{warm['decode_tok_s'] or 0:12.1f}"
    )

# %%
# Cell 3.10: a prompt over the limit is refused with a clear error, and the server stays up
import openai

try:
    ask(filler * 400 + "\n\nSummarize.", max_tokens=16, show=False)
except openai.BadRequestError as e:
    print(f"HTTP {e.status_code}: {e.body.get('message') if isinstance(e.body, dict) else e.body}")
print("server still answering:", ask("Say OK.", temperature=0, max_tokens=4, show=False)["content"])

# %% [markdown]
# ### 3.8 The server's view
#
# The counters the chat page reads, straight from `/metrics`.

# %%
# Cell 3.11: the server's counters
print(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics").read().decode())

# %% [markdown]
# ## 4. A deployed endpoint (optional)
#
# The same client calls work against any OpenAI-compatible server. To compare with a vLLM
# deployment, such as the [`tpu-vllm-v5e1-2b`](https://github.com/xbill9/tpu-inference/tree/main/tpu-vllm-v5e1-2b)
# rig, add a Colab secret named `REMOTE_OPENAI_BASE_URL` (key icon in the left bar) set to its base
# URL, for example `http://<tpu-ip>:8000/v1`, and give this notebook access. Without it, this section
# is skipped.
#
# vLLM returns reasoning only when it was started with a reasoning parser
# (`--reasoning-parser gemma4`); otherwise the thinking text arrives inside `content`.

# %%
# Cell 4.1: the same calls against a remote endpoint, if one is configured
REMOTE = os.environ.get("REMOTE_OPENAI_BASE_URL")
if not REMOTE:
    try:
        from google.colab import userdata

        REMOTE = userdata.get("REMOTE_OPENAI_BASE_URL")
    except Exception:  # not on Colab, secret not set, or notebook not granted access
        REMOTE = None

if not REMOTE:
    print("no REMOTE_OPENAI_BASE_URL set: skipping the deployed-endpoint comparison")
else:
    remote = OpenAI(base_url=REMOTE.rstrip("/"), api_key=os.environ.get("REMOTE_OPENAI_API_KEY", "not-needed"))
    remote_model = remote.models.list().data[0]
    print(f"remote: {REMOTE}\nmodel:  {remote_model.id}  (max_model_len {getattr(remote_model, 'max_model_len', '?')})")
    try:
        print(
            "version:",
            json.load(urllib.request.urlopen(REMOTE.rstrip("/").removesuffix("/v1") + "/version", timeout=10)),
        )
    except OSError:
        pass
    q = "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"
    print("\n== thinking off ==")
    ask(q, cli=remote, model=remote_model.id, temperature=0, max_tokens=512, label="remote: thinking off")
    print("\n== thinking on ==")
    ask(
        q, cli=remote, model=remote_model.id, thinking=True, temperature=0, max_tokens=2048, label="remote: thinking on"
    )
    print("\n== decode speed, 256 tokens ==")
    ask("Describe the water cycle in about 150 words.", cli=remote, model=remote_model.id, temperature=0,
        max_tokens=256, label="remote: greedy 256")  # fmt: skip

# %% [markdown]
# ## 5. llama.cpp on the host CPU
#
# [`xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf)
# holds the same trained 4-bit weights as a GGUF file for llama.cpp. llama.cpp has no TPU backend, so
# here it runs on the Colab VM's CPU, next to the TPU. It has nothing to compile, which makes it a
# useful start-up comparison; the decode rate shows what the TPU adds.
#
# The llama.cpp build is pinned to release `b12061`, and the server is started with its reasoning
# parser so thinking arrives as `reasoning_content`, the same as from the TPU server.

# %%
# Cell 5.1: download llama.cpp and the GGUF, start llama-server, time it to the first answer
import tarfile

LLAMA_TAG = "b12061"
GGUF_REPO, GGUF_REV = "xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf", "74b2c92c8f7f84d3d9c39e8026f6a28f62788255"
GGUF_FILE = "gemma-4-E2B-it-q4_0-exact.gguf"

t_cpu0 = time.perf_counter()
tgz = WORK / f"llama-{LLAMA_TAG}.tar.gz"
if not tgz.exists():
    urllib.request.urlretrieve(
        f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/llama-{LLAMA_TAG}-bin-ubuntu-x64.tar.gz",
        tgz,
    )
with tarfile.open(tgz) as t:
    t.extractall(WORK, filter="data")
LLAMA_DIR = WORK / f"llama-{LLAMA_TAG}"

proc = subprocess.run(
    [sys.executable, "-c", DOWNLOAD.replace("snapshot_download(sys.argv[1], revision=sys.argv[2])",
     f"snapshot_download(sys.argv[1], revision=sys.argv[2], allow_patterns=['{GGUF_FILE}'])"),
     GGUF_REPO, GGUF_REV],
    capture_output=True, text=True,
)  # fmt: skip
if proc.returncode != 0:
    raise RuntimeError(f"GGUF download failed:\n{proc.stderr[-2000:]}")
gguf = pathlib.Path(proc.stdout.strip().splitlines()[-1]) / GGUF_FILE
t_cpu_files = time.perf_counter() - t_cpu0
print(f"llama.cpp {LLAMA_TAG} and {gguf.stat().st_size / 1e9:.2f} GB GGUF fetched in {t_cpu_files:.0f} s")

LLAMA_PORT = 8081
threads = os.cpu_count()
t_launch = time.perf_counter()
llama = subprocess.Popen(
    [
        str(LLAMA_DIR / "llama-server"), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(LLAMA_PORT),
        "-c", str(MAX_MODEL_LEN), "-t", str(threads), "--jinja", "--reasoning-format", "deepseek", "--no-webui",
    ],
    stdout=subprocess.DEVNULL, stderr=open(WORK / "llama-server.log", "w"),
    env={**os.environ, "LD_LIBRARY_PATH": str(LLAMA_DIR)},
)  # fmt: skip
for _ in range(600):
    if llama.poll() is not None:
        raise RuntimeError(f"llama-server exited:\n{(WORK / 'llama-server.log').read_text()[-3000:]}")
    try:
        if urllib.request.urlopen(f"http://127.0.0.1:{LLAMA_PORT}/health", timeout=2).status == 200:
            break
    except OSError:
        time.sleep(0.5)
else:
    raise RuntimeError("llama-server did not become healthy in 5 minutes")
t_llama_ready = time.perf_counter() - t_launch

cpu = OpenAI(base_url=f"http://127.0.0.1:{LLAMA_PORT}/v1", api_key="not-needed")
cpu_model = cpu.models.list().data[0].id
first = ask("Say hello.", cli=cpu, model=cpu_model, temperature=0, max_tokens=16, show=False)
print(
    f"llama-server ({threads} threads) healthy {t_llama_ready:.1f} s after launch; first answer {first['total_s']:.1f} s later"
)
LLAMA_START = {"files_s": t_cpu_files, "launch_to_healthy_s": t_llama_ready, "first_answer_s": first["total_s"]}

# %%
# Cell 5.2: the same requests on the CPU -- thinking off (about two minutes at CPU speed)
q = "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"
_ = ask(q, cli=cpu, model=cpu_model, temperature=0, max_tokens=512, label="llama.cpp CPU: thinking off")

# %%
# Cell 5.3: thinking on (about four minutes at CPU speed)
_ = ask(q, cli=cpu, model=cpu_model, thinking=True, temperature=0, max_tokens=2048, label="llama.cpp CPU: thinking on")

# %%
# Cell 5.4: decode speed, 256 tokens
_ = ask("Describe the water cycle in about 150 words.", cli=cpu, model=cpu_model, temperature=0, max_tokens=256,
    label="llama.cpp CPU: greedy 256")  # fmt: skip

# %% [markdown]
# ## 6. Summary
#
# Every number in these tables was measured in this run.

# %%
# Cell 6.1: start-up and per-request results
print("start-up, seconds since the first cell started")
for stage, t in STAGES.items():
    print(f"  {stage:22s} {t:7.0f}")
print(
    f"\nllama.cpp on the CPU: files {LLAMA_START['files_s']:.0f} s, launch to healthy "
    f"{LLAMA_START['launch_to_healthy_s']:.1f} s, first answer {LLAMA_START['first_answer_s']:.1f} s after that"
)

print(
    f"\n{'request':40s} {'prompt':>6s} {'out':>5s} {'first tok s':>11s} {'decode tok/s':>12s} {'finish':>7s}  compiled"
)
for r in RESULTS:
    rate = f"{r['decode_tok_s']:.1f}" if r["decode_tok_s"] else "n/a"
    ft = f"{r['first_token_s']:.2f}" if r["first_token_s"] is not None else "n/a"
    print(
        f"{r['label']:40s} {r['prompt_tokens'] or 0:6d} {r['completion_tokens'] or 0:5d} {ft:>11s} {rate:>12s} "
        f"{r['finish'] or '':>7s}  {'yes' if r['compiled'] else ''}"
    )

tpu = next(r for r in RESULTS if r["label"] == "sampling: greedy")
cpu_r = next(r for r in RESULTS if r["label"] == "llama.cpp CPU: greedy 256")
if tpu["decode_tok_s"] and cpu_r["decode_tok_s"]:
    print(
        f"\ngreedy decode, 256 tokens: TPU {tpu['decode_tok_s']:.1f} tok/s, CPU {cpu_r['decode_tok_s']:.1f} tok/s "
        f"({tpu['decode_tok_s'] / cpu_r['decode_tok_s']:.1f}x)"
    )

# %% [markdown]
# ## 7. Clean up
#
# The servers stop with the runtime. To free the chip for something else without restarting, run the
# cell below. Disconnect the runtime when you are done (Runtime, then Disconnect and delete runtime)
# so it stops using compute units.

# %%
# Cell 7.1: stop the servers and free the chip
SERVER.shutdown()
for p in (ui, llama):
    p.terminate()
ENGINE = None
import gc

gc.collect()
print(f"HBM in use: {dev.memory_stats()['bytes_in_use'] / 1e9:.2f} GB")
