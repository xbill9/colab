# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# ## 1. Download
#
# The model and the prebuilt tarball download in the background; this cell returns at once and reports
# the TPU runtime library while they arrive.

# %%
# Cell 1: start the downloads, find libtpu
import glob
import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import time
import urllib.request

T_START = time.perf_counter()
WORK = pathlib.Path("/content" if os.path.isdir("/content") else os.path.expanduser("~/llamacpp-tpu"))
WORK.mkdir(parents=True, exist_ok=True)

MODEL_REPO = "xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf"
MODEL_FILE = "gemma-4-E2B-it-q4_0-exact.gguf"
MODEL_REV = "74b2c92c8f7f84d3d9c39e8026f6a28f62788255"
RELEASE = "ggml-tpu-a4a425d-llama.cpp-7182ef71e-linux-x86_64"
RELEASE_URL = f"https://github.com/xbill9/ggml-tpu/releases/download/v0.1.0/{RELEASE}.tar.gz"
RELEASE_SHA256 = "400e2a629c28a5f262a185de09458b56ab0c19147bcd3232a4ab3aa81eea1beb"
# The chat page from a later ggml-tpu commit: one Thinking switch and fixed 2,048-token replies, and a
# reply that runs out while reasoning says so instead of showing a blank answer.
CHAT_UI_URL = "https://raw.githubusercontent.com/xbill9/ggml-tpu/9829844f980832853fa6365b716d9335fa1adff1/ui/chat_ui.py"
CHAT_UI_SHA256 = "a9b6dd3b6fb6775429995ec2be3e72958b60219e45abf7ba735a8059a83d3742"

DOWNLOAD = """
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download(sys.argv[1], sys.argv[2], revision=sys.argv[3], token=False))
"""
model_dl = subprocess.Popen(
    [sys.executable, "-c", DOWNLOAD, MODEL_REPO, MODEL_FILE, MODEL_REV],
    stdout=subprocess.PIPE,
    stderr=open(WORK / "download.log", "w"),
    text=True,
)

tarball = WORK / f"{RELEASE}.tar.gz"
urllib.request.urlretrieve(RELEASE_URL, tarball)
if hashlib.sha256(tarball.read_bytes()).hexdigest() != RELEASE_SHA256:
    raise RuntimeError(f"{tarball.name}: SHA-256 does not match the release")
with tarfile.open(tarball) as tf:
    tf.extractall(WORK, filter="data")
BIN = WORK / RELEASE
chat_ui = urllib.request.urlopen(CHAT_UI_URL, timeout=60).read()
if hashlib.sha256(chat_ui).hexdigest() != CHAT_UI_SHA256:
    raise RuntimeError("chat_ui.py: SHA-256 does not match ggml-tpu 9829844")
(BIN / "chat_ui.py").write_bytes(chat_ui)
print((BIN / "VERSION").read_text())

# libtpu ships with JAX on Colab TPU runtimes; install it only if it is missing
if importlib.util.find_spec("libtpu") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "libtpu"], check=True)
LIBTPU = next(
    iter(
        glob.glob(
            os.path.join(os.path.dirname(importlib.util.find_spec("libtpu").origin), "**", "libtpu.so"), recursive=True
        )
    ),
    None,
)
if LIBTPU is None:
    raise RuntimeError("libtpu.so not found: is this a TPU runtime? Runtime > Change runtime type > v5e-1 TPU")
for pkg in ("libtpu", "jax"):
    try:
        print(f"{pkg:7s} {md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"{pkg:7s} not installed")
print(f"libtpu.so: {LIBTPU}")
print(f"\ncell 1 done {time.perf_counter() - T_START:.0f} s after it started; the model is still downloading")

# %% [markdown]
# ## 2. Serve
#
# This cell waits for the model, starts `llama-server` on the TPU (device `TPU0`), then the chat page.
# `llama-server` is ready once the weights are on the chip; the first reply then waits for XLA to
# compile.

# %%
# Cell 2: llama-server on the TPU, then the chat page
out, _ = model_dl.communicate()
if model_dl.returncode != 0:
    raise RuntimeError(f"model download failed:\n{(WORK / 'download.log').read_text()[-2000:]}")
MODEL_PATH = out.strip().splitlines()[-1]
print(f"model ready {time.perf_counter() - T_START:.0f} s after cell 1 started: {MODEL_PATH}")

PORT = 8000  # 8080 was already taken on a Colab TPU runtime (2026-11-19)
UI_PORT = 8765

# A re-run of this cell first stops the servers the previous run started.
for name in ("ui", "server"):
    proc = globals().get(name)
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

import socket

for port in (PORT, UI_PORT):
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"port {port} is already in use; Runtime, then Restart session, then run all")
env = dict(
    os.environ,
    GGML_BACKEND_PATH=str(BIN / "libggml-tpu.so"),
    GGML_TPU_LIBTPU=LIBTPU,
    GGML_TPU_PRECISION="DEFAULT",
)
server = subprocess.Popen(
    [
        str(BIN / "llama-server"),
        "-m",
        MODEL_PATH,
        "-dev",
        "TPU0",
        "-ngl",
        "99",
        "-fa",
        "off",
        "-c",
        "8192",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--parallel",
        "1",
        "--metrics",
        "--jinja",
        "--alias",
        "gemma-4-E2B-it-q4_0-exact",
    ],
    env=env,
    stdout=subprocess.DEVNULL,
    stderr=open(WORK / "llama-server.log", "w"),
)
t0 = time.perf_counter()
while True:
    try:
        if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200:
            break
    except OSError:
        pass
    if server.poll() is not None:
        raise RuntimeError(f"llama-server exited:\n{(WORK / 'llama-server.log').read_text()[-3000:]}")
    time.sleep(0.2)
print(
    f"llama-server ready in {time.perf_counter() - t0:.1f} s, {time.perf_counter() - T_START:.0f} s after cell 1 started"
)

ui = subprocess.Popen(
    [
        sys.executable,
        str(BIN / "chat_ui.py"),
        "--upstream",
        f"http://127.0.0.1:{PORT}",
        "--port",
        str(UI_PORT),
        "--rig",
        "ACCELERATOR_TYPE=Colab TPU v5e-1",
        "GGML_TPU_EXECUTOR=pjrt",
        "GGML_TPU_QUANT=bf16",
        "GGML_TPU_PRECISION=DEFAULT",
    ],
    stdout=subprocess.DEVNULL,
    stderr=open(WORK / "chat_ui.log", "w"),
)
for _ in range(50):
    try:
        json.load(urllib.request.urlopen(f"http://127.0.0.1:{UI_PORT}/ui/status", timeout=2))
        break
    except OSError:
        time.sleep(0.2)
else:
    raise RuntimeError(f"chat page did not start:\n{(WORK / 'chat_ui.log').read_text()[-2000:]}")

try:
    from google.colab import output

    output.serve_kernel_port_as_iframe(UI_PORT, height=720)
except ImportError:
    print(f"not on Colab: open http://127.0.0.1:{UI_PORT} in a browser on this machine")
