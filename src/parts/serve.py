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
# The first cell fetches the config, tokenizer and chat template (32 MB), then starts the 2.62 GB GGUF
# download in its own process, with the engine's code and the OpenAI client installing beside it. It
# returns in about two seconds. The download runs while the next cells read the chip and compile, so
# it costs no waiting time of its own. Every file and the engine are pinned to a fixed revision, so
# the bytes you run are the bytes this notebook was written against.

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

# The weights: Gemma 4 E2B's QAT 4-bit weights as one Q4_0 GGUF file (2.62 GB), loaded onto the chip
# directly. The safetensors repo supplies only config.json, the tokenizer and the chat template.
GGUF_REPO = "xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf"
GGUF_FILE = "gemma-4-E2B-it-q4_0-exact.gguf"
GGUF_REV = "74b2c92c8f7f84d3d9c39e8026f6a28f62788255"
MODEL_ID = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text"
MODEL_REV = "315aec06cff196fd6479aa2669bf9379b0d0268a"
WORK = pathlib.Path("/content" if os.path.isdir("/content") else os.path.expanduser("~/gemma4-colab"))
WORK.mkdir(parents=True, exist_ok=True)

from huggingface_hub import HfApi, snapshot_download

# The small files first, in this cell (about a second): the engine compiles from config.json while
# the weights download.
MODEL_DIR = pathlib.Path(
    snapshot_download(MODEL_ID, revision=MODEL_REV, ignore_patterns=["*.safetensors"], token=False)
)
gguf_meta = HfApi(token=False).model_info(GGUF_REPO, revision=GGUF_REV, files_metadata=True).siblings
TOTAL_BYTES = next(f.size for f in gguf_meta if f.rfilename == GGUF_FILE)  # public: no token
print(
    f"config and tokenizer: {MODEL_ID} @ {MODEL_REV[:8]}, {sum(f.stat().st_size for f in MODEL_DIR.iterdir()) / 1e6:.0f} MB"
)
print(f"weights:              {GGUF_REPO} @ {GGUF_REV[:8]}, {GGUF_FILE}, {TOTAL_BYTES / 1e9:.2f} GB")
print("estimate: about 25 s. On a Colab v5e-1 on 2026-11-20 this file downloaded in 25.0 s.")
print("          The bar below measures this runtime's own rate.")
print("          The next cells run while it downloads; there is no need to wait here.\n", flush=True)

DOWNLOAD = """
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download(sys.argv[1], sys.argv[2], revision=sys.argv[3], token=False))
"""
download_log = open(WORK / "download.log", "w")
download = subprocess.Popen(
    [sys.executable, "-c", DOWNLOAD, GGUF_REPO, GGUF_FILE, GGUF_REV],
    stdout=subprocess.PIPE,
    stderr=download_log,
    text=True,
)

# The engine's code (tpu-jax, pinned) and the OpenAI client install in the background too.
import importlib.util  # noqa: E402

TPU_JAX_COMMIT = "4b9f8e94619f1995206617e07e3c4ac62685d47b"
ENGINE_DIR = WORK / "tpu-jax"
prep_steps = []
if not (ENGINE_DIR / ".git").exists():
    prep_steps.append(f"git clone -q https://github.com/xbill9/tpu-jax {ENGINE_DIR}")
prep_steps.append(f"git -C {ENGINE_DIR} checkout -q {TPU_JAX_COMMIT}")
missing = [p for p in ("safetensors", "tokenizers") if importlib.util.find_spec(p) is None]
if missing:
    prep_steps.append(f"{sys.executable} -m pip install -q {' '.join(missing)}")
prep_log = open(WORK / "prep.log", "w")
# The OpenAI client is first used in section 3, so its install runs on its own and nothing waits for it.
OPENAI_INSTALL = None
if importlib.util.find_spec("openai") is None:
    OPENAI_INSTALL = subprocess.Popen(
        [sys.executable, "-m", "pip", "install", "-q", "openai"], stdout=prep_log, stderr=prep_log
    )
PREP = subprocess.Popen(["bash", "-c", "set -e; " + "; ".join(prep_steps)], stdout=prep_log, stderr=prep_log)


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


DL = {"updates": 0}  # filled by the watcher: "path", "seconds", "error", and how often it redrew the bar


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
    DL["done_at"] = time.perf_counter() - T_START
    out = download.stdout.read()
    download_log.close()
    if download.returncode == 0:
        DL["path"] = pathlib.Path(out.strip().splitlines()[-1])
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

# %% [markdown]
# ## 2. Load and serve
#
# Three steps, timed separately:
#
# 1. **One compile, during the download.** XLA compiles a program for each request shape, and it
#    needs only the shapes of the weights, not their values. The cell reads every tensor's shape from
#    the GGUF's header (one 16 MB request), loads the engine with zero weights of those shapes and runs
#    one short request, which compiles the shape most chat turns use. Other shapes (long prompts, long
#    answers) compile the first time they are used, or in the background in cell 2.6.
# 2. **The weights.** When the download is done, the GGUF's 4-bit blocks go to the chip and are
#    expanded there to bf16, in place of the zeros. The shapes match, so nothing compiles again. bf16
#    weights decode faster than 4-bit ones unpacked at every step (131 against 103 tokens per second,
#    measured on a Colab v5e-1 on 2026-10-21), and at 9.3 GB they fit this chip's 16.9 GB of HBM.
# 3. **Server and chat page.** An OpenAI-compatible HTTP server over the engine, and this repo's
#    [`ui/chat_ui.py`](https://github.com/xbill9/colab/blob/main/ui/chat_ui.py), a chat page that
#    shows what is running and lets you chat in the browser.

# %%
# Cell 2.1: the engine, pinned, and the checkpoint on the chip
import json

if PREP.wait() != 0:
    raise RuntimeError(f"engine setup failed:\n{(WORK / 'prep.log').read_text()[-2000:]}")
sys.path.insert(0, str(ENGINE_DIR))
from jax_engine import JaxGemmaEngine  # noqa: E402


import io  # noqa: E402
import re  # noqa: E402
import struct  # noqa: E402
import urllib.request  # noqa: E402

import numpy as np  # noqa: E402
import safetensors.flax  # noqa: E402
from ports.gemma4.jax_e_loader import convert_safetensors_to_jax_params  # noqa: E402

import datetime  # noqa: E402

import jinja2  # noqa: E402
import jinja2.ext  # noqa: E402
import jinja2.sandbox  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402


# The tokenizer and chat template come straight from `tokenizers` and Jinja: transformers would import
# PyTorch first, which took about 10 s here. On 2026-11-20 this matched transformers' AutoTokenizer on
# every test conversation (16 of 16, thinking on and off) and on 600 of 600 random decodes.
class ChatTokenizer:
    """The checkpoint's tokenizer.json and chat_template.jinja, rendered the way transformers renders
    them (a sandboxed Jinja environment with trim_blocks, lstrip_blocks and loop controls)."""

    def __init__(self, model_dir):
        self.tk = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        cfg = json.loads((model_dir / "tokenizer_config.json").read_text())
        tok = lambda t: t["content"] if isinstance(t, dict) else t  # noqa: E731
        self.bos_token, self.eos_token = tok(cfg.get("bos_token")), tok(cfg.get("eos_token"))
        self.bos_token_id = self.tk.token_to_id(self.bos_token)
        env = jinja2.sandbox.ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols]
        )

        def raise_exception(message):
            raise jinja2.exceptions.TemplateError(message)

        env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
            x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
        )
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)
        self.template = env.from_string((model_dir / "chat_template.jinja").read_text())

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        text = self.template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            **kwargs,
        )
        return self.encode(text) if tokenize else text

    def encode(self, text, add_special_tokens=False):
        return self.tk.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids, skip_special_tokens=False):
        return self.tk.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def convert_tokens_to_ids(self, token):
        return self.tk.token_to_id(token)


# --- The GGUF, read as the tensors tpu-jax reads from safetensors ---------------------------------
# GGUF names -> the safetensors names the engine expects. rope_freqs is skipped: the engine computes
# RoPE from config.json.
GGUF_TOP = {
    "token_embd.weight": "embed_tokens.weight",
    "per_layer_token_embd.weight": "embed_tokens_per_layer.weight",
    "per_layer_model_proj.weight": "per_layer_model_projection.weight",
    "per_layer_proj_norm.weight": "per_layer_projection_norm.weight",
    "output_norm.weight": "norm.weight",
}
GGUF_BLK = {
    "attn_q": "self_attn.q_proj.weight", "attn_k": "self_attn.k_proj.weight",
    "attn_v": "self_attn.v_proj.weight", "attn_output": "self_attn.o_proj.weight",
    "attn_q_norm": "self_attn.q_norm.weight", "attn_k_norm": "self_attn.k_norm.weight",
    "ffn_gate": "mlp.gate_proj.weight", "ffn_up": "mlp.up_proj.weight", "ffn_down": "mlp.down_proj.weight",
    "inp_gate": "per_layer_input_gate.weight", "proj": "per_layer_projection.weight",
    "attn_norm": "input_layernorm.weight", "post_attention_norm": "post_attention_layernorm.weight",
    "ffn_norm": "pre_feedforward_layernorm.weight", "post_ffw_norm": "post_feedforward_layernorm.weight",
    "post_norm": "post_per_layer_input_norm.weight", "layer_output_scale": "layer_scalar",
}  # fmt: skip
GGUF_F32, GGUF_Q4_0 = 0, 2


def safetensors_name(gguf_name):
    if gguf_name in GGUF_TOP:
        return "model.language_model." + GGUF_TOP[gguf_name]
    m = re.fullmatch(r"blk\.(\d+)\.(.+)\.weight", gguf_name)
    if not m or m.group(2) not in GGUF_BLK:
        raise KeyError(f"GGUF tensor {gguf_name} has no tpu-jax counterpart")
    return f"model.language_model.layers.{m.group(1)}.{GGUF_BLK[m.group(2)]}"


def gguf_index(f):
    """GGUF v3 header -> (offset of the tensor data, [(safetensors name, [out, in] shape, type, offset)])."""
    r = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))  # noqa: E731
    if f.read(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    r("I")
    n_tensors, n_kv = r("QQ")
    align, fixed = 32, {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def value(t):  # metadata values are skipped, except the data alignment
        if t == 8:
            return f.read(r("Q")[0])
        if t == 9:
            at, n = r("IQ")
            if at in fixed:
                return f.seek(fixed[at] * n, 1)
            return [value(at) for _ in range(n)]
        return r(fmt[t])[0]

    for _ in range(n_kv):
        key = f.read(r("Q")[0]).decode()
        v = value(r("I")[0])
        if key == "general.alignment":
            align = v
    index = []
    for _ in range(n_tensors):
        name = f.read(r("Q")[0]).decode()
        dims = r("Q" * r("I")[0])
        t, off = r("IQ")
        if name != "rope_freqs.weight":
            index.append((safetensors_name(name), tuple(int(d) for d in reversed(dims)), t, off))
    p = f.tell()
    return p + (-p % align), index


def q4_0_expand(blocks):
    """uint8 [rows, n, 18] ggml block_q4_0 -> bf16 [rows, n*32]. A block is an fp16 step d and 16 bytes:
    low nibbles are elements 0..15, high nibbles 16..31, each q + 8. The value is bf16(q * d), the
    single rounding the engine's own 4-bit unpack gives."""
    d = jax.lax.bitcast_convert_type(
        blocks[..., 0].astype(jnp.uint16) | (blocks[..., 1].astype(jnp.uint16) << 8), jnp.float16
    )
    qs = blocks[..., 2:]
    q = jnp.concatenate([qs & 0xF, qs >> 4], axis=-1).astype(jnp.int32) - 8
    v = (q.astype(jnp.float32) * d.astype(jnp.float32)[..., None]).astype(jnp.bfloat16)
    return v.reshape(blocks.shape[0], -1)


q4_0_expand_jit = jax.jit(q4_0_expand)
q4_0_expand_into = jax.jit(
    lambda out, blocks, row: jax.lax.dynamic_update_slice(out, q4_0_expand(blocks), (row, 0)), donate_argnums=0
)
CHUNK_ROWS = 16384  # the two embedding tables expand in chunks, so neither needs an f32 copy


def gguf_to_chip(path, data_start, index):
    """Only the 4-bit blocks cross to the chip (2.6 GB); each tensor is expanded to bf16 there."""
    mm = np.memmap(path, np.uint8, "r")
    out = {}
    for name, shape, t, off in index:
        start = data_start + off
        if t == GGUF_F32:
            n = int(np.prod(shape))
            out[name] = jnp.asarray(np.frombuffer(mm[start : start + 4 * n], np.float32).reshape(shape), jnp.bfloat16)
        elif t == GGUF_Q4_0:
            rows, cols = shape
            blocks = mm[start : start + rows * cols // 32 * 18].reshape(rows, cols // 32, 18)
            if rows <= CHUNK_ROWS:
                out[name] = q4_0_expand_jit(np.ascontiguousarray(blocks))
            else:
                w = jnp.zeros(shape, jnp.bfloat16)
                for r0 in range(0, rows, CHUNK_ROWS):
                    w = q4_0_expand_into(w, np.ascontiguousarray(blocks[r0 : r0 + CHUNK_ROWS]), r0)
                out[name] = w
        else:
            raise TypeError(f"{name}: GGUF type {t} is neither F32 nor Q4_0")
    return jax.block_until_ready(out)


def engine_load(engine, tensors):
    """ENGINE.load, unchanged, fed `tensors` in place of the files it would read: it lists the folder's
    *.safetensors and reads each with safetensors.flax.load_file, so the folder holds one placeholder
    and that one call returns `tensors`."""
    folder = WORK / "engine-dir"
    folder.mkdir(exist_ok=True)
    for f in MODEL_DIR.iterdir():
        (folder / f.name).unlink(missing_ok=True)
        (folder / f.name).symlink_to(f.resolve())
    (folder / "gguf.safetensors").write_bytes(b"")
    real = safetensors.flax.load_file
    safetensors.flax.load_file = lambda p, *a, **k: tensors if p.endswith("gguf.safetensors") else real(p, *a, **k)
    try:
        engine.load(local_dir=str(folder))
    finally:
        safetensors.flax.load_file = real


TOK = ChatTokenizer(MODEL_DIR)
GEN_CFG = json.loads((MODEL_DIR / "generation_config.json").read_text())
EOS_IDS = GEN_CFG["eos_token_id"]
# Prompt plus answer. At 8,192 the prefill of a long prompt needs more HBM than this chip has left
# beside the unpacked weights (measured on a Colab v5e-1: 16.77 GB asked, 15.75 GB free), so the
# server stops at 4,096 and refuses longer prompts with a clear error.
MAX_MODEL_LEN = 4096
WARM_UP = [{"role": "user", "content": "Say hello."}]  # cell 2.2 sends this: prompt bucket 64, budget 512

# 1. Tensor shapes from the GGUF header: one 16 MB range request, while the file downloads.
t0 = time.perf_counter()
GGUF_URL = f"https://huggingface.co/{GGUF_REPO}/resolve/{GGUF_REV}/{GGUF_FILE}"
head = urllib.request.urlopen(urllib.request.Request(GGUF_URL, headers={"Range": "bytes=0-16777215"}), timeout=60)
DATA_START, GGUF_INDEX = gguf_index(io.BytesIO(head.read()))
print(f"GGUF header: {len(GGUF_INDEX)} tensors, read in {time.perf_counter() - t0:.1f} s")

# 2. The engine on zero weights of those shapes, and the warm-up request's programs compiled. XLA needs
#    only shapes and dtypes, so this runs during the download.
print(f"compiling the first request shape on zero weights, {time.perf_counter() - T_START:.0f} s after the first cell; "
      "this took 26 s on a Colab v5e-1 on 2026-11-20 ...", flush=True)  # fmt: skip
t0 = time.perf_counter()
ENGINE = JaxGemmaEngine(MODEL_ID, quant_mode="w4a16", max_model_len=MAX_MODEL_LEN, dequant_at_load=True)
engine_load(ENGINE, {name: jnp.zeros(shape, jnp.bfloat16) for name, shape, _, _ in GGUF_INDEX})
ENGINE.bos_token_id = TOK.bos_token_id
enc = TOK.apply_chat_template(WARM_UP, tokenize=True, add_generation_prompt=True, enable_thinking=False)
stream = ENGINE.generate_stream(list(enc["input_ids"] if hasattr(enc, "keys") else enc), 512, 0.0, -1, [], 0)
next(stream), next(stream)  # one prefill and one decode step: both programs compiled
stream.close()
ENGINE.params = None
STAGES["first shape compiled"] = time.perf_counter() - T_START
print(f"compiled in {time.perf_counter() - t0:.0f} s, {STAGES['first shape compiled']:.0f} s after the first cell")

# 3. The real weights, once the download is done, in place of the zeros. Same shapes and dtypes, so
#    the compiled programs are reused.
t0 = time.perf_counter()
if DOWNLOAD_THREAD.is_alive():
    print("waiting for the GGUF download (progress bar in cell 1.1) ...", flush=True)
DOWNLOAD_THREAD.join()
if "error" in DL:
    raise RuntimeError(f"download failed:\n{DL['error']}")
STAGES["download finished"] = DL["done_at"]
print(
    f"GGUF: {TOTAL_BYTES / 1e9:.2f} GB, downloaded in {DL['seconds']:.0f} s, done {STAGES['download finished']:.0f} s "
    f"after the first cell (this cell waited {time.perf_counter() - t0:.0f} s for it)"
)
t0 = time.perf_counter()
params = convert_safetensors_to_jax_params(
    gguf_to_chip(DL["path"], DATA_START, GGUF_INDEX),
    num_layers=ENGINE.config.num_hidden_layers,
    first_kv_shared_idx=ENGINE.config.first_kv_shared_layer_idx,
    attention_k_eq_v=ENGINE.config.attention_k_eq_v,
)
ENGINE.params = jax.device_put(params, ENGINE.device)
del params
ENGINE.weight_bytes = sum(int(x.size) * int(x.dtype.itemsize) for x in jax.tree_util.tree_leaves(ENGINE.params))
STAGES["weights on chip"] = time.perf_counter() - T_START
print(
    f"engine: tpu-jax {TPU_JAX_COMMIT[:7]}, {ENGINE.weight_bytes / 1e9:.2f} GB of bf16 weights on the chip, "
    f"expanded from the GGUF in {time.perf_counter() - t0:.0f} s"
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


# The first request: the shape most chat turns use (prompt bucket 64, answer budget 512), compiled in
# cell 2.1 during the download. The other common chat shapes compile in the background once the
# server is up (cell 2.6).
t0 = time.perf_counter()
warm = run_request({"messages": WARM_UP, "max_tokens": 16, "temperature": 0})
assert warm["new_shape"]
STAGES["first answer"] = time.perf_counter() - T_START
print(f"first answer: {warm['content']!r} in {time.perf_counter() - t0:.1f} s")

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
# **What you should see.** The compile finishing while or just after the download does, the weights
# expanding on the chip in a few seconds, then the chat page. On a Colab v5e-1 on 2026-11-20 the page
# was up 46 to 53 seconds after the first cell started, in three runs. The page's side panel shows the model,
# the server version and the request counters, all read from the server above. **You can chat with
# the model now**. Requests from the page and from any later cell take turns on the chip.
#
# The page sends no `enable_thinking`, so the server's `THINKING_DEFAULT` decides. To demo
# thinking in the page, run `THINKING_DEFAULT = True` in a new cell; the next message shows its
# reasoning in a collapsible section. The page also has max tokens, temperature, top-k and a system
# prompt in its side panel, and shows the first-token time and decode rate of each reply.

# %% [markdown]
# ### The chip on two simple kernels
#
# This cell runs after the server is up, so it never delays the first answer. A chat message sent
# while it runs waits a few seconds.

# %%
# Cell 2.5: what the chip does on two simple kernels: a bf16 matmul and a large copy
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
# ### Warming the other chat shapes in the background
#
# A chat's first few turns move through the 128, 256 and 512-token prompt buckets, and a thinking
# answer needs the 2048-token budget. Each would compile the first time a user hit it. The next cell
# compiles them in a background thread instead, one at a time. A chat message sent meanwhile waits
# for at most the one compile in progress.

# %%
# Cell 2.6: compile the common chat shapes in the background
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
