# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

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
if OPENAI_INSTALL is not None and OPENAI_INSTALL.wait() != 0:
    raise RuntimeError(f"pip install openai failed:\n{(WORK / 'prep.log').read_text()[-2000:]}")
from openai import OpenAI

t0 = time.perf_counter()
if WARM_THREAD.is_alive():
    print("waiting for the background compiles to finish, so no timing below includes one ...", flush=True)
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
# llama.cpp reads the same GGUF file the TPU engine was loaded from in section 2. llama.cpp has no TPU
# backend, so here it runs on the Colab VM's CPU, next to the TPU. It has nothing to compile, which makes it a
# useful start-up comparison; the decode rate shows what the TPU adds.
#
# The llama.cpp build is pinned to release `b12061`, and the server is started with its reasoning
# parser so thinking arrives as `reasoning_content`, the same as from the TPU server.

# %%
# Cell 5.1: download llama.cpp, start llama-server on the same GGUF, time it to the first answer
import tarfile

LLAMA_TAG = "b12061"

# The GGUF is the file the TPU engine was loaded from in section 2, already on disk.
print("downloading llama.cpp (about 23 MB); the GGUF is the one section 2 downloaded ...", flush=True)
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
gguf = DL["path"]
t_cpu_files = time.perf_counter() - t_cpu0
print(f"llama.cpp {LLAMA_TAG} fetched in {t_cpu_files:.0f} s; GGUF {gguf.name}, {gguf.stat().st_size / 1e9:.2f} GB")

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
for stage, t in sorted(STAGES.items(), key=lambda kv: kv[1]):
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
