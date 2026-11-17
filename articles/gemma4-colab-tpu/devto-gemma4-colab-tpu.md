---
title: "Gemma 4 on a Hosted Colab TPU: The QAT Repack, Served in Two Minutes"
published: false
series: Colab
description: "One notebook loads the Gemma 4 E2B QAT repack onto a Colab v5e-1, serves it behind an OpenAI-compatible endpoint with a chat page, and steps through thinking, sampling, stop sequences, long prompts and a CPU comparison."
tags: gemma, tpu, jax, colab
cover_image: https://raw.githubusercontent.com/xbill9/colab/main/articles/gemma4-colab-tpu/devto-cover.b0d9000a.jpg
---

This article provides a step by step guide to serving Gemma 4 E2B on a hosted Colab v5e-1 runtime: one TPU v5e chip. A single notebook downloads the QAT repack, loads it with a pure-JAX engine, serves it behind an OpenAI-compatible endpoint with a browser chat page, and then exercises the serving options an application uses.

https://github.com/xbill9/colab

**The model is ready to chat 117 seconds after Run all, and decodes at 123.7 to 124.6 tokens per second.** One Colab v5e-1 runtime, 2026-11-17:

| Seconds after Run all | Stage |
| ---: | :--- |
| 8 | chip and versions read |
| 63 | 6.56 GB checkpoint downloaded |
| 93 | weights on the chip |
| 116 | first request compiled |
| **117** | **server and chat page up** |
| 296 | six more chat shapes compiled in the background |

---

#### What Is Being Served?

The model is [`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text). Google trained Gemma 4 E2B with quantization-aware training, which holds every weight on a 4-bit grid during training. The repack stores those trained 4-bit levels and their trained scales in the W4A16 format, with the vision and audio towers removed.

| Build | Download |
| :--- | ---: |
| Google `gemma-4-E2B-it-qat-w4a16-ct` | 8.32 GB |
| Repack `gemma-4-E2B-it-qat-q4_0-w4a16-ct` | 7.51 GB |
| **Repack, text only (used here)** | **6.56 GB** |
| Repack, int4 embeddings too (`-text-emb4`) | 2.83 GB |

Google's W4A16 export re-rounds every group of 32 weights onto a new grid. Served by vLLM on one v5e chip, the two builds decode at the same speed, 136.6 and 136.5 tokens per second, and the repack scores 0.678 against 0.655 on a 3,880-record test suite. The `-text-emb4` build is the smallest download, and the engine used here reads bf16 embedding tables, so the notebook uses `-text`.

---

#### At This Point You Should Have…

- A Google account with Colab access. A TPU v5e-1 runtime is listed as eligible on the free tier for this account.
- Nothing else. Every download is public and no Hugging Face token is needed.

---

#### Step 1 — Open the Notebook

Open [`gemma4_e2b_serving_v5e1.ipynb`](https://colab.research.google.com/github/xbill9/colab/blob/main/notebooks/gemma4_e2b_serving_v5e1.ipynb) in Colab. Pick **Runtime → Change runtime type → v5e-1 TPU**, then **Runtime → Run all**.

The notebook is built from a jupytext source, `src/gemma4_e2b_serving_v5e1.py`, and every version in the repo was executed end to end on a real v5e-1 before it was committed. The executed copy is in `runs/2026-11-17/`.

---

#### Step 2 — Read the Chip While the Model Downloads

The first cell starts the download in its own process and returns at once:

```python
download = subprocess.Popen(
    [sys.executable, "-c", DOWNLOAD, MODEL_ID, MODEL_REV],
    stdout=subprocess.PIPE,
    stderr=download_log,
    text=True,
)
```

```text
downloading xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text @ 315aec06 in the background (pid 2547)
```

While it runs, the next cell reports the software, the chip and the host:

```text
software
  Python        3.13.18
  jax           0.11.2
  jaxlib        0.11.2
  libtpu        0.0.50
  transformers  5.22.1
  huggingface_hub 1.33.0

TPU
  devices       1 x TPU v5 lite
  HBM           16.91 GB, 0.00 GB in use
  $TPU_ACCELERATOR_TYPE = v5e-1

host
  CPU           24 x AMD EPYC 7B13
  RAM           50.5 GB
  free disk     241 GB
```

Then two kernels show what the chip does:

```text
bf16 matmul 8192x8192: 6.17 ms  ->  178 TFLOP/s
1 GiB elementwise:     3.62 ms  ->  593 GB/s of HBM traffic
```

Decoding one token reads every weight from HBM, so the bandwidth figure is the one that sets decode speed.

---

#### 🔎 Tip: Start the Long Download First

A 6.56 GB checkpoint takes about a minute on Colab. Started in the first cell, it finishes while the version and chip cells run, and the load cell waited 0 seconds for it:

```text
checkpoint: 6.56 GB, done 63 s after the first cell (this cell waited 0 s for it)
```

The download runs in a separate Python process. When that process exits, the VM gets back the host RAM the downloader held.

---

#### Step 3 — Pick the Engine That Answers First

Start-up time decides the engine for a demo. Three were measured on this chip type:

| Engine | Time to first answer | Where it runs |
| :--- | :--- | :--- |
| [`tpu-jax`](https://github.com/xbill9/tpu-jax) (used here) | 117 s after Run all, download included | TPU |
| vLLM on TPU, `tpu-vllm-v5e1-2b` rig | 17 min from TPU grant, 16 min on a container restart | TPU |
| llama.cpp `llama-server` | 2.6 s from launch to healthy | host CPU |

`tpu-jax` is a Gemma 4 E2B decoder in pure JAX. It compiles only the shapes a request uses. llama.cpp starts fastest because it has nothing to compile, and it has no TPU backend, so on this runtime it uses the host CPU. Step 12 measures what that costs.

---

#### Step 4 — Load the Weights

The engine is cloned at a pinned commit. `dequant_at_load=True` unpacks the 4-bit weights to bf16 once, at load:

```python
ENGINE = JaxGemmaEngine(
    MODEL_ID, quant_mode="w4a16", max_model_len=MAX_MODEL_LEN, dequant_at_load=DEQUANT_AT_LOAD
)
ENGINE.load(local_dir=str(MODEL_DIR))
```

```text
engine: tpu-jax 4b9f8e9, dequant_at_load=True, 9.26 GB of weights on the chip, loaded in 17 s
HBM in use: 9.27 of 16.91 GB
```

On a Colab v5e-1 on 2026-10-21, the same engine and repack decoded at 130.8 tokens per second unpacked and 103.1 packed, with identical tokens, using 9.30 GB of HBM against 6.62 GB. The chip has room for the unpacked weights, so the notebook defaults to `True`.

---

#### Step 5 — Serve It Behind an OpenAI Endpoint

A standard-library HTTP server runs in a background thread, with routes shaped like vLLM's: `/health`, `/version`, `/v1/models`, `/metrics` and `/v1/chat/completions`. Any OpenAI client works against it.

Thinking is a chat-template switch, passed the way vLLM takes it, in `chat_template_kwargs`. The model writes its reasoning between the special tokens `<|channel>` and `<channel|>`, and the server returns that text as `reasoning_content` and the rest as `content`:

```python
class ReasoningSplitter:
    """Turn generated token ids into (kind, text) deltas, kind being 'reasoning' or 'content'."""

    LABEL = "thought\n"

    def feed(self, tid):
        if tid in (CH_OPEN, CH_CLOSE):
            self.kind = "reasoning" if tid == CH_OPEN else "content"
            self.ids, self.sent = [], 0
            return None
        self.ids.append(tid)
        text = TOK.decode(self.ids, skip_special_tokens=True)
        # the "thought" label and half-decoded characters are held back here
        delta, self.sent = text[self.sent :], len(text)
        return (self.kind, delta) if delta else None
```

One short request compiles the shape most chat turns use:

```text
warm-up: 'Hello! How can I help you today?' in 24 s, including the compile
OpenAI-compatible server on http://127.0.0.1:8000/v1
```

---

#### 🔎 Tip: Round `max_tokens` Up So One Program Serves Many Requests

The engine compiles one program per pair of padded prompt length and answer budget. Prompts pad to 64, 128, 256 and so on. The server rounds the answer budget up to 512, 2048 or 4096 tokens and cuts the stream at the requested `max_tokens`:

```python
budget = next((b for b in ANSWER_BUDGETS if b >= max_tokens), max_tokens)
```

A request for 16 tokens and one for 500 share one compiled program. Every response carries `x_new_shape: true` when serving it compiled a new one, so a slow first token can be traced to a compile.

---

#### Step 6 — The Chat Page

The chat page is `chat_ui.py` from the [`tpu-vllm-v5e1-2b`](https://github.com/xbill9/tpu-inference/tree/main/tpu-vllm-v5e1-2b) rig, written for vLLM. It runs unchanged against this server, because the routes and the metric names match. The notebook starts it and shows it in the cell output through Colab's port proxy:

```python
output.serve_kernel_port_as_iframe(UI_PORT, height=720)
```

Its "what is running" panel reads the server:

```text
  "model": {
    "id": "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text",
    "owned_by": "tpu-jax",
    "max_model_len": 4096
  },
  "version": "tpu-jax 4b9f8e9 (JAX 0.11.2)",

READY: model served and chat page up 117 s after the first cell started
```

The side panel has max tokens, temperature, top-k and a system prompt, and shows each reply's first-token time and decode rate. The page sends no thinking flag, so the server's `THINKING_DEFAULT` decides. Run `THINKING_DEFAULT = True` in a new cell and the next reply shows its reasoning in a collapsible section.

---

#### Step 7 — Compile the Chat Shapes in the Background

A conversation's first turns move through the 128, 256 and 512-token prompt buckets, and a thinking answer needs the 2048-token budget. A background thread compiles those shapes one at a time, holding the engine lock for each, so a chat message sent meanwhile waits for at most one compile:

```text
background compiles (waited 178 s for them here):
  prompt bucket  128, answer budget  512:  25.5 s
  prompt bucket  256, answer budget  512:  29.0 s
  prompt bucket   64, answer budget 2048:  23.1 s
  prompt bucket  128, answer budget 2048:  25.7 s
  prompt bucket  512, answer budget  512:  46.6 s
  prompt bucket  256, answer budget 2048:  29.2 s
```

Every sample after this point runs on a warm shape.

---

#### Step 8 — Thinking Off and On

Every sample goes through the official `openai` client. The same question, greedy, both ways:

```python
r_think = ask(
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    thinking=True,
    temperature=0,
    max_tokens=2048,
)
```

```text
[stop] prompt 43 tok, completion 391 tok, first token 0.19 s, decode 123.7 tok/s, total 3.3 s
[stop] prompt 50 tok, completion 857 tok, first token 0.17 s, decode 123.9 tok/s, total 7.1 s

thinking used 857 tokens against 391 without; wall time 7.1 s against 3.3 s
```

Both answers say the ball costs $0.05. Thinking more than doubles the tokens and the wall time, at the same decode rate.

---

#### Step 9 — Sampling, Seeds and Top-k

A fixed `seed` repeats a sampled answer exactly, and other seeds vary it:

```text
seed 7: Sweet treats after dark.
seed 7: Sweet treats after dark.
seed 8: Stay hungry after dark with our magical late-night bakery.
seed 9: Wake up your night with delicious treats from [Bakery Name]!

seed 7 repeated exactly: True
distinct answers across seeds 7, 8, 9: 3 of 3
```

`top_k` travels in the request body, because it is outside the OpenAI schema:

```text
greedy                      124.6 tok/s  (157 tokens)
temperature 1, top-k off    123.1 tok/s  (145 tokens)
temperature 1, top-k 64     119.4 tok/s  (145 tokens)
```

On the vLLM rig on the same chip type, top-k 64 took decode from 125 to 45 tokens per second. In this engine it costs about 4 percent.

---

#### Step 10 — `max_tokens` and Stop Sequences

```text
max_tokens=20:
Mercury: The smallest planet in our solar system and the closest to the Sun.
Venus: The

[length] prompt 26 tok, completion 20 tok, first token 0.16 s, decode 125.9 tok/s, total 0.3 s

stop=['Saturn']:
Mercury
Venus
Earth
Mars
Jupiter

[stop] prompt 26 tok, completion 12 tok, first token 0.01 s, decode 126.5 tok/s, total 0.1 s
```

"Saturn" arrives as more than one token. The server holds back any text that could be the start of a stop sequence until the next tokens settle it, so the answer ends cleanly at "Jupiter".

---

#### Step 11 — Multi-Turn Chat and JSON

The client keeps the history and sends it back each turn, with a system prompt setting the voice:

```text
You are in the North Atlantic Ocean. This is near the coast of North America.

The nearest land is the coast of Newfoundland, Canada. It is a relatively close point.

[stop] prompt 79 tok, completion 18 tok, first token 0.13 s, decode 118.8 tok/s, total 0.3 s
```

JSON by instruction, then parsed:

```text
{"name": "Ada", "city": "Lisbon", "year": 2019}

parsed: {'name': 'Ada', 'city': 'Lisbon', 'year': 2019}
```

The server does not constrain decoding to a schema, so the cell checks that the reply parses.

---

#### Step 12 — Long Prompts and the Context Limit

Each prompt length lands in a new bucket, so the first call compiles. The second call shows the time a user sees once the shape is warm:

```text
prompt tokens first call s warm first token s decode tok/s
          108         0.15              0.015          0.0
          668        50.78              0.042         58.5
         2588        41.84              0.230        123.6
         3868        43.15              0.232         80.8
```

The context limit is 4,096 tokens. With the weights unpacked, an 8,192-token prefill asks XLA for 16.77 GB against 15.75 GB free, so the server refuses longer prompts with a clear error and keeps serving:

```text
HTTP 400: prompt is 6414 tokens; this server's limit is 4095

server still answering: OK.
```

---

#### Step 13 — Call a Deployed Endpoint (Optional)

The same `ask()` helper takes any OpenAI client. Add a Colab secret named `REMOTE_OPENAI_BASE_URL`, for example `http://<tpu-ip>:8000/v1` for the vLLM rig, and section 4 repeats thinking off, thinking on and a 256-token decode against it. Without the secret it prints a skip line:

```text
no REMOTE_OPENAI_BASE_URL set: skipping the deployed-endpoint comparison
```

vLLM returns reasoning in its own field only when it is started with a reasoning parser (`--reasoning-parser gemma4`).

---

#### Step 14 — llama.cpp on the Host CPU

[`xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf) holds the same trained weights as a 2.62 GB GGUF. The notebook downloads llama.cpp release `b12061` and starts its server with the reasoning parser:

```python
llama = subprocess.Popen(
    [
        str(LLAMA_DIR / "llama-server"), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(LLAMA_PORT),
        "-c", str(MAX_MODEL_LEN), "-t", str(threads), "--jinja", "--reasoning-format", "deepseek", "--no-webui",
    ],
```

```text
llama.cpp b12061 and 2.62 GB GGUF fetched in 27 s
llama-server (24 threads) healthy 2.6 s after launch; first answer 3.8 s later
```

The same three requests, through the same client:

```text
llama.cpp CPU: thinking off                  43   392        1.79          3.8    stop
llama.cpp CPU: thinking on                   50   870        2.81          3.7    stop
llama.cpp CPU: greedy 256                    21   163        1.89          4.0    stop

greedy decode, 256 tokens: TPU 124.6 tok/s, CPU 4.0 tok/s (31.2x)
```

---

#### Compare and Contrast

| Gemma 4 E2B on a Colab v5e-1 runtime | Ready after | Decode tok/s | Thinking-on answer |
| :--- | :--- | ---: | ---: |
| `tpu-jax` on the TPU 🥇 | 117 s after Run all | 124.6 | 7.1 s |
| llama.cpp on the host CPU 🥈 | 2.6 s after launch | 4.0 | 230 s |
| vLLM on the TPU (rig, 2026-11-03) | 17 min from TPU grant | 125 | not run here |

---

#### So, Which One?

For a live demo on a Colab TPU, serve with `tpu-jax`: chatting in under two minutes, 123.7 to 124.6 tokens per second, and a vLLM-shaped API that existing clients and the rig's chat page use unchanged.

For many users at once, deploy vLLM, which batches requests, and point section 4 at it. This engine serves one request at a time.

Keep llama.cpp for machines with no accelerator. On this runtime it starts in seconds and decodes at 4 tokens per second.

---

#### Summary

The goal of this article was to serve Gemma 4 E2B on a hosted Colab TPU quickly enough for a live demo and to step through the common serving options. The key to the solution was overlapping the download with the chip cells, an engine that compiles only the shapes a request uses, and a background thread that compiles the chat shapes after the server is up. The Colab v5e-1 results were:

- 🟢 Server and chat page up 117 seconds after Run all
- 🟢 123.7 to 124.6 tokens per second, with thinking on or off
- 🟢 Thinking on uses 857 tokens against 391, returned as `reasoning_content`
- 🟢 Seeds repeat exactly; top-k 64 costs about 4 percent
- 🟢 Stop sequences end cleanly; over-limit prompts get an HTTP 400 and the server keeps answering
- 🟢 The vLLM rig's chat page works unchanged
- ⚠️ One request at a time; the context limit is 4,096 tokens
- ⚠️ A new prompt size compiles on first use, 23 to 51 seconds
- ❌ llama.cpp has no TPU backend: 4.0 tokens per second on the host CPU, 31.2x slower

Scope: Gemma 4 E2B-it QAT repack, text only, one Colab v5e-1 runtime on 2026-11-17, one run of the notebook through the Colab CLI, one request at a time, greedy decoding unless stated. The packed-against-unpacked comparison comes from a Colab v5e-1 run on 2026-10-21, the vLLM start-up and top-k figures from the `tpu-vllm-v5e1-2b` rig on 2026-11-03, and the repack-against-Google figures from vLLM on one v5e chip on 2026-09-29. Section 4, the deployed endpoint, was skipped in this run.

The strategy for serving Gemma 4 on a hosted Colab TPU was validated with an incremental step by step approach.

---

#### References

- Repository: https://github.com/xbill9/colab
- Notebook: https://github.com/xbill9/colab/blob/main/notebooks/gemma4_e2b_serving_v5e1.ipynb
- Executed run: https://github.com/xbill9/colab/tree/main/runs/2026-11-17
- Repack: https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text
- Exact GGUF: https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf
- tpu-jax engine: https://github.com/xbill9/tpu-jax
- vLLM rig and chat page: https://github.com/xbill9/tpu-inference/tree/main/tpu-vllm-v5e1-2b
- llama.cpp: https://github.com/ggml-org/llama.cpp
