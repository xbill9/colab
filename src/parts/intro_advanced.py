# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Gemma 4 E2B on a Colab v5e-1, advanced: the QAT repack served, with thinking on and off
#
# **Author:** William McLean, Google Developer Expert
#
# This notebook serves Gemma 4 E2B from one Colab TPU v5e chip and puts it through the requests a
# real deployment sees: thinking on and off, streaming, sampling and seeds, stop sequences, token
# limits, system prompts, multi-turn chat, JSON output and long prompts. Every request goes through
# an OpenAI-compatible endpoint, so the same client code also talks to a vLLM deployment and to
# llama.cpp.
#
# The [basic notebook](https://colab.research.google.com/github/xbill9/colab/blob/main/notebooks/gemma4_e2b_basic_v5e1.ipynb)
# runs sections 1 and 2 of this one: the model served and the chat page up, in under a minute.
# This notebook continues from there.
#
# ### The model
#
# Google trained every Gemma 4 E2B weight onto a 4-bit grid (quantization-aware training, QAT).
# [`xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf)
# stores those trained values as one GGUF file: every layer and both embedding tables in Q4_0, 32
# 4-bit values and one step per block, text only. It is the smallest build, so the download is
# shortest. The engine expands it to bf16 on the chip. The config, tokenizer and chat template (32 MB)
# come from the W4A16 repack below. Both are public and ungated.
#
# | Build | Download |
# | --- | ---: |
# | Google's `google/gemma-4-E2B-it-qat-w4a16-ct` | 8.32 GB |
# | The repack, `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct` | 7.51 GB |
# | The repack, text only (`-text`) | 6.56 GB |
# | **The Q4_0 GGUF (used here)** | **2.62 GB** |
#
# Google's W4A16 export re-rounds every weight onto a new grid; the repack keeps the trained one.
# Served by vLLM on one v5e chip the two decode at the same speed (136.6 and 136.5 tokens per second)
# and the repack scores 2.4 points higher on a 3,880-record test suite. Measured against Google's
# full-precision QAT checkpoint on 2026-11-20, 96.8% of the GGUF's layer values are bit-identical,
# against 90.0% of the `-text` repack's: the GGUF stores each block's step in fp16, the repack in bf16.
# The repack's embedding tables are an exact bf16 copy; 96.9% of the GGUF's are bit-identical.
#
# ### The serving engine, and why this one
#
# The goal is a model you can talk to soon after **Run all**. The engine is
# [`tpu-jax`](https://github.com/xbill9/tpu-jax), a Gemma 4 E2B decoder in pure JAX. It compiles only
# the shapes a request uses, and XLA needs only the weights' shapes to compile, so this notebook
# compiles the first request's shape during the download. On a Colab v5e-1 on 2026-11-20 that compile
# took 26 seconds and the GGUF's weights reached the chip 7 seconds after it. vLLM on the same chip took 17 minutes from TPU grant
# to its first answer when measured on 2026-11-03, and 16 on a container restart (most of it compiling), which is why vLLM appears
# here as an optional remote endpoint rather than as the local server.
#
# ### How the notebook runs
#
# 1. **Start the download, then read the chip.** The 2.62 GB GGUF downloads in the background
#    while the first cells report on the TPU.
# 2. **Load and serve.** The engine compiles one request shape during the download, puts the weights
#    on the chip, and starts an
#    OpenAI-compatible server and a browser chat page. The cell prints how long all of this took.
#    **From here you can test the model yourself.**
# 3. **Samples.** Each common serving option, called through the OpenAI client.
# 4. **A deployed endpoint (optional).** The same calls against a vLLM server you run elsewhere.
# 5. **llama.cpp on the host CPU.** The same GGUF file, for a start-up and speed comparison.
# 6. **Summary** of every number measured in this run.
#
# ### Before you start
#
# * **Runtime.** Runtime, then Change runtime type, then **v5e-1 TPU**.
# * **No token.** Every download is public. Nothing asks for a Hugging Face token.
# * **Time.** The chat page was up 46 to 53 seconds after the first cell started, in three runs on
#   a Colab v5e-1 on 2026-11-20. On 2026-11-17 the whole notebook ran in 901 s, about 15 minutes; the
#   llama.cpp section is a third of that.
# * **Context.** Prompt plus answer can be up to 32,768 tokens; section 3.7 times prompts up to
#   16,000 tokens.
# * **Safety.** Nothing is uploaded. The servers listen on `127.0.0.1` inside the Colab VM.
