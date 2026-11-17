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
# runs sections 1 and 2 of this one: the model served and the chat page up, in about two minutes.
# This notebook continues from there.
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
# in 7 to 16 seconds and compiled its first request in 23 to 40 seconds. vLLM on the same chip took 17 minutes from TPU grant
# to its first answer when measured on 2026-11-03, and 16 on a container restart (most of it compiling), which is why vLLM appears
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
# * **Time.** The chat page is up about two minutes after Run all. The whole notebook ran in 901 s,
#   about 15 minutes, on a Colab v5e-1 on 2026-11-17; the llama.cpp section is a third of that.
# * **Safety.** Nothing is uploaded. The servers listen on `127.0.0.1` inside the Colab VM.
