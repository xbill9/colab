# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Gemma 4 E2B on a Colab v5e-1: chat with the QAT model in under a minute
#
# **Author:** William McLean, Google Developer Expert
#
# This notebook loads Gemma 4 E2B onto one Colab TPU v5e chip, serves it behind an
# OpenAI-compatible endpoint, and opens a chat page right here in the notebook. On a Colab v5e-1 on
# 2026-11-20 the page was up 46 to 53 seconds after the first cell started, in three runs.
#
# ### The model
#
# [`xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf)
# holds Google's quantization-aware-trained (QAT) Gemma 4 E2B weights as one GGUF file, 2.62 GB: every
# layer and both embedding tables in Q4_0, 32 trained 4-bit values and one step per block. The engine
# expands them to bf16 on the chip. The config, tokenizer and chat template (32 MB) come from
# [`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text).
# Both are public and ungated.
#
# ### The engine
#
# [`tpu-jax`](https://github.com/xbill9/tpu-jax), a Gemma 4 E2B decoder in pure JAX. It compiles only
# the shapes a request uses, and this notebook runs that compile during the download.
#
# ### How the notebook runs
#
# 1. **Start the download, then read the chip.** The GGUF downloads in the background, with a
#    progress bar, while the next cells report on the TPU.
# 2. **Load and serve.** The engine compiles one request shape during the download, puts the weights
#    on the chip, and starts the server and the chat page. **From there you can chat with the model.**
#
# The [advanced notebook](https://colab.research.google.com/github/xbill9/colab/blob/main/notebooks/gemma4_e2b_advanced_v5e1.ipynb)
# runs these same two sections, then steps through thinking on and off, sampling, stop sequences,
# long prompts, a deployed endpoint and a llama.cpp comparison.
#
# ### Before you start
#
# * **Runtime.** Runtime, then Change runtime type, then **v5e-1 TPU**.
# * **No token.** Every download is public. Nothing asks for a Hugging Face token.
# * **Safety.** Nothing is uploaded. The server and the chat page listen on `127.0.0.1` inside the
#   Colab VM; the page reaches your browser through Colab's own port proxy.
