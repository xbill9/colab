# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Gemma 4 E2B on a Colab v5e-1: chat with the QAT repack in two minutes
#
# **Author:** William McLean, Google Developer Expert
#
# This notebook loads Gemma 4 E2B onto one Colab TPU v5e chip, serves it behind an
# OpenAI-compatible endpoint, and opens a chat page right here in the notebook. On a Colab v5e-1 on
# 2026-11-17 the page was up 117 to 123 seconds after **Run all**.
#
# ### The model
#
# [`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text)
# is an exact repack of Google's quantization-aware-trained (QAT) Gemma 4 E2B weights: the trained
# 4-bit levels and scales in the W4A16 format, text only, 6.56 GB. It is public and ungated.
#
# ### The engine
#
# [`tpu-jax`](https://github.com/xbill9/tpu-jax), a Gemma 4 E2B decoder in pure JAX. It compiles only
# the shapes a request uses, so the model answers after one short compile.
#
# ### How the notebook runs
#
# 1. **Start the download, then read the chip.** The checkpoint downloads in the background, with a
#    progress bar, while the next cells report on the TPU.
# 2. **Load and serve.** The engine loads, compiles one request shape, and starts the server and the
#    chat page. **From there you can chat with the model.**
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
