# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # llama.cpp on a Colab TPU: Gemma 4 E2B in three cells
#
# **Author:** William McLean, Google Developer Expert
#
# This notebook runs llama.cpp's `llama-server` on one Colab TPU v5e chip, through
# [`ggml-tpu`](https://github.com/xbill9/ggml-tpu), an unofficial ggml backend that lowers each llama.cpp
# graph to one XLA program. It serves Gemma 4 E2B behind an OpenAI-compatible endpoint and opens a chat
# page in the notebook.
#
# **Unofficial.** `ggml-tpu` is an independent project, not part of ggml-org/llama.cpp and not made or
# endorsed by Google.
#
# ### What runs
#
# * **llama.cpp** `llama-server` at commit `7182ef71e`, and **`libggml-tpu.so`**, from one prebuilt
#   7 MB tarball ([release v0.2.0](https://github.com/xbill9/ggml-tpu/releases/tag/v0.2.0)). Nothing
#   compiles C++ here. llama.cpp carries one 11-line patch,
#   [`llama.cpp-kv-full.patch`](https://github.com/xbill9/ggml-tpu/blob/main/scripts/llama.cpp-kv-full.patch):
#   with `LLAMA_KV_FULL=1` every request attends the whole context, so the graph keeps one shape.
# * **The model:** [`xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf),
#   Gemma 4 E2B with Google's quantization-aware-trained 4-bit weights stored exactly as Q4_0, 2.62 GB.
#   Public and ungated.
# * **The TPU runtime:** `libtpu`, the library JAX uses, loaded directly by the backend. This notebook
#   never imports JAX: only one process can hold the TPU, and it has to be `llama-server`.
#
# ### Before you start
#
# * **Runtime.** Runtime, then Change runtime type, then **v5e-1 TPU**. Then Runtime, Run all.
# * **No token.** Every download is public.
# * **No compile during a chat.** The server feeds prompts one token at a time (`-ub 1`) at a fixed
#   context, so it compiles its one program while it starts. On a Colab v5e-1 on 2026-11-22, eight
#   chat turns took 0.2 to 9.6 s each, read prompts at 325 to 499 tokens per second and decoded at
#   104 to 108 tokens per second.
