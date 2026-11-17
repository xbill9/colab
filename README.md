Sample Notebooks for testing Google Colab extenion.

Mersenne Primes was fully generated with the Agent Manager in Antigravity

## Gemma 4 E2B on a Colab v5e-1

| Notebook | What it shows | Token? |
| --- | --- | --- |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xbill9/colab/blob/main/notebooks/gemma4_e2b_serving_v5e1.ipynb) [`gemma4_e2b_serving_v5e1`](notebooks/gemma4_e2b_serving_v5e1.ipynb) | The TPU and host, then the Gemma 4 E2B QAT repack ([`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text)) served by the pure-JAX [`tpu-jax`](https://github.com/xbill9/tpu-jax) engine behind an OpenAI-compatible endpoint, with the [`tpu-vllm-v5e1-2b`](https://github.com/xbill9/tpu-inference/tree/main/tpu-vllm-v5e1-2b) chat page in the notebook. Samples: thinking on and off, sampling and seeds, top-k cost, `max_tokens` and `stop`, system prompt and multi-turn, JSON, long prompts. Optional: the same calls against a deployed vLLM endpoint. Last: the exact GGUF build on llama.cpp on the host CPU, for start-up and speed | No |

Open it in Colab, pick **Runtime → Change runtime type → v5e-1 TPU**, then **Runtime → Run all**. The
chat page is up a few minutes in; the whole notebook takes 30 to 60 minutes.

To compare with a deployed vLLM server, add a Colab secret `REMOTE_OPENAI_BASE_URL`
(e.g. `http://<tpu-ip>:8000/v1`) and grant the notebook access.

#### Authoring

The layout and tooling follow [`xbill9/torchtpu-dev/colab`](https://github.com/xbill9/torchtpu-dev/tree/main/colab).
**Edit `src/*.py`, never `notebooks/*.ipynb`**: sources are [jupytext](https://jupytext.readthedocs.io)
`py:percent`, and the build adds the license cell, the Colab badge and the TPU accelerator setting.

```bash
make setup     # jupytext, nbformat, google-colab-cli
make build     # src/*.py -> notebooks/*.ipynb
make check     # fail if committed notebooks are stale
make verify    # run on a real Colab v5e-1 via the Colab CLI; executed copy lands in runs/<date>/
make sessions  # list live Colab runtimes (these cost compute units)
```
