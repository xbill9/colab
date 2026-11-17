Sample Notebooks for testing Google Colab extenion.

Mersenne Primes was fully generated with the Agent Manager in Antigravity

## Gemma 4 E2B on a Colab v5e-1

| Notebook | What it shows | Token? |
| --- | --- | --- |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xbill9/colab/blob/main/notebooks/gemma4_e2b_basic_v5e1.ipynb) [`gemma4_e2b_basic_v5e1`](notebooks/gemma4_e2b_basic_v5e1.ipynb) | **Start here.** The TPU and host, then the Gemma 4 E2B QAT repack ([`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text)) loaded by the pure-JAX [`tpu-jax`](https://github.com/xbill9/tpu-jax) engine, served behind an OpenAI-compatible endpoint, with a chat page ([`ui/chat_ui.py`](ui/chat_ui.py)) embedded in the notebook. About two minutes to a working chat | No |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xbill9/colab/blob/main/notebooks/gemma4_e2b_advanced_v5e1.ipynb) [`gemma4_e2b_advanced_v5e1`](notebooks/gemma4_e2b_advanced_v5e1.ipynb) | The basic notebook, then samples through the OpenAI client: thinking on and off, sampling and seeds, top-k cost, `max_tokens` and `stop`, system prompt and multi-turn, JSON, long prompts. Optional: the same calls against a deployed vLLM endpoint. Last: the exact GGUF build on llama.cpp on the host CPU, for start-up and speed | No |

Open it in Colab, pick **Runtime → Change runtime type → v5e-1 TPU**, then **Runtime → Run all**. The
chat page is up about two minutes in; the advanced notebook ran in about 15 minutes on 2026-11-17.

To compare with a deployed vLLM server, add a Colab secret `REMOTE_OPENAI_BASE_URL`
(e.g. `http://<tpu-ip>:8000/v1`) and grant the notebook access.

#### Drive Colab from an agent: the Colab API and MCP endpoint

Google's [Colab API and hosted MCP endpoint](https://medium.com/google-colab/the-google-colab-api-and-mcp-endpoint-for-developers-and-agents-65cfef497056)
create runtimes, run code on them and delete them, from a script or from an agent. This repo's
`make verify` uses the MCP endpoint to run the notebook on a fresh v5e-1. To set it up yourself:

1. **A credential with the `colaboratory` scope.** The Colab CLI's login carries it:
   `pip install google-colab-cli`, then `colab --auth oauth2 sessions` once. A plain
   `gcloud auth application-default login` token lacks the scope and every call is refused.
2. **A quota project** with the API enabled:
   `gcloud services enable colaboratory.googleapis.com --project <your-project>`.
3. **Check it:** `COLAB_QUOTA_PROJECT=<your-project> make specs` prints your Colab tier and which
   runtimes (CPU, GPU, TPU v5e-1, v6e-1) you can create.
4. **Add it to Claude Code.** This repo ships a `.mcp.json`, so the simplest way is to run `claude`
   in the cloned repo. Claude Code asks you to trust the folder and approve the `colab` server once;
   after that `/mcp` lists it as connected. Its `headersHelper` runs `tools/colab_token.py --header`,
   which supplies the token and the quota project (`COLAB_QUOTA_PROJECT`, else your active gcloud
   project). To have it in every project instead:

   ```bash
   claude mcp add-json colab --scope user '{"type": "http",
     "url": "https://colaboratory.googleapis.com/mcp",
     "headersHelper": "COLAB_QUOTA_PROJECT=<your-project> python3 /path/to/colab/tools/colab_token.py --header",
     "timeout": 700000}'
   ```

   The agent then has `list_runtime_specs`, `create_runtime`, `create_session`, `execute_code`,
   `get_runtime`, `delete_runtime` and four more. A runtime uses compute units until it is deleted,
   and an `execute_code` call must finish within 300 s (measured; the stream closes there).

| File | Role |
| --- | --- |
| `tools/colab_token.py` | prints a `colaboratory`-scoped token, or the request headers for Claude Code's `headersHelper` |
| `tools/colab_mcp.py` | a small MCP client: `python3 tools/colab_mcp.py tools`, `... call list_runtime_specs` |
| `tools/run_on_colab.py` | runs a notebook cell by cell on a new runtime via `execute_code`, then deletes the runtime |

#### Authoring

The layout and tooling follow [`xbill9/torchtpu-dev/colab`](https://github.com/xbill9/torchtpu-dev/tree/main/colab).
**Edit `src/parts/*.py`, never `notebooks/*.ipynb`**: `notebooks.json` lists the parts each notebook is built from, and sources are [jupytext](https://jupytext.readthedocs.io)
`py:percent`, and the build adds the license cell, the Colab badge and the TPU accelerator setting.

```bash
make setup     # jupytext, nbformat, google-colab-cli
make build     # src/*.py -> notebooks/*.ipynb
make check     # fail if committed notebooks are stale
make verify    # run on a fresh Colab v5e-1 via the MCP endpoint; executed copy lands in runs/<date>/
make verify-cli  # the same through the Colab CLI (colab exec)
make specs     # Colab tier and the runtimes you can create
make sessions  # list live Colab runtimes (these cost compute units)
```
