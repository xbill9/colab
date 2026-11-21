# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# ## 3. Try it
#
# Type into the chat page under cell 2.4, or click one of its example prompts. Tick **Thinking** to
# see the model reason before it answers. **New** starts a fresh conversation, **Stop** cuts a reply
# short. The side panel shows each reply's first-token time and decode rate.
# * **From code.** The endpoint is `http://127.0.0.1:8000/v1` inside this runtime, and any OpenAI
#   client works against it:
#
# ```python
# from openai import OpenAI
#
# client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")
# reply = client.chat.completions.create(
#     model=MODEL_ID,
#     messages=[{"role": "user", "content": "Why is the sky blue? One sentence."}],
#     extra_body={"chat_template_kwargs": {"enable_thinking": False}},
# )
# print(reply.choices[0].message.content)
# ```
#
# * **More.** The [advanced notebook](https://colab.research.google.com/github/xbill9/colab/blob/main/notebooks/gemma4_e2b_advanced_v5e1.ipynb)
#   steps through every serving option with measurements.
#
# When you are done, disconnect the runtime (Runtime, then Disconnect and delete runtime) so it stops
# using compute units.

# %%
# Cell 3.1: one request through the OpenAI client, to check the endpoint from code
if OPENAI_INSTALL is not None and OPENAI_INSTALL.wait() != 0:
    raise RuntimeError(f"pip install openai failed:\n{(WORK / 'prep.log').read_text()[-2000:]}")
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")
t0 = time.perf_counter()
reply = client.chat.completions.create(
    model=MODEL_ID,
    messages=[{"role": "user", "content": "Why is the sky blue? One sentence."}],
    max_tokens=128,
    temperature=0,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(reply.choices[0].message.content)
print(f"\n{reply.usage.completion_tokens} tokens in {time.perf_counter() - t0:.2f} s")
