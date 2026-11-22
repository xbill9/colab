# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# ## 3. Use it
#
# Type into the chat page above. Its side panel turns thinking on and off and shows each reply's speed,
# measured in the browser and by `llama-server`. From code, any OpenAI client works against
# `http://127.0.0.1:8000/v1` inside this runtime.
#
# When you are done, disconnect the runtime (Runtime, then Disconnect and delete runtime) so it stops
# using compute units.

# %%
# Cell 3: one request through the OpenAI-compatible endpoint
body = {
    "messages": [{"role": "user", "content": "Why is the sky blue? One sentence."}],
    "max_tokens": 128,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}
req = urllib.request.Request(
    f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
)
t0 = time.perf_counter()
reply = json.load(urllib.request.urlopen(req, timeout=600))
print(reply["choices"][0]["message"]["content"])
timings = reply["timings"]
print(
    f"\n{timings['predicted_n']} tokens in {time.perf_counter() - t0:.1f} s; the server decoded at {timings['predicted_per_second']:.1f} tokens/s"
)
