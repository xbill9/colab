# Start llama-server from the fresh build with the given flags, run chat turns like the chat page
# (history resent each turn), report per-turn wall time, compiles during the turn, and decode rate.
import glob, json, os, re, subprocess, sys, time, urllib.request, importlib.util
CFG = json.loads(sys.argv[1])
BIN = sorted(glob.glob("/content/ggml-tpu-*-linux-x86_64"))[-1] if CFG.get("bin") is None else CFG["bin"]
MODEL = open("/content/model_path.txt").read().strip()
LIBTPU = glob.glob(os.path.join(os.path.dirname(importlib.util.find_spec("libtpu").origin), "**", "libtpu.so"), recursive=True)[0]
PORT = 8000
LOG = f"/content/server_{CFG['tag']}.log"
env = dict(os.environ, GGML_BACKEND_PATH=f"{BIN}/libggml-tpu.so", GGML_TPU_LIBTPU=LIBTPU, GGML_TPU_PRECISION="DEFAULT",
           GGML_TPU_VERBOSE="1", **CFG.get("env", {}))
args = [f"{BIN}/llama-server", "-m", MODEL, "-dev", "TPU0", "-ngl", "99", "-fa", "off", "-c", str(CFG["ctx"]),
        "--host", "127.0.0.1", "--port", str(PORT), "--parallel", "1", "--jinja", "--alias", "g"] + CFG.get("args", [])
subprocess.run("pkill -f llama-server; sleep 1", shell=True)
t0 = time.perf_counter()
srv = subprocess.Popen(args, env=env, stdout=subprocess.DEVNULL, stderr=open(LOG, "w"))
while True:
    try:
        if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200: break
    except Exception: pass
    if srv.poll() is not None: print(open(LOG).read()[-3000:]); sys.exit(1)
    time.sleep(0.5)
print(f"[{CFG['tag']}] server ready {time.perf_counter()-t0:.1f} s")
def compiles():
    return [float(m) for m in re.findall(r"(?:compiled|loaded cached) program \d+ .* in ([\d.]+) s", open(LOG).read())]
def chat(msgs, max_tokens, think=False):
    body = json.dumps({"model": "g", "messages": msgs, "max_tokens": max_tokens, "temperature": 0.7,
                       "chat_template_kwargs": {"enable_thinking": think}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))
TURNS = [("Hi! Who are you, in one sentence?", False),
         ("Explain a key-value cache in a transformer, simply.", False),
         ("Now give me three bullet points on why it speeds up decoding.", False),
         ("Write a short story about a lighthouse keeper and a cat.", False),
         ("Is 1001 prime? Think it through.", True),
         ("Summarise everything we talked about so far in five bullets.", False),
         ("Write a long poem about the sea, at least 40 lines.", False),
         ("Translate the first four lines of that poem into French.", False)]
if CFG.get("warm"):
    w0 = time.perf_counter(); chat([{"role": "user", "content": "hi"}], 4)
    print(f"[{CFG['tag']}] warm request {time.perf_counter()-w0:.1f} s, compiles {compiles()}")
msgs = []
for i, (q, think) in enumerate(TURNS[:CFG.get("turns", len(TURNS))]):
    msgs.append({"role": "user", "content": q})
    before = len(compiles()); a = time.perf_counter()
    r = chat(msgs, 2048, think)
    dt = time.perf_counter() - a
    c = compiles()[before:]
    tm = r.get("timings", {})
    msgs.append({"role": "assistant", "content": r["choices"][0]["message"]["content"]})
    print(f"[{CFG['tag']}] turn {i+1}: prompt {r['usage']['prompt_tokens']} tok (new {tm.get('prompt_n')}), out {r['usage']['completion_tokens']} tok, "
          f"{dt:.1f} s, prefill {tm.get('prompt_per_second', 0):.1f} tok/s, decode {tm.get('predicted_per_second', 0):.1f} tok/s, "
          f"compiles {len(c)} totalling {sum(c):.1f} s")
print(f"[{CFG['tag']}] all compiles: {len(compiles())}, {sum(compiles()):.1f} s")
srv.terminate(); srv.wait()
