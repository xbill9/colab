"""Every tensor tpu-jax builds from the safetensors checkpoint, against the same tensor from the GGUF,
both as the bf16 values the engine holds after dequant_at_load. Works in row chunks (low RAM)."""
import os, sys, json, struct, collections
import numpy as np
from gguf import GGUFReader
from gguf_map import hf_name, SKIP, q4_0_to_bf16_bits, f32_to_bf16_bits

st_dir, gguf_path, src_dir = sys.argv[1], sys.argv[2], sys.argv[3]
CHUNK = 4096
DT = {"BF16": np.uint16, "F32": np.float32, "I32": np.int32, "I64": np.int64}
def open_dir(d):
  out = {}
  for f in sorted(x for x in os.listdir(d) if x.endswith(".safetensors")):
    p = os.path.join(d, f)
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    hdr.pop("__metadata__", None)
    for k, v in hdr.items():
        a, b = v["data_offsets"]
        dt = DT[v["dtype"]]
        out[k] = (v["dtype"], np.memmap(p, dt, "r", offset=8 + n + a,
                                       shape=((b - a) // np.dtype(dt).itemsize,)).reshape(v["shape"]))
  return out
st = open_dir(st_dir); src = open_dir(src_dir)

def st_rows(name, lo, hi, st=st):
    base = name[: -len(".weight")] if name.endswith(".weight") else name
    if base + ".weight_packed" in st:
        packed = np.asarray(st[base + ".weight_packed"][1][lo:hi])
        scale = (np.asarray(st[base + ".weight_scale"][1][lo:hi]).astype(np.uint32) << 16).view(np.float32)
        r, pk = packed.shape
        q = ((packed[:, :, None] >> (np.arange(8, dtype=np.int32) * 4)) & 0xF).reshape(r, pk * 8) - 8
        v = q.reshape(r, -1, 32).astype(np.float32) * scale[:, :, None]
        return f32_to_bf16_bits(v.reshape(r, pk * 8)), "packed"
    dt, a = st[name]
    a = a.reshape(-1, a.shape[-1]) if a.ndim > 1 else a.reshape(1, -1)
    a = np.asarray(a[lo:hi])
    return (f32_to_bf16_bits(a) if dt == "F32" else a), dt.lower()

r = GGUFReader(gguf_path)
tot = collections.Counter()
for t in r.tensors:
    h = hf_name(t.name)
    if t.name in SKIP or t.tensor_type.name != "Q4_0":
        continue
    shape = tuple(int(x) for x in reversed(t.shape)); nrows, ncols = shape
    raw = np.asarray(t.data).reshape(nrows, -1)
    grp = "embedding" if "embed_tokens" in h else "linear"
    for lo in range(0, nrows, CHUNK):
        hi = min(lo + CHUNK, nrows)
        g = q4_0_to_bf16_bits(raw[lo:hi], (hi - lo, ncols))
        s_, _ = st_rows(h, lo, hi)
        o, _ = st_rows(h, lo, hi, st=src)
        s_ = s_.reshape(g.shape); o = o.reshape(g.shape)
        of = (o.astype(np.uint32) << 16).view(np.float32)
        for tag, x in (("gguf", g), ("safetensors", s_)):
            xf = (x.astype(np.uint32) << 16).view(np.float32)
            tot[grp, tag, "same"] += int((x == o).sum())
            tot[grp, tag, "abs"] += float(np.abs(xf - of).astype(np.float64).sum())
        tot[grp, "n"] += g.size
        tot[grp, "srcabs"] += float(np.abs(of).astype(np.float64).sum())
for grp in ("linear", "embedding"):
    n = tot[grp, "n"]
    for tag in ("gguf", "safetensors"):
        print(f"{grp:10s} {tag:12s} bit-identical to source {tot[grp, tag, 'same']/n:.4%}   "
              f"mean |error| / mean |w| {tot[grp, tag, 'abs']/tot[grp, 'srcabs']:.3e}")
