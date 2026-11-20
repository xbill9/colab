"""Every tensor tpu-jax builds from the safetensors checkpoint, against the same tensor from the GGUF,
both as the bf16 values the engine holds after dequant_at_load. Works in row chunks (low RAM)."""
import os, sys, json, struct, collections
import numpy as np
from gguf import GGUFReader
from gguf_map import hf_name, SKIP, q4_0_to_bf16_bits, f32_to_bf16_bits

st_dir, gguf_path = sys.argv[1], sys.argv[2]
CHUNK = 4096
DT = {"BF16": np.uint16, "F32": np.float32, "I32": np.int32, "I64": np.int64}
st = {}
for f in sorted(x for x in os.listdir(st_dir) if x.endswith(".safetensors")):
    p = os.path.join(st_dir, f)
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    hdr.pop("__metadata__", None)
    for k, v in hdr.items():
        a, b = v["data_offsets"]
        dt = DT[v["dtype"]]
        st[k] = (v["dtype"], np.memmap(p, dt, "r", offset=8 + n + a,
                                       shape=((b - a) // np.dtype(dt).itemsize,)).reshape(v["shape"]))

def st_rows(name, lo, hi):
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
used, rows, tot = set(), [], collections.Counter()
for t in r.tensors:
    if t.name in SKIP:
        continue
    h = hf_name(t.name)
    assert h is not None, f"unmapped GGUF tensor {t.name}"
    shape = tuple(int(x) for x in reversed(t.shape))
    nrows, ncols = (shape if len(shape) == 2 else (1, shape[0]))
    raw = np.asarray(t.data).reshape(nrows, -1)
    same = 0; maxdiff = 0.0; maxabs = 0.0
    for lo in range(0, nrows, CHUNK):
        hi = min(lo + CHUNK, nrows)
        if t.tensor_type.name == "Q4_0":
            g = q4_0_to_bf16_bits(raw[lo:hi], (hi - lo, ncols))
        else:
            g = f32_to_bf16_bits(raw[lo:hi].astype(np.float32))
        s, kind = st_rows(h, lo, hi)
        s = s.reshape(g.shape)
        same += int((g == s).sum())
        gf = (g.astype(np.uint32) << 16).view(np.float32); sf = (s.astype(np.uint32) << 16).view(np.float32)
        maxdiff = max(maxdiff, float(np.abs(gf - sf).max())); maxabs = max(maxabs, float(np.abs(sf).max()))
    n = nrows * ncols
    used.add(h)
    grp = "embedding" if "embed_tokens" in h else ("linear" if t.tensor_type.name == "Q4_0" else "norm/scalar")
    tot[grp, "n"] += n; tot[grp, "same"] += same; tot[grp, "tensors"] += 1
    rel = maxdiff / max(maxabs, 1e-30)
    tot[grp, "maxrel"] = max(tot[grp, "maxrel"], rel)
    rows.append((t.name, h, t.tensor_type.name, kind, n, same, rel))

def covered(k):
    for suf in (".weight_packed", ".weight_scale", ".weight_shape"):
        if k.endswith(suf):
            return k[: -len(suf)] + ".weight" in used
    return k in used
print("safetensors tensors with no GGUF counterpart:", sorted(k for k in st if not covered(k)) or "none")
for grp in ("linear", "embedding", "norm/scalar"):
    n, same = tot[grp, "n"], tot[grp, "same"]
    print(f"{grp:12s} {tot[grp,'tensors']:4d} tensors {n:>14,d} values  bit-identical {same/n:.4%}  "
          f"max|diff|/max|w| {tot[grp,'maxrel']:.2e}")
print("lowest bit-identical share:")
for w in sorted(rows, key=lambda x: x[5] / x[4])[:6]:
    print(f"  {w[0]:30s} {w[2]}/{w[3]:6s} {w[5]/w[4]:.4%}  maxrel {w[6]:.2e}")
json.dump(rows, open("compare_rows.json", "w"))
