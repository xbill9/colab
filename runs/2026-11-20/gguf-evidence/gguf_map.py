"""GGUF (Gemma 4 E2B, Q4_0 + F32) -> the safetensors names and orientation tpu-jax reads."""
import re
import numpy as np

P = "model.language_model."
TOP = {
    "token_embd.weight": "embed_tokens.weight",
    "per_layer_token_embd.weight": "embed_tokens_per_layer.weight",
    "per_layer_model_proj.weight": "per_layer_model_projection.weight",
    "per_layer_proj_norm.weight": "per_layer_projection_norm.weight",
    "output_norm.weight": "norm.weight",
}
BLK = {
    "attn_q": "self_attn.q_proj.weight", "attn_k": "self_attn.k_proj.weight",
    "attn_v": "self_attn.v_proj.weight", "attn_output": "self_attn.o_proj.weight",
    "attn_q_norm": "self_attn.q_norm.weight", "attn_k_norm": "self_attn.k_norm.weight",
    "ffn_gate": "mlp.gate_proj.weight", "ffn_up": "mlp.up_proj.weight", "ffn_down": "mlp.down_proj.weight",
    "inp_gate": "per_layer_input_gate.weight", "proj": "per_layer_projection.weight",
    "attn_norm": "input_layernorm.weight", "post_attention_norm": "post_attention_layernorm.weight",
    "ffn_norm": "pre_feedforward_layernorm.weight", "post_ffw_norm": "post_feedforward_layernorm.weight",
    "post_norm": "post_per_layer_input_norm.weight", "layer_output_scale": "layer_scalar",
}
SKIP = {"rope_freqs.weight"}  # the engine computes RoPE frequencies from config.json


def hf_name(g):
    if g in TOP:
        return P + TOP[g]
    m = re.fullmatch(r"blk\.(\d+)\.(.+)\.weight", g)
    if m and m.group(2) in BLK:
        return f"{P}layers.{m.group(1)}.{BLK[m.group(2)]}"
    return None


def f32_to_bf16_bits(x):
    """Round to nearest even, as JAX/XLA does for an f32 -> bf16 cast."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) >> 16).astype(np.uint16)


def q4_0_to_bf16_bits(raw, shape):
    """ggml block_q4_0 bytes -> bf16 bits [rows, cols]: fp16 d, then 16 bytes; low nibbles are
    elements 0..15, high nibbles 16..31, each stores q + 8. Value = bf16(q * d), the same single
    rounding the engine's own unpack (bf16 q times bf16 scale) gives."""
    rows, cols = shape
    b = np.asarray(raw, np.uint8).reshape(rows, cols // 32, 18)
    d = b[..., :2].copy().view(np.float16).astype(np.float32)  # [rows, nb, 1]
    qs = b[..., 2:]
    q = np.concatenate([qs & 0xF, qs >> 4], axis=-1).astype(np.int8) - 8
    return f32_to_bf16_bits(q.astype(np.float32) * d).reshape(rows, cols)


def tensor_bf16_bits(t):
    """gguf ReaderTensor -> (bf16 bits in HF [out, in] orientation)."""
    shape = tuple(int(x) for x in reversed(t.shape))  # GGUF lists ne0 (the inner axis) first
    if t.tensor_type.name == "Q4_0":
        return q4_0_to_bf16_bits(t.data, shape)
    if t.tensor_type.name == "F32":
        return f32_to_bf16_bits(np.asarray(t.data, np.float32).reshape(shape))
    raise TypeError(f"{t.name}: unsupported GGUF type {t.tensor_type.name}")
