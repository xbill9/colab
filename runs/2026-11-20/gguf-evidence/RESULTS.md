# GGUF against the safetensors repack, 2026-11-20

Output captured verbatim from runs on a 16-thread Linux host (CPU only), with the files below.

- GGUF: `xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf` @ `74b2c92`, sha256 `419db9a6bf3bc15d...`, 2,620,370,912 B
- Repack: `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text` @ `315aec06`
- Source: `google/gemma-4-E2B-it-qat-q4_0-unquantized` (bf16)

Every value is compared as the bf16 the engine holds after expanding (`bf16(q * step)`).

## `compare.py` — GGUF against the repack

```
safetensors tensors with no GGUF counterpart: none
linear        276 tensors  1,876,819,968 values  bit-identical 91.7398%  max|diff|/max|w| 7.41e-03
embedding       2 tensors  2,751,463,424 values  bit-identical 96.8924%  max|diff|/max|w| 2.15e-03
norm/scalar   262 tensors        285,987 values  bit-identical 100.0000%  max|diff|/max|w| 0.00e+00
```

## `compare_src.py` — both against Google's full-precision source

```
linear     gguf         bit-identical to source 96.7704%   mean |error| / mean |w| 1.881e-04
linear     safetensors  bit-identical to source 90.0339%   mean |error| / mean |w| 7.819e-04
embedding  gguf         bit-identical to source 96.8924%   mean |error| / mean |w| 1.872e-04
embedding  safetensors  bit-identical to source 100.0000%   mean |error| / mean |w| 0.000e+00
```

## `gguf_experiment_output.ipynb` — one Colab v5e-1 runtime, both routes

GGUF 2.62 GB downloaded in 25.0 s and loaded in 9.2 s; safetensors 6.56 GB downloaded in 62.7 s and
loaded in 16.9 s. Greedy tokens: the bat-and-ball prompt identical (391 tokens); "Why is the sky
blue?" agrees for the first 109 tokens; the key-value cache prompt with thinking on agrees for the
first 718.

## `tok_lite.py` — tokenizer without transformers

Against transformers 5.19.0 `AutoTokenizer` on the repack's tokenizer files:

```
load: lite 0.72 s, transformers 4.76 s
chat template: 16/16 identical
decode: 600/600 identical
encode same: True | bos 2 2 | channel ids [True, True]
```
