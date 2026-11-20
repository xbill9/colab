# Context length on a Colab v5e-1, 2026-11-20

`ctx_experiment_output.ipynb`: the basic notebook's startup cells, then one engine per setting on the
same bf16 weights (9.26 GB). Each long prompt puts "The access code for the blue door is 7351." first
and asks for it last; greedy, 256-token budget. "Repeat" is the second identical request, after the
first one compiled. The engine's chunk step was rebuilt per call in this run (see below).

| setting | prompt | result | first call s | repeat first token s | decode tok/s | answer |
|---|---:|---|---:|---:|---:|---|
| 4096, one pass | 21 | ok | 16.53 | 0.01 | 132.0 | - |
| 8192, chunked 512, full KV | 21 | ok | 16.93 | 0.01 | 127.5 | - |
| 4096, one pass | 3,830 | ok | 45.18 | 0.23 | 94.3 | 7351 |
| 8192, one pass | 7,926 | CompileTimeHbmOom: 16.77G temporaries, 15.75G available | - | - | - | - |
| 8192, chunked 512 | 7,926 | ok | 79.60 | 1.90 | 86.5 | 7351 |
| 16384, chunked 512 | 15,800 | ok | 79.88 | 3.19 | 83.1 | 7351 |
| 32768, chunked 512 | 32,200 | ok | 84.33 | 7.59 | 74.0 | 7351 |

HBM peak over the whole run: 16.39 of 16.91 GB.

In the notebooks (`gemma4_e2b_advanced_v5e1_output.ipynb`, same date), with prompts up to 2,048
tokens read in one pass and the chunk step compiled once: warm first token 0.041 s at 668 tokens,
0.220 s at 2,588, 0.226 s at 3,868 and 1.885 s at 16,028. With every prompt of 512 or more chunked
and the step rebuilt per call (the run before): 1.605, 1.491, 1.732 and 3.165 s.
