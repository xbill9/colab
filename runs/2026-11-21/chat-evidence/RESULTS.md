# Chat turns right after READY, Colab v5e-1, 2026-11-21

Each turn sends the whole conversation, as the chat page does.

## Before (`before_stall.txt`, the build at 5cbfc54)

Background compiles held the chip lock, so every message waited behind one:

```
READY: model served and chat page up 48 s after the first cell started
chat starts 49 s after the first cell
turn 1: prompt 18 tok -> bucket 64, 114 tok out, 27.9 s, new shape False, 77 s after the first cell
turn 2: prompt 153 tok -> bucket 256, 117 tok out, 30.6 s, new shape False, 107 s after the first cell
turn 3: prompt 287 tok -> bucket 512, 19 tok out, 47.8 s, new shape False, 155 s after the first cell
turn 4: prompt 325 tok -> bucket 512, 221 tok out, 62.8 s, new shape False, 218 s after the first cell
background compiles so far: [(128, 2048, 26.1), (256, 2048, 29.5), (512, 2048, 47.5), (1024, 2048, 60.8)]
```

## After (`chat_turns_final_output.ipynb`, with `JAX_LOG_COMPILES=1`)

No compile over 0.5 s during the first answer or any of the seven chat turns. Ready 69 s.

| turn | prompt tokens | out | seconds |
|---:|---:|---:|---:|
| 1 | 18 | 117 | 1.1 |
| 2 | 156 | 118 | 1.1 |
| 3 (thinking) | 302 | 536 | 4.6 |
| 4 (max_tokens 8000, capped) | 525 | 234 | 2.1 |
| 5 | 776 | 20 | 0.4 |
| 6 | 817 | 719 | 6.2 |
| 7 | 1,555 | 51 | 0.8 |

Two engine bugs, both fixed in the notebook's code, are checked in the same run:

- New tokens were written after the padding, and the sliding-window layers mask by cache slot, so a
  prompt padded by more than 512 lost sight of itself. A 709-token prompt (padded to 1,024) now
  answers "Biscuit" and a 4,112-token prompt (padded to 8,192) answers "pomegranate"; before the
  fixes they returned an empty answer and `<unused2118>`.
- Chunked prefill took the next-token logits from the last chunk even when that chunk was all
  padding. Greedy tokens from the 64-token-chunk path now equal the engine's one-pass prefill: 48 of
  48 on two prompts, and the 709-token answer.
