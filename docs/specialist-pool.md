# Specialist pool: design note (#29)

Status: design only, no code. It fixes the rules *before* anything is measured, so a specialist can't be added because
it looks good on one run. Numbers marked *measured* are on `main` (RX 9070 XT 16 GB, 32 GB RAM, llama.cpp b11476);
everything else is a plan.

## What the pool is for

`localllm` routes each message to the model with the best measured score for its language and task (0.6,
`--models auto`). Today every model in the pool is a *generalist*. A specialist is a model that is clearly better at one
task, kept next to the generalist or swapped in for that task only. The user always talks to a model that is good in
their language; a specialist that is weak in that language only does the task and hands the result back.

## Slots

| Slot | Candidate today | Where it runs | Size | Measured so far | Join gate passed? |
|---|---|---|---|---|---|
| Chat (generalist) | the catalog pick per PC and language | GPU | 2.7-13.3 GB | full suites, 7+ languages | - (it *is* the baseline) |
| Math | gemma-4-26B-A4B vs Qwen3.8-27B | GPU | 13.3 / 12.2 GB | MGSM: gemma leads by +2.4 en, +2.4 th, +4.4 zh | **no**: under the 5-point gate; both stay generalists |
| Translation | same pair | GPU | | FLORES chrF++: gemma +1.5 to +4.7 in th/ja/hi/ar, Qwen +1.4 in zh | **no** |
| Code | unmeasured small coders; gemma's CRUXEval (#65) | GPU | | `--suites code`/`codegen` exist; no coder model measured | not yet measured |
| Vision | the chat model's own projector (`--vision`) | GPU | +0.63-1.11 GB VRAM | MaXM suite exists (7 languages) | n/a: same model, no separate specialist |
| Embeddings | multilingual-e5-small | **CPU** | 126 MB | router 97.9% on the AI-written set (#40 checks humans) | **in**: no VRAM, so no budget conflict (router, `serve --embeddings` #158) |
| Speech-to-text | Whisper via babelscribe | GPU or CPU | unmeasured | nothing in this repo yet | not yet measured |

## The join gate (fixed now, applied to every candidate)

A candidate joins the pool for task *T* only if all of these hold, on `localllm eval` with the full suite:

1. **Clearly better at T.** It beats the current generalist pick on T by more than the benchmark margin: **+5 points**
   on %-scored suites (MGSM, CRUXEval, codegen pass@1, MaXM), **+3 chrF++** on translation. Ties go to the generalist
   (no swap, no extra memory).
2. **Not weak in the user's language.** If it is more than 2 points below the generalist on the user's language suites,
   it may still do T, but only in *hand-off* mode: the generalist reads the user's message, gives the specialist a
   task in English, and writes the answer in the user's language.
3. **Fits the budget** (next section): resident next to the generalist, or swapped per *task phase* (not per message)
   with the measured swap cost shown to the user, or on the CPU.
4. **Pays for its latency.** Swap time + its answer time must stay under the generalist's answer time + 10 s for the
   task, measured end to end on the routing test set.
5. **Same rules as the catalog:** pinned file + SHA-256, license recorded, quality floor (no post-training quant below
   ~2.7 bpw).

Results go into `catalog.py` like every model, so the router uses the specialist only for the tasks it won.

## VRAM budget

The CLI already decides "resident vs one at a time" with: weights + KV cache for the context + fixed caches, per model,
plus ~1 GB of driver and compute buffers once (`cli._start`). With the lean 8k context:

| | VRAM needed | Left on a 16 GB card (15.9) |
|---|---|---|
| gemma-4-26B-A4B QAT alone | 13.5 GB + 1 GB | **1.4 GB** |
| Qwen3.8-27B Q3 alone | 12.6 GB + 1 GB | **2.3 GB** |
| both resident | **27.1 GB** | does not fit: lazy-load (one at a time) |

So on a 16 GB card no GPU specialist can sit next to a 26-27B generalist except a projector (0.63-1.11 GB) or a
specialist under ~1.4-2.3 GB. Everything else is *swapped*: measured 8.4 s median without prefetch and **5.3 s with the
page-cache prefetch** (#63, `tools/swap_results.jsonl`, 12 swaps each). That is why switching happens per task phase
with hysteresis, and why a specialist must beat the generalist by a clear margin to be worth a swap.

Smaller generalists leave room: Qwen3.5-9B (~6.9 GB with its 8k KV and buffers) leaves ~9 GB on a 16 GB card, enough for
a GPU specialist of up to ~8 GB to stay resident (the generalist then wouldn't be the 26-27B pick, which is the trade
the gate has to show is worth it).

*Finding while writing this:* `docs/how-it-works.md` says both chat models stay loaded on "24 GB+ cards"; by the CLI's
own formula the pair needs ~27.1 GB at 8k, so on a 24 GB card it runs one at a time. That sentence should say 32 GB.

## Speech (babelscribe)

The speech slot is where the budget matters most: Whisper must run while the chat model is loaded. Plan:
1. Measure babelscribe's models on the test PC: VRAM, RAM, real-time factor for Thai, on GPU and on CPU.
2. Prefer CPU if the real-time factor stays under ~0.5 for short voice messages (no VRAM conflict, like e5).
3. Otherwise budget it like a GPU specialist: resident only next to a small generalist, else transcribe first, then load.
4. Demo target (from #29): record -> transcript -> answer in Thai, offline, with latency and RAM printed.

## Order of work

1. Code: measure two small coding models (codegen pass@1, CRUXEval) against the generalist pick; join only through the
   gate. (GPU, Outside Dock's queue)
2. Speech: the measurements in the section above. (GPU + CPU)
3. Only then: router support for hand-off mode (gate rule 2), if any specialist needs it.
