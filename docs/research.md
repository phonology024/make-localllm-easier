# Research

> Moved here from the README on 2026-10-14 so the README fits on one screen. Nothing was removed.

Bigger models on a 16 GB card: [research-big-models.md](research-big-models.md). Thai-specific notes: [research-th.md](research-th.md).

## Trimming the vocabulary to your languages

`tools/trim_vocab.py` drops the tokens of scripts you don't use, removes the BPE merges that build them (so other text
still encodes, in smaller pieces) and slices every per-token tensor: embeddings, output layer, Gemma's per-layer
embeddings. Text written in the kept scripts tokenizes exactly as before.

```
pip install gguf numpy
python tools/trim_vocab.py trim  MODEL.gguf OUT.gguf --langs th,en --dry-run     # what would go, sizes saved
python tools/trim_vocab.py trim  MODEL.gguf OUT.gguf --langs th,en               # keep every Thai/Latin/symbol token
python tools/trim_vocab.py check MODEL.gguf OUT.gguf --texts DIR --require th,en # same tokens, decodes back exactly
python tools/trim_vocab.py agree MODEL.gguf OUT.gguf --texts DIR --require th,en # same next-token pick
```

Measured on real models with `--langs th,en` (GitHub CPU runner, 4 threads, llama.cpp b11487, Q8_0;
[vocab-trim workflow](../.github/workflows/vocab-trim.yml)). Texts: the same 1,000 news/Wikipedia sentences per language
(UD PUD), 400 GSM8K problems, 400 KB of Python.

| | Qwen3-0.6B | Qwen3.5-2B (Qwen3.8's 248k vocabulary) | Gemma 4 E2B |
|---|---|---|---|
| vocabulary kept | 106,149 of 151,936 (69.9%) | 148,738 of 248,320 (59.9%) | 162,451 of 262,144 (62.0%) |
| file | 0.60 -> 0.55 GiB | 1.87 -> 1.67 GiB | 4.63 -> 3.59 GiB |
| Thai, English, code, math tokens | identical | identical | identical |
| same most likely next token as the original | 100% (2,040 positions each) | 100% | 100% |
| probability the original gave dropped tokens, Thai text | 0.19% (p99 1.6%) | 0.88% (p99 4.2%) | 0.08% (p99 1.0%) |
| decode speed, CPU | 51.5 -> 55.8 tok/s (+8%) | 17.0 -> 19.0 tok/s (+12%) | 12.9 -> 14.3 tok/s (+11%) |
| with `--keep-top 64000` | 0.50 GiB, 59.7 tok/s | 1.49 GiB, 21.0 tok/s | 2.56 GiB, 15.5 tok/s |
| ... same most likely next text, en / code / math / th | 90.7 / 92.8 / 96.7 / 97.5% | 92.7 / 90.4 / 96.9 / 97.5% | 81.6 / 85.1 / 87.9 / 96.8% |

Prompt speed doesn't change. Two more runs on other runner CPUs gave +9.6 to +12.8% for the lossless trim, so read the
speed-ups as +-5%.
ASCII alone is 57-62% of these vocabularies, so the lossless trim saves about a third; on Gemma 4 E2B that is 1 GB,
because its per-layer embeddings have a row per token too. `--keep-top N` goes further by dropping the rarest
ASCII/symbol tokens (latest BPE merges first; the languages' own letters are never capped), which splits rare English
words and code identifiers into more pieces: English +3.8-7.6% tokens, code +2.4-4.6%, math +1.5-3.0%, Thai <= +0.1%.
That is not lossless: compared where both models' tokens line up, the most likely next text differs at 3-18% of
positions (worst on Gemma 4 English), so the cap needs a benchmark before anyone uses it; the default trim doesn't.
Text in a dropped script (Chinese, Hindi, Arabic, ...) still works and decodes back exactly, but takes 2.5-9x more
tokens.

Measured on the catalog models (RX 9070 XT 16 GB, Windows, llama.cpp Vulkan b11476, `--langs th,en`, same launch
settings for both files; held-out Wikipedia texts for `check`):

| | Qwen3.8-27B UD-Q3_K_XL | gemma-4-26B-A4B QAT UD-Q4_K_XL |
|---|---|---|
| vocabulary kept | 148,738 / 248,320 (59.9%) | 162,451 / 262,144 (62.0%) |
| file | 12.24 -> 11.71 GiB | 13.27 -> 13.12 GiB |
| `check`: English | identical | identical |
| `check`: Thai | +0.5% tokens (Chinese/Japanese names inside the Thai text) | +0.4% tokens (same cause) |
| `check`: Hindi / Arabic | ~5x tokens, decode back exactly | 3.5-7.7x tokens, decode back exactly |
| llama-bench pp512 / tg128 | 881 -> 899 / 36.2 -> 36.5 tok/s | 993 -> 1012 / 108.9 -> 105.7 tok/s (within noise) |
| decode in llama-server (real chats) | 53.1 -> 54.8 tok/s (+3%) | 122.9 -> 138.9 tok/s (+13%) |
| VRAM | 12.83 -> 12.66 GB | 13.02 -> 13.12 GB |
| llama-server process RAM | 1.08 -> **0.66 GB (-39%)** | 1.41 -> **0.99 GB (-30%)** |
| eval th regional / th math / en global / en math | 66.4 / 87.2 / 81.8 / 94.4 -> **identical** | 65.7 / 89.6 / 82.2 / 96.8 -> **identical** |

So on real weights the trim costs no accuracy in the kept languages and frees 30-39% of the server's system RAM.
gemma-4's +13% in llama-server but not in llama-bench is probably sampling, which scans the whole vocabulary on the CPU
for every token (llama-bench doesn't sample); not verified yet. Real Thai text often embeds Chinese or Japanese names, so
East/Southeast Asian users may want `--langs th,en,zh,ja`: on Qwen3.8 that keeps 83.9% of the vocabulary (0.21 GiB
smaller) and Thai then tokenizes identically, with the same top-1 next token at 100% of positions.

Limitations: BPE vocabularies with merges only (Qwen, Llama 3, Gemma 4; SentencePiece/WordPiece files are refused). A
trimmed file no longer shares token ids with separate draft models, LoRA adapters or anything else that stores ids.
`--keep-top` caps are measured on CPU with Q8_0 files only.
