# make-localllm-easier — run the best local LLM your GPU can handle, in one command

**The leanest way to run local AI: least CPU, RAM and GPU memory, smallest files, measured quality.**

`localllm` picks, downloads and runs the most accurate local AI model for your PC and your language, chosen from real
benchmark measurements, with llama.cpp tuned for AMD, NVIDIA, Intel and Apple GPUs.

```
pip install make-localllm-easier
localllm
```

That's it. `localllm` checks your GPU and RAM, picks the most accurate model we have *measured* for your language that
fits your card, downloads llama.cpp and the model, starts it with settings profiled op by op, and opens the chat page.
You also get an OpenAI-compatible API at `http://127.0.0.1:8080/v1` for any app that speaks it. Offline, private, free.

```
localllm chat       # chat right here in the terminal (Thai, Japanese, any language)
localllm doctor     # what this GPU is good for: model sizes, speed, how much text it can hold
localllm list       # every model we have measured, with scores per language
localllm serve      # API only (OpenAI, Anthropic, Ollama and Gemini formats), no browser
localllm route      # optional: mix in your own cloud key, compare local vs cloud
localllm eval       # score any running server in English + your language
```

## FAQ

**Which local LLM should I run on my GPU?** Run `localllm doctor`. It lists which model sizes fit your card (4B up to
120B MoE), at which quantization, how fast they should run, the most accurate measured model for your language, and
how much system RAM that model uses (est.).

**Can a 16 GB GPU run a 27B model?** Yes. Qwen3.8-27B at ~3.5 bits (12.2 GB) runs at ~50 tok/s on an RX 9070 XT and
keeps 81.8% on English Global-MMLU-Lite. gemma-4-26B-A4B (13.3 GB) runs at ~85 tok/s with similar accuracy.

**Is a 2-bit quantized model good enough?** Usually not for non-English use: 2-bit costs 8-13 accuracy points, and
Hindi, Arabic and Thai lose the most (13 points).

**Can it use a different local model for each message?** Yes, opt-in: `localllm serve --models auto` routes each
message to the model with the best measured score for its language and task (e.g. Chinese knowledge to Qwen3.8,
Chinese math to gemma-4) and says which model answered. On a 16 GB card a swap costs 4-9 s, so it only switches for a
gain of 3+ points; on 24 GB+ cards both models stay loaded and routing is instant.

**Which local model is better at math?** On MGSM (the same 250 word problems in 11 languages), gemma-4-26B-A4B QAT
scores 96.8 / 89.6 / 88.8 in English / Thai / Chinese vs 94.4 / 87.2 / 84.4 for Qwen3.8-27B Q3, at 1.8x the speed.
Measure any server yourself with `localllm eval --suites math --langs en,de,ja`.

**Which local model translates better?** On FLORES-101 (100 sentences each way, chrF++), gemma-4-26B-A4B QAT beats
Qwen3.8-27B Q3 in Hindi (+4.7), Arabic (+2.9), Japanese (+1.6) and Thai (+1.5); Qwen leads Chinese by 1.4. Try
`localllm eval --suites translate --langs th,ja,sw` (101 languages).

**Why is llama.cpp slow on my AMD (or Intel) GPU on Windows?** If Resizable BAR is off, llama.cpp's Vulkan backend puts
buffers in a 256 MB host-visible heap backed by system RAM and decode drops up to 1.7x. `localllm` sets
`GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` for you ([llama.cpp#27097](https://github.com/ggml-org/llama.cpp/issues/27097)).

**Can I chat with a local LLM in the terminal?** Yes: `localllm chat`. Answers stream as they're written, the
conversation is remembered, `/save` writes it to a file, `/think` shows the model's reasoning, Ctrl+C stops an answer.

**Can I use it as an Ollama, OpenAI, Anthropic or Gemini replacement?** Yes. One local endpoint at
`http://127.0.0.1:8080` speaks all four APIs, so existing apps and SDKs only need a new base URL. See
[docs/apis.md](docs/apis.md).

**Can it fall back to my cloud API key?** Only if you turn it on. Routing is off by default; with your own key in an
environment variable it sends a request to the cloud only when a rule says so (prompt too long, a cloud model asked for
by name, or the local model scores below your floor in that language) and tells you where each answer came from.

**How much RAM does a local LLM need?** With the model fully on the GPU, about 2-2.5 GB of system RAM with `localllm`
(vs ~9 GB growing with llama-server's defaults). `localllm doctor` prints the estimate for your PC.

**Does it work offline?** After the first download, yes. Nothing leaves your PC.

**Which languages are measured?** 23 languages on Global-MMLU-Lite, 44 countries' own exams on INCLUDE, plus Thai
(ThaiExam). `localllm eval --langs ...` measures any of them on your hardware.

## What `localllm doctor` tells you

```
GPU AMD Radeon RX 9070 XT  15.9 GB  (640 GB/s)    RAM 32 GB    language: th

Model sizes for this PC (whole model on the GPU = fast):
  [OK  ] 4B                       Q8 4.2 GB  ~90 tok/s (est.)
  [OK  ] 8B                       Q8 8.5 GB  ~45 tok/s (est.)
  [OK  ] 14B                      Q6 11.5 GB  ~33 tok/s (est.)
  [OK  ] 24-32B                   Q3 13.2 GB  ~29 tok/s (est.)
  [SLOW] 30B MoE (3B active)      Q4 18.0 GB with experts in RAM - works, ~10-25 tok/s
  [NO  ] 70B                      needs ~42.0 GB - too big for this PC

Best measured model for you: gemma4-26b-a4b-qat  (MoE with ~4B active params: fastest)
What it can do here:
  TH  real local school/licence exams   65.7% correct  <- your language
  holds ~78k tokens at once (~130 pages of text) next to the model
  uses ~34.7 GB of system RAM: ~0.3 GB embeddings/CPU-mapped + ~8.0 GB prompt cache + ~25.9 GB ctx checkpoints + ~0.5 GB host (est.)
  leaves ~0 GB of RAM free for other apps (est.)
  answers at ~85 tok/s
```

Speeds marked *est.* come from your card's memory bandwidth, calibrated on measured runs. Everything else is measured.

## Less RAM than running llama.cpp yourself (0.2)

llama-server's defaults keep a prompt cache of up to 8 GiB plus 32 conversation checkpoints in system RAM, so RAM keeps
growing while you chat. `localllm` sizes both to your PC. Same 30-turn chat, RX 9070 XT, 32 GB RAM:

| model | llama.cpp defaults | `localllm` | speed |
|---|---|---|---|
| Qwen3.8-27B Q3 | 9.30 GB RAM | **2.37 GB** | 35.2 tok/s both |
| gemma-4-26B-A4B QAT | 9.24 GB RAM | **2.21 GB** | ~85 tok/s both |

About **4x less RAM, same speed, ~15% less CPU per answer, ~0% CPU while idle.** On cards too small for gemma-4, its
experts can stay in RAM: 45 / 36 / 31 tok/s with 8 / 13 / 18 layers' experts off the GPU (12 / 10 / 8 GB cards).

## Smart routing between local models (experimental, opt-in)

```
localllm serve --models auto          # or --models qwen3.8-27b-q3,gemma4-26b-a4b-qat
```

Each message goes to the model with the best *measured* score for its language and task (math vs general). If all the
models fit in VRAM together (24 GB+ cards for the pair below), they all stay loaded and routing is instant. Otherwise
one model is on the GPU at a time and it only swaps when the other is at least 3 points better, because a swap costs
seconds.
Every answer says which model wrote it and why (`X-Localllm-Model` header; shown under each answer in `localllm chat`).

What we measured on an RX 9070 XT (16 GB) with 32 GB RAM, and why this is **not the default**:

| | Qwen3.8-27B Q3 | gemma-4-26B-A4B QAT |
|---|---|---|
| Knowledge, Chinese / Spanish / Japanese | **+5.5 / +3.7 / +2.4 pts** | |
| Knowledge, English / Thai / Hindi / Arabic | within 2 pts | within 2 pts, **1.8x faster** |
| Math (MGSM), English / Thai / Chinese | 94.4 / 87.2 / 84.4 | **96.8 / 89.6 / 88.8** |
| Translation (FLORES chrF++), Thai / Chinese / Japanese / Hindi / Arabic | 54.1 / 49.1 / 46.4 / 58.1 / 59.0 | **55.6** / 47.7 / **48.0 / 62.8 / 61.9** |
| Model swap (stop one, load the other) | 8.6 s (4.1 s with the page cache warm and 16 MiB upload buffers) | |

Example: Chinese *knowledge* questions go to Qwen (+5.5); Chinese *math* (+4.4) and translation (within 1.4) stay
on gemma. On a 16 GB card a
second model only pays off for Chinese and Spanish (Japanese +2.4 is inside the benchmark margin), and every
swap costs 4-9 s. Keeping both
models on the card by letting Windows page VRAM made **both** 4-5x slower, so that's not an option. Faster switching
is tracked in [#32](https://github.com/phonology024/make-localllm-easier/issues/32): LoRA adapters on one shared base, or
specialists small enough to sit next to the main model.

## Measured results (RX 9070 XT 16 GB, Windows 11, llama.cpp Vulkan)

Accuracy (%) on multiple-choice exams, zero-shot. **global** = Global-MMLU-Lite: the same 400 questions translated, so
languages compare like for like. **regional** = INCLUDE: real exams written in each country (ThaiExam for Thai).

| | Qwen3.8-27B Q3 (12.2 GB) | gemma-4-26B-A4B QAT Q4 (13.3 GB) | Qwen3.8-27B 2-bit (7.8 GB) |
|---|---|---|---|
| English | 81.8 | 82.2 | 74.2 |
| Chinese | 76.2 / 74.7 | 73.5 / 66.5 | 67.8 / 67.8 |
| Spanish | 80.2 / 76.8 | 74.5 / 75.2 | 70.8 / 69.2 |
| Japanese | 73.5 / 87.6 | 74.5 / 81.9 | 65.8 / 77.9 |
| Arabic | 70.8 / 71.2 | 71.5 / 73.6 | 60.8 / 57.2 |
| Hindi | 69.0 / 74.3 | 69.5 / 71.0 | 56.2 / 55.5 |
| Thai | – / 67.1 | – / 65.7 | – / 54.2 |
| **decode speed** | **50 tok/s** (MTP) | **85 tok/s** | 40 tok/s |

Cells are global / regional. Margins are about ±4 (global) and ±5 (regional) points at 95%, so `localllm` treats gaps
under 2 points as a tie and picks the faster model.

## Findings worth knowing

1. **2-bit costs 8-13 points, and lower-resource languages pay the most.** Hindi, Arabic and Thai lose 13; English,
   Chinese and Spanish about 8-9. A 177B MoE squeezed to 1.6 bits scored *below* a 27B at 3 bits.
2. **Build 2-bit files from the original weights, and don't expect a small calibration set to beat a good vendor
   build.** Re-quantizing an 8-bit file down to 2 bits cost Thai about 9 points (45.0 vs 54.2 for the same recipe made
   from BF16). A Thai or mixed-language importance matrix won most of that back (51-52 Thai) - but built from BF16, our
   mixed matrix was *worse* than Unsloth's own UD-IQ2_S in 3 of 4 languages by KL divergence (table below), so we are not
   publishing it. At ~3.5 bits the matrix made no difference (64.8 vs 64.6 Thai). Details in
   [#24](https://github.com/phonology024/make-localllm-easier/issues/24).
3. **AMD/Intel cards without Resizable BAR: the Vulkan fix is model-dependent - so measure.**
   `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` makes Qwen3.8-27B (hybrid DeltaNet, rewrites a 3 MB state per layer per token)
   1.64x faster on an RX 9070 XT with ReBAR off, but gemma-4-26B-A4B 7-9% *slower*. `localllm` applies it per model,
   and `localllm tune` measures it on your PC. Note: llama.cpp treats any value, even `0`, as on - unset it to turn it
   off. See [llama.cpp#27097](https://github.com/ggml-org/llama.cpp/issues/27097).
4. **Qwen3.8 GGUFs ship a multi-token-prediction head.** Drafting 2 tokens with it adds ~40% decode speed for free;
   drafting 3 is slower.
5. **The first run of a new llama.cpp build is slow** while the GPU driver compiles its shaders once (~15 s).

### How much each quant changes the model, per language

KL divergence from the Qwen3.8-27B Q8_0 reference (lower is better) and how often the most likely next token stays
the same, on held-out Wikipedia text (8 x 512 tokens per language; `tools/kld_per_language.py`):

| Quant | Size | Thai | Hindi | Arabic | English |
|---|---|---|---|---|---|
| UD-Q3_K_XL | 12.2 GB | 0.034 / 91.7% | 0.037 / 90.2% | 0.078 / 92.5% | 0.021 / 93.2% |
| UD-IQ2_S (Unsloth) | 7.8 GB | 0.188 / 81.5% | 0.224 / 77.5% | 0.291 / 84.4% | 0.111 / 86.0% |
| IQ2_S, our mixed imatrix, from BF16 | 7.8 GB | 0.210 / 81.8% | 0.215 / 77.1% | 0.365 / 82.1% | 0.137 / 83.6% |

Going from 3 to 2 bits multiplies the KL divergence 4-6x in every language, and top-token agreement ends 2-9 points
lower in Arabic, Thai and Hindi than in English - the same order as the benchmark losses.

### Trimming the vocabulary to your languages (research)

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
[vocab-trim workflow](.github/workflows/vocab-trim.yml)). Texts: the same 1,000 news/Wikipedia sentences per language
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

For the catalog models (estimate, not measured on the GPU yet): Qwen3.8-27B shares Qwen3.5's vocabulary, so about 60%
of it stays: the output matrix (~0.87 GB of VRAM) and the CPU-mapped embeddings (~0.51 GB of RAM) shrink by ~40%, and
the output layer's ~1.4 ms per token from the op profile in
[#27](https://github.com/phonology024/make-localllm-easier/issues/27) by ~0.6 ms, a few percent of decode.

Limitations: BPE vocabularies with merges only (Qwen, Llama 3, Gemma 4; SentencePiece/WordPiece files are refused). A
trimmed file no longer shares token ids with separate draft models, LoRA adapters or anything else that stores ids.
Measured on CPU with Q8_0 files; GPU speed and VRAM on the RX 9070 XT, and benchmark accuracy, are still to measure.

## How the benchmark works

`localllm eval` asks each question with thinking off and reads the log-probability of every answer letter from the first
generated token, then takes the most likely one. It's prompt processing only, so a language takes a few minutes, and the
result is deterministic. Data is downloaded at eval time from the original Apache-2.0 datasets
([Global-MMLU-Lite](https://huggingface.co/datasets/CohereLabs/Global-MMLU-Lite),
[INCLUDE](https://huggingface.co/datasets/CohereLabs/include-lite-44),
[ThaiExam](https://huggingface.co/datasets/typhoon-ai/thai_exam)) and never redistributed. It measures knowledge and
reasoning in multiple choice, not writing quality.

## Contributing

The catalog only grows with measurements. Run `localllm eval --langs en,<yours>` on your GPU and open a PR with
`~/.localllm/results.json` and your GPU name. Other languages' local exams are very welcome. See [ROADMAP.md](ROADMAP.md)
for what's next: using less system RAM (0.2), working alongside cloud provider APIs (0.3), a speed-only release (0.4), and per-language compression research (0.5).

## License

MIT. Models keep their own licenses; benchmark data keeps its own (Apache-2.0).
