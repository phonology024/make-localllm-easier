# Benchmarks

> Moved here from the README on 2026-10-14 so the README fits on one screen. Nothing was removed.

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
6. **Pruning experts (REAP) costs the most where the calibration data is thinnest.** GLM-4.7-Flash 30B-A3B vs Cerebras'
   REAP-pruned 23B-A3B, same Q4 quant: English lost 6-12 points, Thai math fell from 70.4% to 15.6%, Thai and Hindi
   translation lost about 25 chrF++ points
   ([#83](https://github.com/phonology024/make-localllm-easier/issues/83), `tools/runner_lab_glm.json`).
7. **A bigger model squeezed onto the card loses to a smaller one that fits.** Llama-3.3-70B at ~2 bit (IQ2_XXS, 54 of
   80 layers on the GPU) vs gemma-4-26B-A4B on the same questions: Thai 50 vs 78, Hindi 58-63 vs 69-70, Arabic 56-58 vs
   69-73, MGSM 75-80 vs 90-95 ([#86](https://github.com/phonology024/make-localllm-easier/issues/86)).
8. **A 120B MoE runs on 16 GB VRAM + 32 GB RAM, slowly.** gpt-oss-120b (63 GB, MXFP4) loads memory-mapped; with the
   experts of 31 of 36 layers in RAM it answers at 9.1 tok/s, with all of them in RAM 5.0 tok/s; at 28 the card spills
   ([#82](https://github.com/phonology024/make-localllm-easier/issues/82)).

## Chat with your files: does `/attach` find the passage? (#113)

`localllm eval --suites docqa --langs en,th,ja,hi,ar,zh,es` measures the retrieval step of `/attach`, with no chat
model and on the CPU only.
- **The documents:** per language, 10 documents of 100 consecutive FLORES-101 devtest sentences, in paragraphs of 5,
  each about 23 passages once indexed.
- **The needles:** six "secret number" sentences per document, each under a different name. They are hard negatives
  for each other: the same words, another name.
- **The questions:** one per needle, as a person would ask it ("What is the secret number of Falcon?"), so 60 per
  language.
- **The pipeline:** the real reader, chunker, index and multilingual-e5-small.
- **Scoring:** recall@1 means the right passage comes first; recall@5 means it is among the five the model gets.
  The brackets are 95% Wilson intervals.

`tools/docqa_pdf.py` prints the same documents to PDF with Edge and reads them back through pypdf, as people's PDFs
are. A passage counts when it holds the needle's name and number, on both sides.

| lang | TXT recall@1 | TXT recall@5 | PDF recall@1 | PDF recall@5 | questions |
|---|---|---|---|---|---|
| en | 93.3% [84.1-97.4] | 100% [94.0-100] | 88.3% [77.8-94.2] | 100% [94.0-100] | 60 |
| th | 80.0% [68.2-88.2] | 100% [94.0-100] | 80.0% [68.2-88.2] | 100% [94.0-100] | 60 |
| ja | 86.7% [75.8-93.1] | 100% [94.0-100] | 91.7% [81.9-96.4] | 100% [94.0-100] | 60 |
| hi | 86.7% [75.8-93.1] | 100% [94.0-100] | 86.7% [75.8-93.1] | 98.3% [91.1-99.7] | 60 |
| ar | 91.7% [81.9-96.4] | 100% [94.0-100] | **63.3% [50.7-74.4]** | **93.3% [84.1-97.4]** | 60 |
| zh | 73.3% [61.0-82.9] | 100% [94.0-100] | 73.3% [61.0-82.9] | 100% [94.0-100] | 60 |
| es | 95.0% [86.3-98.3] | 100% [94.0-100] | 96.7% [88.6-99.1] | 100% [94.0-100] | 60 |

**What this means.**
- **The answer is almost always in what the model gets.** Recall@5 is 100% for text files in every language
  measured.
- **First place is less sure.** Chinese ranks the right passage first 73% of the time, Spanish 95%. The model reads
  all five, so this matters less than recall@5.
- **PDFs behave like text, except Arabic.**
  - **Fixed so far.** Arabic PDFs started at 36.7% recall@1 and 83.3% recall@5. The fonts' glyph maps turn some
    Arabic letters into their Persian and Urdu forms, and lines that mix directions glue Arabic and Latin words
    together. Both are now repaired on mangled pages only, which brought Arabic to the numbers above.
  - **Still lost.** Word order on lines that mix Arabic with Latin names or numbers. For Arabic documents, DOCX or
    TXT keeps every word in order.
- **A chunker bug the suite caught.** The first run found a Thai needle cut in two by the chunker: Thai has no
  sentence punctuation, so long runs were packed word by word to the limit with no overlap. Chunks of unpunctuated
  text now repeat their last words in the next chunk, and Thai recall@5 went from 98.3% to 100%.

Setup: Windows 11, CPU only, multilingual-e5-small Q8_0, chunker version 3, 2026-10-15. Thai recall@1 read 86.7% in a
run before that chunker change and 80.0% after it, 4 questions apart, inside both intervals.

## Smaller cards and CPU-only PCs (#38)

Same tests as the catalog models, full sets, llama.cpp b11476. Quality doesn't depend on the GPU, so it was measured on
the RX 9070 XT; *CPU tok/s* is decode speed with no GPU at all (`-ngl 0`, 20 threads, DDR5) on the same PC.

| model | GB | EN know. | TH exams | TH math | HI know. | AR know. | ZH know. | JA exams | ES know. | GPU tok/s | CPU tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B UD-Q4_K_XL | 5.6 | 80.5 | 61.1 | 83.2 | 61.0 | 65.8 | 74.5 | 82.3 | 76.0 | 42 | 7.3 |
| gemma-4-12B QAT UD-Q4_K_XL | 6.3 | 74.8 | 59.6 | 90.8 | 62.2 | 66.5 | 66.5 | 79.5 | 73.2 | not measured | 6.2 |
| gemma-4-E4B UD-Q4_K_XL | 4.8 | 71.5 | 52.9 | 87.6 | 57.8 | 62.5 | 63.5 | 69.5 | 66.2 | ~105 | 12.5 |
| Qwen3.5-4B UD-Q4_K_XL | 2.7 | 75.5 | 51.2 | 81.2 | 54.2 | 63.0 | 69.0 | 70.7 | 70.5 | 78 | 12.8 |
| gemma-4-E2B UD-Q4_K_XL | 3.0 | 63.8 | 38.4 | 72.4 | 48.2 | 48.5 | 54.5 | 55.4 | 56.5 | ~160 | 23.7 |

For reference, gemma-4-26B-A4B on the same tests: 82.2 / 65.7 / 89.6 / 69.5 / 71.5 / 73.5 / 81.9 / 74.5.

- **8-12 GB cards: Qwen3.5-9B at 4 bit beats Qwen3.8-27B at 2 bit** in every language measured (Thai exams 61.1 vs
  54.2, Hindi 61.0 vs 56.2) and is the pick there now. gemma-4-12B QAT is the stronger one at math (Thai 90.8%).
- **Thai drops fastest as models shrink**: exams 61 (9B) -> 51-53 (4B) -> 38 (E2B), while English stays at 64-80.
- **CPU only works, slowly**: 23.7 tok/s for E2B, 12-13 for the 4B class, 6-7 for 9-12B. Enough to chat, not for agents.
- The ~ GPU speeds for gemma E2B/E4B were taken while ~2-2.5 GB of their per-layer embeddings sat in system RAM,
  which these models do by design.
- Raw rows: `tools/runner_results.jsonl` (ids `38-*`). gemma-4-12B's Spanish math/translation and its speeds are missing
  (its run was cut short); it joins the catalog once they are measured.

## How much each quant changes the model, per language

KL divergence from the Qwen3.8-27B Q8_0 reference (lower is better) and how often the most likely next token stays
the same, on held-out Wikipedia text (8 x 512 tokens per language; `tools/kld_per_language.py`):

| Quant | Size | Thai | Hindi | Arabic | English |
|---|---|---|---|---|---|
| UD-Q3_K_XL | 12.2 GB | 0.034 / 91.7% | 0.037 / 90.2% | 0.078 / 92.5% | 0.021 / 93.2% |
| UD-IQ2_S (Unsloth) | 7.8 GB | 0.188 / 81.5% | 0.224 / 77.5% | 0.291 / 84.4% | 0.111 / 86.0% |
| IQ2_S, our mixed imatrix, from BF16 | 7.8 GB | 0.210 / 81.8% | 0.215 / 77.1% | 0.365 / 82.1% | 0.137 / 83.6% |

Going from 3 to 2 bits multiplies the KL divergence 4-6x in every language, and top-token agreement ends 2-9 points
lower in Arabic, Thai and Hindi than in English - the same order as the benchmark losses.

## How the benchmark works

`localllm eval` asks each question with thinking off and reads the log-probability of every answer letter from the first
generated token, then takes the most likely one. It's prompt processing only, so a language takes a few minutes, and the
result is deterministic. Data is downloaded at eval time from the original Apache-2.0 datasets
([Global-MMLU-Lite](https://huggingface.co/datasets/CohereLabs/Global-MMLU-Lite),
[INCLUDE](https://huggingface.co/datasets/CohereLabs/include-lite-44),
[ThaiExam](https://huggingface.co/datasets/typhoon-ai/thai_exam)) and never redistributed. It measures knowledge and
reasoning in multiple choice, not writing quality.

## Measured on contributors' PCs

Generated from shared reports by `tools/merge_reports.py`:

<!-- gpu-table:start -->
| GPU | VRAM | RAM | OS | llama.cpp | model | decode tok/s | kept settings | scores |
|---|---|---|---|---|---|---|---|---|
<!-- gpu-table:end -->
