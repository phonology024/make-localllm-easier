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
