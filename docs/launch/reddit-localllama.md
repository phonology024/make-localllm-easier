# r/LocalLLaMA draft

**Title:**

> Measured: llama.cpp's defaults use ~9 GB system RAM over a 30-turn chat, 2.4 GB tuned. Plus which model wins per
> language on a 16 GB card

**Flair:** Resources (or Other)

**Body:**

I built a small launcher around llama.cpp that picks the model by *measured* scores in your language and runs it with
lean settings. Some numbers from one PC (RX 9070 XT 16 GB, 32 GB RAM, Windows 11, llama.cpp b11476 Vulkan):

**System RAM, same 30-turn chat**

| model | llama.cpp defaults | localllm | speed |
|---|---|---|---|
| Qwen3.8-27B Q3 | 9.30 GB | 2.37 GB | 35.2 tok/s both |
| gemma-4-26B-A4B QAT | 9.24 GB | 2.21 GB | ~85 tok/s both |

Most of the difference is llama-server's default host prompt cache (up to 8 GiB) and 32 context checkpoints; localllm
sizes both to the PC.

**Per language, 16 GB card** (Global-MMLU-Lite / INCLUDE / ThaiExam / MGSM, zero-shot)
- Thai: gemma-4-26B-A4B QAT (exams 65.7, math 89.6) at ~85 tok/s.
- Chinese knowledge: Qwen3.8-27B Q3 is +5.5 points, so with `--models auto` Chinese knowledge questions go to Qwen and
  math stays on gemma.
- 2-bit costs 8-13 points; Hindi, Arabic and Thai lose the most.

**Smaller cards and CPU-only**
- 8-12 GB: Qwen3.5-9B at 4 bit beats Qwen3.8-27B at 2 bit in every language measured (Thai exams 61.1 vs 54.2).
- CPU only (20 threads, DDR5): 23.7 tok/s for gemma-4-E2B, 12-13 for the 4B class, 6-7 for 9-12B.
- A 70B at ~2 bit, partly on the GPU, loses to the 26B MoE that fits (Thai 50 vs 78).

**AMD on Windows without Resizable BAR:** `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` made Qwen3.8-27B 1.64x faster but
gemma 7-9% slower, so it's set per model.

Install: `pip install make-localllm-easier`, then `localllm`. Or grab the single file for Windows/macOS/Linux from the
0.7.1 release (not code-signed yet). It gives you a chat page plus OpenAI/Anthropic/Ollama/Gemini APIs on 127.0.0.1,
and `localllm connect opencode|aider|hermes|claude-code|codex` points an agent at it.

Everything above is from **one** PC. If you have NVIDIA, Intel, Apple or another AMD card, `localllm tune`,
`localllm eval --langs en,<yours>` and `localllm report` (anonymous JSON, you review it first) put your GPU on the
leaderboard: https://phonology024.github.io/make-localllm-easier/leaderboard/

Repo (MIT): https://github.com/phonology024/make-localllm-easier. Method and raw data are in docs/benchmarks.md.
