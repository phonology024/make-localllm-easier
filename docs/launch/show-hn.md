# Show HN draft

**Title** (under 80 characters):

> Show HN: localllm – picks the local LLM that fits your GPU, measured in your language

**URL:** https://github.com/phonology024/make-localllm-easier

**First comment** (post it yourself right after submitting):

> Hi HN. I run local models on a 16 GB AMD card in Thai and English, and kept hitting two problems: the model
> "everyone recommends" was chosen on English benchmarks, and llama.cpp's defaults let system RAM grow to ~9 GB over a
> long chat.
>
> localllm is a small Python tool around llama.cpp. `pip install make-localllm-easier && localllm` checks the GPU and
> RAM, picks the most accurate model we have measured *in your language* that fits the card, downloads it (SHA-256
> checked) and starts a chat page plus an OpenAI/Anthropic/Ollama/Gemini-compatible API on 127.0.0.1. No Python?
> 0.7.1 has single-file downloads for Windows, macOS and Linux.
>
> Things we measured that surprised us:
> - Same 30-turn chat on Qwen3.8-27B: 9.30 GB of system RAM with llama.cpp's defaults, 2.37 GB with the settings
>   localllm uses, same 35 tok/s. Most of it is the default prompt cache and checkpoints.
> - Quantizing to 2 bits costs 8-13 points, and Hindi, Arabic and Thai lose the most. A 9B model at 4 bits beats a 27B
>   at 2 bits in every language we measured (Thai exams 61.1 vs 54.2).
> - A 70B squeezed to 2 bits onto a 16 GB card loses to a 26B MoE that fits (Thai 50 vs 78).
>
> Honest limits: every number comes from one PC (RX 9070 XT, Windows 11, Vulkan). NVIDIA, Intel and Apple should
> work but are unmeasured, which is why `localllm report` exists: it makes an anonymous JSON (no names, paths or keys)
> that adds your GPU to the leaderboard (https://phonology024.github.io/make-localllm-easier/leaderboard/). The binaries
> aren't code-signed yet. MIT licensed.
>
> I'd love results from other GPUs and from languages we haven't covered, and to hear what you'd want it to pick
> differently.
