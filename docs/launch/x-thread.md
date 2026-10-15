# X / Twitter thread draft

Each post is under 280 characters. Attach `docs/img/ram-30-turns.png` to post 1 and `docs/img/models-tab.png` to
post 4.

**1/**
Same 30-turn chat with Qwen3.8-27B:
llama.cpp defaults: 9.30 GB system RAM
localllm: 2.37 GB
Same speed (35 tok/s).
localllm picks the local LLM that fits your GPU, measured in your language, and runs it lean. Open source, MIT 🧵

**2/**
Where the RAM goes: llama-server keeps a host prompt cache of up to 8 GiB plus 32 context checkpoints by default.
localllm sizes both to your PC. One command:
pip install make-localllm-easier && localllm

**3/**
"Best model" depends on your language. 2-bit quants cost 8-13 points, and Hindi, Arabic and Thai lose the most.
On 8-12 GB cards, a 9B at 4 bit beats a 27B at 2 bit in every language we measured (Thai 61.1 vs 54.2).

**4/**
The Models tab shows what fits your PC (whole on the GPU, part in RAM, or too big), with exam scores in your
language and a download button. Chat page and CLI in 7 languages, Arabic right to left.

**5/**
No Python? 0.7.1 has single-file downloads for Windows, macOS and Linux.
Every number so far is from one PC (RX 9070 XT). Run `localllm report` to add your GPU to the leaderboard:
https://phonology024.github.io/make-localllm-easier/leaderboard/

**6/**
Repo, method and raw data: https://github.com/phonology024/make-localllm-easier
