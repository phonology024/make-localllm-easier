# Changelog

## 0.7.0 (not released yet)

Agents, a chat page, and the groundwork for 1.0. Since 0.6.1:

**New**
- `localllm --ui` / `serve --ui`: our own chat page with saved chats (plain files in `~/.localllm/sessions/`), a model
  picker with `--models`, live VRAM/RAM/spill; on the LAN it asks for the key (#76)
- `--profile lean|max`: lean is today's default; max fills the GPU with context and gives the prompt cache more RAM for
  agents and long documents; `doctor` shows both (#90)
- `localllm connect hermes|opencode|claude-code|codex|aider|cline|obsidian` with backup and `--undo` (#75, #89)
- `localllm savings`: tokens answered locally per month, prompt-cache reuse, $ at the prices you set (#97)
- `localllm eval --suites longctx`: needle retrieval at 8k/32k/64k tokens in 7 languages (#93)
- `/status` in `localllm chat` (#79); `localllm report` and a public leaderboard (#73, #99)
- Vision through all four APIs (`--vision`), tool calling for Ollama and Gemini, LAN mode with an API key,
  idle unload and `/stay` with `--models`, the embedding task router
- Single-file executables for Windows, macOS and Linux plus an AppImage on every release; winget and Homebrew
  manifests (#46). The first run shows `doctor` and asks before the model download (`--yes` skips it)

**Trust**
- Every download is checked against a pinned SHA-256; llama.cpp pinned to b11476 (#92)
- `doctor` says when your llama.cpp build differs from the one the numbers came from; bump process with
  `tools/smoke_build.py` (#95)
- Stability contract and deprecation policy (#98); CI checks that README samples match real output (#80)

**Fixed**
- With `LOCALLLM_LLAMA_BUILD` set, the requested build is used even when another one is already unpacked
- Leaderboard page escapes text from shared reports
