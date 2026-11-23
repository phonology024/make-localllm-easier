# Changelog

## 0.8.1 (2026-11-23)

- Plain `localllm` opened a blank page at http://127.0.0.1:8080: it showed llama-server's web page, the gateway proxied
  only `/` and answered 404 for the files that page loads (`/_app/immutable/bundle.*.js`, icons, manifest). **Our own
  chat page is now the default** for `localllm` and `localllm serve` (saved chats, model picker, live VRAM/RAM; it does
  not depend on llama.cpp's web UI, which changes between builds). `--ui` still works and does nothing new;
  `--no-ui` serves llama-server's page as before, and its files are now proxied (by name: `/_app/`, icons, manifest,
  `sw.js`; every other path is still a 404), so that mode is not blank either. Key and Host checks are unchanged. The
  server `localllm mcp --model KEY` starts runs with `--no-ui` (#188)
- The chat page has a new look: black and gold in dark mode, white and gold in light mode (follows the system setting),
  in Poppins (SIL OFL 1.1, latin subset, 400/500/600, about 26 KB) that ships in the package and is served by localllm
  itself at `/fonts/poppins-*.woff2` (public like the page, Host check applies; CSP adds `font-src 'self'`). Thai and
  other scripts use the system font (Leelawadee UI / Noto Sans Thai). Gold marks only the Send/Unlock buttons, the
  thinking switch and download progress; fewer borders and colours elsewhere. Nothing loads from the internet. Same
  features, ids, requests and keyboard behaviour.
  The Models tab now says `localllm --model KEY` (no `--ui`: the page is the default)

## 0.8.0 (2026-11-23)

- **Python 3.10 or newer is required** (`requires-python = ">=3.10"`). 0.7.1 already did not run on 3.9: importing
  `localllm.catalog` fails there with a `TypeError` on `str | None`. `pip` now refuses to install on 3.9 instead of
  installing a package that cannot start; the single-file downloads need no Python at all.

- `localllm mcp`: an MCP server (stdio) so Claude, Cursor, Codex, VS Code and Gemini CLI can hand private or bulk work to
  the model on this PC: `ask_local_model`, `translate`, `summarize`, `status`. Localhost only (a server on your
  network needs `LOCALLLM_MCP_ALLOW_REMOTE=1` in its environment, never a tool argument), no telemetry, progress
  notifications on long answers. It starts `localllm serve` itself only for `--model KEY` that is already downloaded and
  never downloads anything. Install with `pip install "make-localllm-easier[mcp]"` (Python 3.10+). `server.json` for the
  MCP Registry, a `.mcpb` for Claude Desktop, [docs/mcp.md](docs/mcp.md) (#183)
- An answer that comes back empty (the model ends its turn after a few characters of "thinking"; seen on the 2-bit
  qwen3.8-27b-iq2 in Thai) is now asked again up to twice with a new seed and no prompt cache, then reported as an error;
  an empty answer that ran out of `max_tokens` is not retried. The MCP progress note no longer shows the time twice (#183)
- `localllm serve` no longer leaves llama-server running (and holding VRAM) when it is killed outright, and
  `localllm mcp --model KEY` stops the server it started together with llama-server when the client closes it. Windows:
  llama-server runs in a job object that ends with its parent; elsewhere the group gets SIGTERM (#183)

## 0.7.1 (2026-10-14)

- Release build: the Linux AppImage step installs `desktop-file-utils`, so 0.7.1 is the first release with the
  single-file `localllm` downloads (0.7.0 reached PyPI, but its binary build failed).

## 0.7.0 (2026-10-14, PyPI only)

Agents, a chat page, and the groundwork for 1.0. Since 0.6.1:

**New**
- `localllm --ui` / `serve --ui`: our own chat page with saved chats (plain files in `~/.localllm/sessions/`), a model
  picker with `--models`, live VRAM/RAM/spill; on the LAN it asks for the key (#76)
- `--profile lean|max`: lean is today's default; max fills the GPU with context and gives the prompt cache more RAM for
  agents and long documents; `doctor` shows both (#90)
- `localllm connect hermes|opencode|claude-code|codex|aider|cline|obsidian` with backup and `--undo` (#75, #89)
- `localllm savings`: tokens answered locally per month, prompt-cache reuse, $ at the prices you set (#97)
- `localllm eval --suites longctx`: needle retrieval at 8k/32k/64k tokens in 7 languages (#93)
- `localllm eval --suites summarize`: LR-Sum news summaries (CC BY 4.0) in 24 languages incl. Thai, chrF++ and
  ROUGE-L; summarize requests are routed by their own scores once a model has them (#43)
- `localllm tune` measures GPU board power: watts and joules per token for the kept settings, shown in `doctor`
  (AMD via the driver on Windows / hwmon on Linux, NVIDIA via nvidia-smi); `tune --prefer-energy` (#42)
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
- A download whose connection dropped was hashed as if complete, failed the checksum and was deleted; it now resumes
  (up to 5 times) from where it stopped
- `eval --suites longctx`: the three-needle question said "three digits" in Thai and Chinese ("ตัวเลขสามตัว",
  "三个数字"), and models answered 3-digit numbers; questions now ask for the numbers
- With `LOCALLLM_LLAMA_BUILD` set, the requested build is used even when another one is already unpacked
- Leaderboard page escapes text from shared reports
