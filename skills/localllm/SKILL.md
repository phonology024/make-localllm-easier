---
name: localllm
description: Run and use a local LLM on this PC with the `localllm` CLI (make-localllm-easier) - check what the GPU can run, start an OpenAI-compatible server at http://127.0.0.1:8080/v1, point a coding agent at it and undo that, score a model in the user's language. Use when the user asks to run a model locally, use their GPU instead of a cloud API, check what fits their GPU, connect Claude Code / Codex / opencode / aider / hermes / cline / Obsidian to a local model, or install localllm.
---

# localllm

`localllm` picks the most accurate local model that fits this PC's GPU and the user's language, downloads llama.cpp and
the model (SHA-256 checked), and serves it at `http://127.0.0.1:8080` in OpenAI, Anthropic, Ollama and Gemini formats.

Not installed yet (`localllm --version` fails)? Follow
https://raw.githubusercontent.com/phonology024/make-localllm-easier/main/docs/install-for-agents.md and its rules.

## Ground rules

- **Ask before a download or a write.** Say how big it is and where it goes, and wait for a yes. Never pass `--yes` /
  `-y` and never set `LOCALLLM_YES=1`: the model download prompt belongs to the user.
- **No sudo.** localllm never needs root or admin.
- **Leave the workspace alone.** Data lives in `~/.localllm` (`$LOCALLLM_HOME`). Never use `connect --path` to write
  into the user's project.
- **Keep it on this PC.** Don't use `--host 0.0.0.0` / `::` (LAN access) unless the user asks for it by name.
- **Hand these to the user:** deleting models or `~/.localllm`, anything with a price or a cloud key (`savings
  --price`, `route` to a cloud model), and `report` (it opens a prefilled public GitHub issue).

## Look (no model download)

| command | what it does | writes |
|---|---|---|
| `localllm --version` | installed version | nothing |
| `localllm --help` | every command and flag | nothing |
| `localllm list` | every measured model with its size and scores | nothing |
| `localllm doctor` | GPU, VRAM, RAM, which model sizes fit, expected speed, the pick for this language | first run only: the pinned llama.cpp build (tens of MB) into `~/.localllm/llama.cpp/` |
| `localllm savings` | tokens answered locally this month | nothing |
| `curl -s http://127.0.0.1:8080/v1/models` | is a server up, and which model | nothing |

## Run

| command | when |
|---|---|
| `localllm` | first run: pick, download (asks first), serve, open the chat page |
| `localllm serve` | just the API, no browser; add `--profile max` for agent work (largest context that fits) |
| `localllm serve --models auto` | route each message to the best local model for its language/task |
| `localllm chat` | chat in the terminal; inside it, `/attach FILE` answers from PDF/DOCX/TXT/MD with sources (a PDF asks before its one 0.4 MB download) |
| `localllm launch <tool>` | start the server, connect the tool, run it, undo the config when it exits |

`localllm` and `localllm serve` keep running in the foreground. Start them as a background process, or ask the user to
run them in their own terminal. The server is ready when `GET /v1/models` answers.

Use it from code: base URL `http://127.0.0.1:8080/v1`, any API key (e.g. `local`), model name from `/v1/models`.

## Connect a coding agent (edits that tool's user config)

```
localllm connect <tool> --dry-run     # show the new config; change nothing
localllm connect <tool>               # only after the user says yes; the original is kept as <file>.localllm-backup
localllm connect <tool> --undo        # put the original back
```

Tools: `claude-code`, `codex`, `opencode`, `hermes`, `aider`, `cline` (prints settings, writes nothing), `obsidian`
(needs `--path VAULT`). Run the server with `--profile max` for agents. If you are the agent being connected, tell the
user your own settings will change.

## Measure

`localllm eval --langs en,th` scores whatever server runs at `--url` (default `http://127.0.0.1:8080`). It downloads
the test sets into `~/.localllm/bench/` on first use, so ask first. Other suites: `--suites math`, `translate`, `code`,
`longctx` (start the server with `--profile max`), `summarize`. Add `--limit 50` for a quick run.

## When something fails

1. `localllm doctor`: does it see the GPU? "no GPU found" usually means a missing Vulkan/GPU driver. Tell the user
   which one; don't install drivers yourself.
2. Port 8080 busy: add `--port 8081` (and `--url http://127.0.0.1:8081` for `connect` / `eval`).
3. Still stuck: `localllm bugreport` writes one zip into `~/.localllm` with secrets removed and lists what's inside before
   asking. The user attaches it to an issue at https://github.com/phonology024/make-localllm-easier/issues
