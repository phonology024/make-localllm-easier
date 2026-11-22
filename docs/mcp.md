# Use localllm from an AI app (MCP)

`localllm mcp` is an [MCP](https://modelcontextprotocol.io) server. An AI app that supports local MCP servers (Claude
Desktop, Claude Code, Cursor, VS Code, Gemini CLI, Codex) can hand work to the model on **your** PC: private text you do
not want in a cloud service, or bulk jobs (translate or summarize fifty documents) that would use up the app's own
budget. The text goes from the app to `localllm` on this computer and the answer comes back; nothing else is sent
anywhere, and there is no telemetry.

## Set it up once

```
pip install "make-localllm-easier[mcp]"      # the MCP SDK is an optional extra; it needs Python 3.10+
localllm                                     # first run: picks the model for your PC, asks before downloading it
```

Then add the server to your app. Everything runs the same command, `localllm mcp` (with [uv](https://docs.astral.sh/uv/)
you can skip the install: `uvx --from "make-localllm-easier[mcp]" localllm mcp`).

| App | How to add it |
|---|---|
| Claude Desktop | download `localllm.mcpb` from the [latest release](https://github.com/phonology024/make-localllm-easier/releases/latest) and double-click it (needs [uv](https://docs.astral.sh/uv/) and the setup above) |
| Claude Code | `claude mcp add localllm -- localllm mcp` |
| Cursor | add the JSON below to `~/.cursor/mcp.json` |
| VS Code (Copilot) | `.vscode/mcp.json`: the same entry under `"servers"` with `"type": "stdio"` |
| Gemini CLI | add the JSON below to `~/.gemini/settings.json` |
| OpenAI Codex CLI | `codex mcp add localllm -- localllm mcp`, then under `[mcp_servers.localllm]` in `~/.codex/config.toml` set `tool_timeout_sec = 3600` and `startup_timeout_sec = 120` (the defaults, 60 s and 10 s, are too short for a long answer or a model that is still loading) |

```json
{
  "mcpServers": {
    "localllm": { "command": "localllm", "args": ["mcp"], "timeout": 3600000 }
  }
}
```

(`timeout` is in milliseconds and only some apps read it. With uv: `"command": "uvx", "args": ["--from",
"make-localllm-easier[mcp]", "localllm", "mcp"]`.) Web-only chat apps (chatgpt.com, grok.com, ...) can only reach
servers on the internet, not your computer, so they cannot use this.

## Tools

| Tool | What it does |
|---|---|
| `ask_local_model(prompt, system?, max_tokens?)` | one answer from the model on this PC |
| `translate(text, target_lang)` | translate; with `localllm serve --models auto` the model measured best for that language answers |
| `summarize(text, length?)` | summarize in the language of the text; `length` is `short`, `medium`, `long` or your own wording |
| `status()` | is a model running, which one, VRAM and RAM in use, speed of the last answer; starts nothing |

Every answer comes back with the model that wrote it and `truncated: true` if it ran out of `max_tokens`. Long answers
send a progress notification every 5 seconds so the app does not time out. The local model is smaller than the app's
own: ask the app to check what matters.

## Which server it talks to

It talks to the localllm already running on `http://127.0.0.1:8080` (`localllm`, `localllm serve`). If nothing answers:

- **No `--model`** (the default): the tools say "run `localllm` once". Nothing is started or downloaded.
- **`--model KEY`** (for example `"args": ["mcp", "--model", "qwen3.5-4b-q4"]`, keys in `localllm list`): the server
  starts `localllm serve KEY` itself and stops it when the app closes, but only if that model is **already
  downloaded** and llama-server is installed. It never downloads anything; the first load of a model takes a while.

A different port: `--port 9000`. The environment variable `LOCALLLM_URL` or `--url` points it at another server.

## What it will not do

- It only talks to a server on this computer (127.0.0.1, `localhost`, `::1`). A server on your network is used only if
  **you** put `LOCALLLM_MCP_ALLOW_REMOTE=1` (and `LOCALLLM_API_KEY` if that server wants a key) in the MCP server's
  environment in your app's config. No tool takes an address, so an assistant cannot send your text anywhere else.
- No telemetry, no accounts, no calls of its own to the internet. The localllm it talks to does what it always does
  (downloads you agreed to, and update notices unless you turned them off: `localllm updates off`).
- stdout is the protocol channel: everything else the server prints goes to stderr.

## Troubleshooting

- *"localllm mcp needs Python 3.10+ and the MCP SDK"*: `pip install "make-localllm-easier[mcp]"` into the Python the app
  launches (`python --version`).
- *"no local model is running"*: run `localllm` once in a terminal, or add `--model KEY` as above.
- *"the model returned no text ... larger max_tokens"*: a reasoning model spent its tokens thinking; raise `max_tokens`.
