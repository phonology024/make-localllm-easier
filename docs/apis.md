# Use localllm from any app: OpenAI, Anthropic, Ollama and Gemini APIs

`localllm` (and `localllm serve`) puts one endpoint at `http://127.0.0.1:8080` in front of the local model. Point the
app or SDK you already use at it - no code change beyond the base URL. Any API key value works locally.

| API your app speaks | Base URL | Endpoints |
|---|---|---|
| OpenAI | `http://127.0.0.1:8080/v1` | `/chat/completions`, `/completions`, `/models`, `/embeddings`, `/responses` |
| Anthropic Messages | `http://127.0.0.1:8080` | `/v1/messages`, `/v1/messages/count_tokens` (tools, vision, thinking) |
| Ollama | `http://127.0.0.1:8080` | `/api/chat`, `/api/generate`, `/api/tags`, `/api/version` |
| Gemini | `http://127.0.0.1:8080/v1beta` | `models/{m}:generateContent`, `models/{m}:streamGenerateContent` |

## Examples

OpenAI Python SDK:
```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
print(client.chat.completions.create(model="local", messages=[{"role": "user", "content": "Hello"}]).choices[0].message.content)
```

Anthropic Python SDK (also works for tools built on it):
```python
import anthropic
client = anthropic.Anthropic(base_url="http://127.0.0.1:8080", api_key="local")
print(client.messages.create(model="local", max_tokens=200, messages=[{"role": "user", "content": "Hello"}]).content[0].text)
```

Apps that talk to Ollama: set the Ollama host to `http://127.0.0.1:8080`.

Gemini REST:
```
curl http://127.0.0.1:8080/v1beta/models/local:generateContent -H "Content-Type: application/json" \
  -d "{\"contents\":[{\"parts\":[{\"text\":\"Hello\"}]}]}"
```

## Coding agents: `localllm connect`

```
localllm serve --profile max        # agents send long prompts: a context that fills the GPU (Hermes asks for 64k)
localllm connect hermes             # or opencode, claude-code, codex, aider, cline
localllm connect hermes --undo      # put the old config back
```

`connect` changes only the base URL and the model name in the tool's own config. The first time, the original file is
copied next to it as `<file>.localllm-backup`; `--undo` puts it back (or deletes the file if `connect` created it).
`--dry-run` prints the new file without writing it, `--path` writes another file (a project config, say).

| tool | file | what is set | then run |
|---|---|---|---|
| Hermes Agent | `~/.hermes/config.yaml` | the `model:` block: `provider: custom`, `base_url`, `default`, `context_length` | `hermes` |
| OpenCode | `~/.config/opencode/opencode.json` | a `localllm` provider (`@ai-sdk/openai-compatible`) and `model` | `opencode` |
| Claude Code | `~/.claude/settings.json` | `env`: `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, the model names | `claude` |
| Codex CLI | `~/.codex/config.toml` | a `localllm` provider (`wire_api = "responses"`) and a `localllm` profile; your default stays | `codex --profile localllm` |
| Aider | `~/.aider.conf.yml` | `openai-api-base`, `openai-api-key`, `model: openai/<id>` | `aider` |
| Cline | (inside VS Code) | nothing is written; `connect cline` prints the four fields to fill in | |
| Obsidian (Copilot plugin) | `<vault>/.obsidian/plugins/copilot/data.json` (`--path VAULT`) | a `3rd party (openai-format)` chat model, made the default | reload Obsidian |

The model name is the one the running server reports at `/v1/models` (pass `--model-id` to choose another).
Claude Code with `connect` sends everything to the local model, including the small background requests; run
`localllm connect claude-code --undo` to go back to your Anthropic account.

### Obsidian

```
localllm serve
localllm connect obsidian --path "D:/Notes"     # the vault folder; Copilot for Obsidian must be installed there
localllm connect obsidian --undo --path "D:/Notes"
```

`connect obsidian` adds the local model to [Copilot for Obsidian](https://github.com/logancyang/obsidian-copilot) and
makes it the default chat model. Your other models stay in the list. It then tells you what would still reach a cloud
service: a cloud embedding model for vault search (indexing sends note chunks to it) and any cloud chat model still
enabled in the menu. Turn those off in Copilot's settings if nothing should leave the PC.

Vault search needs an embedding model. A small local one: the multilingual e5 model `localllm` already downloads for
its task router (`~/.localllm/models/multilingual-e5-small-Q8_0.gguf`, 126 MB, ~100 languages):

```
llama-server -m ~/.localllm/models/multilingual-e5-small-Q8_0.gguf --embedding --pooling mean --port 8081
```

In Copilot: Settings > QA > Embedding model > add a custom model, provider "3rd party (openai-format)", base URL
`http://127.0.0.1:8081/v1`, model `multilingual-e5-small`, then rebuild the index. Smart Connections works the same
way through its "custom local (OpenAI format)" option. An Obsidian plugin of our own (#89, part B) comes only if this
isn't enough.

Not measured yet: how well each agent works with our models over long sessions, and how much RAM a 30-turn tool session
uses at 20k+ tokens. Both are open issues (#77, #96); please report what you see.

## Tool / function calling

All four APIs pass tools through: OpenAI `tools` / `tool_calls`, Anthropic `tools` / `tool_use`, Ollama `tools` /
`message.tool_calls` (arguments as an object) and Gemini `functionDeclarations` / `functionCall` / `functionResponse`
(with `toolConfig` mapped to `tool_choice`). Streaming works too: Ollama gets the calls in the final chunk, Gemini when
the model finishes the turn. The model needs a chat template that knows tools; current llama-server builds use the GGUF's
own Jinja template by default.

Checked end to end in CI (`.github/workflows/tools.yml`, `tools/tool_check.py`): call a `get_weather` tool, send the
result back, get the final answer - every API, with and without streaming:

| Model (Q8_0) | OpenAI | Anthropic | Ollama | Gemini |
|---|---|---|---|---|
| Qwen3-0.6B | ok / ok | ok / ok | ok / ok | ok / ok |
| Qwen3.5-2B | ok / ok | ok / ok | ok / ok | ok / ok |
| Gemma 4 E2B | ok / ok | ok / ok | ok / ok | ok / ok |

(json / stream). Try your own model: `python tools/tool_check.py --llama http://127.0.0.1:8081 --name MODEL`.

## Images (vision)

Start with the model's vision projector: `localllm --vision` (or `localllm serve --vision`). It is downloaded once
(gemma-4 26B: 1.1 GB, Qwen3.8-27B: 0.9 GB) and loaded only with this flag, so text-only use costs no extra memory.
Then send images the way each API already does:

| API | Image in the request |
|---|---|
| OpenAI | `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` content part |
| Anthropic | `{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}` block |
| Ollama | `"images": ["<base64>"]` on a message (or on `/api/generate`) |
| Gemini | `{"inlineData": {"mimeType": "image/png", "data": "<base64>"}}` part (`fileData` URIs are not supported) |

Without the projector, an image gets a `400` in that API's error format telling you to restart with `--vision`; text
keeps working.

Checked in CI (`.github/workflows/vision.yml`, `tools/vision_check.py`, CPU runner, llama.cpp b11487): every API, with
and without streaming, answers "green" for a green picture; without the projector every API gets the `--vision` error.
What the projector costs, and a first look at `localllm eval --suites vision` (MaXM, first 30 questions per language,
so +-17 points):

| Model (Q8_0) + projector | Extra RAM (CPU, peak RSS) | MaXM en | MaXM th | MaXM zh |
|---|---|---|---|---|
| Qwen3.5-2B + mmproj F16 | +0.65-0.80 GiB | 43.3% | 63.3% | 46.7% |
| Gemma 4 E2B + mmproj Q8_0 | +0.62 GiB | 23.3%* | 56.7%* | 33.3%* |

\* Gemma 4 E2B often starts its reply with a written-out "Thinking Process" on image questions in this build, even with
thinking off, and such replies count as wrong, so its scores understate it. GPU VRAM with and without the projector
still needs a measurement on a real card.

## Mixing in your own cloud keys (optional, off by default)

Everything stays on your PC unless you create `~/.localllm/route.json` with `"enabled": true` and put a key in an
environment variable (or the OS keychain via the `keyring` package). Then a request goes to your cloud provider only when
a rule says so:

- the app asked for a cloud model by name (`gpt-...`, `claude-...`, `gemini-...`)
- the prompt is longer than `max_local_prompt_tokens`
- the local model's **measured** score in the message's language is below `language_floor`

Each response carries an `X-Localllm-Route` header saying where it went and why. `localllm route` shows the config;
`localllm route --test "your prompt"` sends the same prompt to the local model and to each configured provider and
prints time, tokens and cost side by side. Prices per million tokens are taken from `price_in` / `price_out` in the
config when you set them.

```json
{
  "enabled": true,
  "provider": "openrouter",
  "providers": {
    "openrouter": {"kind": "openai", "url": "https://openrouter.ai/api", "key_env": "OPENROUTER_API_KEY",
                   "model": "anthropic/claude-sonnet-4", "price_in": 3.0, "price_out": 15.0},
    "anthropic":  {"kind": "anthropic", "url": "https://api.anthropic.com", "key_env": "ANTHROPIC_API_KEY",
                   "model": "claude-sonnet-4-5"}
  },
  "rules": {"max_local_prompt_tokens": 6000, "language_floor": 60, "cloud_model_names": true},
  "local_model": "qwen3.8-27b-q3"
}
```

## Use it from another device on your network (LAN mode)

Run the model on the desktop with the GPU and use it from a laptop or phone on the same network:

```
localllm serve --host 0.0.0.0                  # prints the LAN address and a new API key, once
localllm serve --host 0.0.0.0 --api-key KEY    # or bring your own key (or set LOCALLLM_API_KEY)
localllm serve --host ::                       # IPv6 too (and IPv4 where the OS allows dual stack)
```

Other devices send the key the way their SDK already does:

| API | How the key is sent |
|---|---|
| OpenAI | `api_key="KEY"` (sent as `Authorization: Bearer KEY`) |
| Anthropic | `api_key="KEY"` (sent as `x-api-key: KEY`) |
| Ollama | `Authorization: Bearer KEY` header (e.g. `ollama.Client(host=..., headers={"Authorization": "Bearer KEY"})`) |
| Gemini | `x-goog-api-key: KEY` header, or `?key=KEY` on the URL |

Requests without the right key get `401`. Requests from the desktop itself need no key, so local apps, `localllm chat`
and the browser page keep working. On another device the chat page loads without the key, but its requests need it:
enter the key in the page's settings (API key).

Security notes:
- `localllm` refuses to listen on a network address without a key. Keep the key secret: anyone who has it can use your
  GPU, and through your cloud key too if you turned cloud routing on.
- The key is checked in constant time and stripped before anything is forwarded, so it never reaches llama-server or a
  cloud provider. llama-server itself still listens only on `127.0.0.1`.
- Behind a reverse proxy or tunnel on the same PC (nginx, Caddy, cloudflared, ngrok), every request reaches localllm
  from `127.0.0.1`, so "no key from this PC" would let the whole proxy through without one. Start it with
  `--require-key-local` there: then every client, this PC included, must send the key.
- Traffic is plain HTTP. Use it on a network you trust (home Wi-Fi), or put it behind a VPN such as WireGuard or
  Tailscale. Don't forward the port to the internet.
- `?key=` ends up in browser history and proxy logs; prefer the header when you can.
