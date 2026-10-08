# Use localllm from any app: OpenAI, Anthropic, Ollama and Gemini APIs

`localllm` (and `localllm serve`) puts one endpoint at `http://127.0.0.1:8080` in front of the local model. Point the
app or SDK you already use at it - no code change beyond the base URL. Any API key value works locally.

| API your app speaks | Base URL | Endpoints |
|---|---|---|
| OpenAI | `http://127.0.0.1:8080/v1` | `/chat/completions`, `/completions`, `/models`, `/embeddings` |
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
