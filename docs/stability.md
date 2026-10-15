# What stays stable

From 1.0 on, everything on this page keeps working inside 1.x. Until 1.0 (now: 0.x) it is a target, not a promise,
but we already treat it that way and `tests/test_stability.py` fails if any of it disappears.

## The stable surface

**Commands and flags.** `localllm` (no command), `doctor`, `list`, `tune`, `chat`, `serve`, `route`, `eval`,
`report`, `connect`, and the flags each one documents in `--help` today: `--model`, `--port`, `--ctx`,
`--no-browser`, `--models`, `--idle-unload`, `--host`, `--api-key`, `--require-key-local`, `--vision`, `--profile lean|max`, `--ui`,
`eval --url/--name/--limit/--langs/--suites`, `chat --url`, `report --no-browser`,
`connect --undo/--dry-run/--url/--model-id/--path`.

**HTTP endpoints** on the local port (default 8080):

- OpenAI: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`, `/v1/responses`
- Anthropic: `/v1/messages`, `/v1/messages/count_tokens`
- Ollama: `/api/chat`, `/api/generate`, `/api/tags`, `/api/version`
- Gemini: `/v1beta/models/{m}:generateContent`, `:streamGenerateContent`
- `/health`, and the chat page at `/`

**Headers:** `X-Localllm-Route` and `X-Localllm-Model` (responses), `X-Localllm-Stay` (requests).

**Files and environment.** `~/.localllm/` (or `LOCALLLM_HOME`) with `models/`, `llama.cpp/`, `tune.json`,
`results.json`, `route.json`, `connect.json`, `report-<date>.json`. Variables: `LOCALLLM_HOME`, `LOCALLLM_MODELS`,
`LOCALLLM_LLAMA_SERVER`, `LOCALLLM_LLAMA_BUILD`, `LOCALLLM_API_KEY`.

**Data formats.** The `localllm report` JSON and `data/leaderboard.json`, both versioned by their `schema` field,
and the published catalog `catalog/v1/catalog.json`, versioned by its folder ([report-schema.md](report-schema.md)). Catalog keys (`qwen3.8-27b-q3`, `gemma4-26b-a4b-qat`, ...) stay valid names
for `--model`; a model we drop from the catalog keeps its key as an alias for one minor release with a warning.

## Not covered

What the automatic pick chooses (it follows new measurements), speeds and scores, the text output of `doctor` and
`list` (read the JSON from `report` if you parse it), anything under `tools/`, and Python imports from `localllm.*`.

## How things get removed

1. The release that decides to remove something keeps it working and prints a warning when it is used, saying what to
   use instead. The release notes list it under "Deprecated".
2. It is removed no earlier than the next minor release (1.3 deprecates, 1.4 removes at the earliest), and the release
   notes list it under "Removed".
3. A breaking change to a data format bumps its `schema`; readers keep accepting the old one for one minor release.

Security fixes are the exception: if keeping something working would leave users exposed, it can change at once, and
the release notes say why.
