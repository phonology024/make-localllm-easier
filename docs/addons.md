# Optional add-ons (planned)

Nothing here is installed by default. `pip install make-localllm-easier` and `localllm` stay exactly as small as they
are today. Each add-on is opt-in and can be removed without touching the core. **None of them exists yet**; this page
says what they will be and what you give up by installing them, so you can decide before we build them. Status lives in
the linked issues.

## Harness (agent loop for small local models) - #88

**What it is.** A small program that wraps the local model in a loop: plan, call a tool (read files, run a command, run
tests in the existing sandbox), check the result, retry from the error. Not to be confused with `localllm eval`, which is
the *measurement* harness and already exists.

**Why.** A 7-35B model alone is weak at multi-step work; a loop that verifies its own output (tests pass, answer matches
a checker) is where small models gain the most. See [research-big-models.md](research-big-models.md), sections 5-6.

| Pros | Cons |
|---|---|
| Better results on checkable tasks (code with tests, math with a known answer) than a single answer | Slower: several model calls per task, so several times more tokens and wall time (unmeasured) |
| Everything stays on your PC; tool use runs inside the sandbox | Does not make a small model know more. On open knowledge questions the gain is small or none |
| Same endpoint as everything else, so `localllm connect` tools can use it | More RAM/VRAM pressure from long contexts (the KV cache grows every step) |
| We publish measured before/after numbers per language | A tool-using agent can run commands; it must stay sandboxed and ask before touching files |
| | One more thing to maintain; if it does not beat existing agents (OpenCode, Aider) on our measurements, we will not ship it |

## Obsidian integration - #89

Two layers, cheapest first.

**A. `localllm connect obsidian` (config only).** Writes the settings for an existing, mature plugin (for example
Copilot for Obsidian or Smart Connections) so it talks to `http://127.0.0.1:8080/v1`. No new plugin to maintain.

**B. Our own Obsidian plugin (only if A is not enough).** A TypeScript plugin that adds: chat over your vault using the
local model, model status in the status bar (RAM, GPU spill, tokens/s), and "this note is too long for the loaded
context" warnings.

| | A. Connect existing plugin | B. Our own plugin |
|---|---|---|
| Pros | Small to build; mature UI and vault search; nothing new to keep alive | Knows about our runtime: footprint, spill, router choice, language scores; one install for everything |
| Cons | Plugin quality and privacy are the plugin's; some also offer cloud providers in the same settings panel, so check it is set to local; limited to what that plugin exposes | A second codebase (TypeScript) and release process; Obsidian community review takes time; duplicates plugins that already work |

**Privacy notes that apply to both.** Local model does not automatically mean nothing leaves your PC: check that no cloud
provider is configured, and remember vault embeddings are stored inside the vault (they sync wherever your vault syncs).
Mobile is weak: your phone has to reach the PC over LAN mode with its API key. Local models are weaker than the biggest
cloud models; summarising and searching notes works well, complex reasoning over many notes works less well.

**Our recommendation:** build A first (a day of work), measure whether people want B.
