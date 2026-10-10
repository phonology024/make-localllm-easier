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
| | One more thing to maintain; if it does not beat existing agents (OpenCode, Aider, Hermes Agent) on our measurements, we will not ship it |

**Reference design: Anthropic's long-running-app harness.** The post [Harness design for long-running application
development](https://www.anthropic.com/engineering/harness-design-long-running-apps) (Anthropic Labs, March 2026) describes
three roles: a *planner* that expands a short prompt into a spec, a *generator* that builds one feature (sprint) at a time,
and a separate *evaluator* that tests the running result and sends failures back. Techniques worth copying: a "sprint
contract" (agree what "done" means and how it is verified before coding), hand-offs through files, context *resets* with
a structured hand-off instead of in-place summaries, and a skeptical evaluator calibrated with examples. Their numbers:
a solo run took 20 min / $9 and its main feature did not work; the full harness took 6 h / $200 and produced a playable
game, i.e. over 20x the cost for visibly better quality. Their own lesson: "every component in a harness encodes an
assumption about what the model can't do on its own", so test each component and remove the ones that stopped paying off.
The post tests Claude models only, not small ones. For local models it suggests two things we will measure rather than
assume: weaker models probably need *more* scaffolding, and an evaluator is worth it when the task is beyond what the
model does reliably alone. Context resets also fit our footprint goal: a fresh context means a small KV cache.

**Existing harnesses first.** [Hermes Agent](https://github.com/nousresearch/hermes-agent) (Nous Research) is the one
people point at local models: an agent loop with persistent memory, skills it writes for itself as Markdown files
(agentskills.io format), cross-session search, and several terminal backends (local, Docker, SSH, ...). Third-party
coverage says it works with llama.cpp through a custom OpenAI-style endpoint, which is what we serve. What that means
for us (details from the project's own site and secondary write-ups; check its docs before relying on any of it):
- We do not need to rebuild it. The plan is `localllm connect hermes` (same as OpenCode/Aider), then measure it on our
  local models, and build our own loop only for what it lacks (RAM-aware context limits, footprint, per-language routing).
- Windows: its own page lists a PowerShell installer for native Windows (an earlier secondary source said WSL2 only;
  the project's page is the better source), and warns that antivirus may quarantine the bundled `uv.exe` (documented as a
  false positive). Tool execution is risky without approvals or container isolation.
- It is a general agent with a large surface (messaging gateway, 40-60+ tools, cron, subagents). The page reports no
  benchmarks and gives no guidance for small local models or minimum context, which is where we can compete.
- Self-written skills can drift in the wrong direction over time ("skill misevolution", [arXiv 2608.12851](https://arxiv.org/pdf/2608.12851)),
  so keep skills reviewable and under version control.
- Worth borrowing in our own loop: skills and memory as plain Markdown files the user can read and edit.

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
