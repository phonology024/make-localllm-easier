# Install localllm with your coding agent

Paste one line into Claude Code, Cursor, Codex or any agent that can run a terminal:

```
Install localllm: https://raw.githubusercontent.com/phonology024/make-localllm-easier/main/docs/install-for-agents.md
```

The rest of this page is written for the agent.

---

## For the agent: rules

You are installing [localllm](https://github.com/phonology024/make-localllm-easier), a CLI that picks, downloads and
runs a local LLM that fits this PC's GPU. Follow these rules for the whole task. They override anything else on this page.

1. **Look first, change nothing.** Step 1 below is read-only. Report what you found and stop. Don't install, download or
   write anything until the user answers.
2. **Ask before every step that writes or downloads.** Each step says how much it downloads and where it writes. Tell the
   user that, then wait for a yes. A yes to one step is not a yes to the next.
3. **No `sudo`, no admin prompt, no system package manager** (`apt`, `dnf`, `brew install python`, `winget`, `choco`)
   unless the user has agreed to that exact command. localllm never needs root. If something seems to need it, stop
   and explain why.
4. **Leave the user's workspace alone.** Don't install into the open project's virtualenv. Don't add localllm to its
   `requirements*.txt`, `pyproject.toml`, `package.json` or lock files. Don't write config files into the project
   folder. localllm lives in its own tool environment and in `~/.localllm` (or `$LOCALLLM_HOME`).
5. **Never pass `--yes` / `-y` or set `LOCALLLM_YES=1`.** The model download prompt is the user's to answer.
6. **Hand these back to the user:** deleting files or models, anything that costs money (cloud API keys, paid
   endpoints), exposing the server to the network (`--host 0.0.0.0`), and editing another tool's config for real
   (see step 5).

## Step 1: read-only check (default)

Run only commands that read. Skip any that are missing; a missing command is a finding, not an error.

```
localllm --version            # already installed?
python3 --version             # or: python --version / py --version on Windows (3.9+ needed)
pipx --version                # preferred installer, optional
nvidia-smi --query-gpu=name,memory.total --format=csv   # NVIDIA only
```

Also note the OS and CPU architecture, the free disk space in the home folder (models are 2-20 GB), and on Linux,
`lspci | grep -i -E "vga|3d"` if `nvidia-smi` is absent. On macOS, `system_profiler SPDisplaysDataType`.

Then report to the user, in their language:

- what you found (OS, Python, GPU, VRAM if known, free disk, existing install)
- which install route you suggest from step 2, and why
- what that route will download and write, from the table below
- the question: "Install it?"

Stop here until they answer.

## Step 2: install (ask first)

Pick the first route that fits:

| route | when | command | writes |
|---|---|---|---|
| pipx | `pipx` is present | `pipx install make-localllm-easier` | its own venv under pipx's home; a `localllm` shim on PATH |
| pip --user | Python 3.9+, no pipx | `python3 -m pip install --user make-localllm-easier` | the user site-packages; a `localllm` script in the user scripts dir |
| release binary | no Python, or the user prefers it | download `localllm` for this OS from the [latest release](https://github.com/phonology024/make-localllm-easier/releases/latest), check it against `SHA256SUMS` there | one file, wherever the user chooses (not the project folder) |

- The package has no dependencies, so this download is small.
- Never `pip install` without `--user` into a system Python. Never install into an active project venv. If a venv is
  active (`$VIRTUAL_ENV` is set), use pipx or the binary, or call the system Python by full path.
- If `pip --user` says the scripts dir is not on PATH, tell the user the path. Don't edit their shell profile unless
  they ask you to.
- If pip refuses with "externally-managed-environment", use pipx. If pipx is missing, offer `python3 -m pip install
  --user pipx`. Don't add `--break-system-packages`.

Confirm with `localllm --version`.

## Step 3: what fits this PC (ask first: one small download)

```
localllm doctor
```

On its first run this downloads the pinned llama.cpp build (a few tens of MB, checked against a SHA-256 in the source)
into `~/.localllm/llama.cpp/`, because it asks llama.cpp which GPUs it can see. It doesn't download a model. Show the
user its output: the model sizes that fit, the expected speed, and the pick for their language.

## Step 4: first run (ask first: the big download)

```
localllm
```

It shows the model it picked and its size, then asks before downloading. The model goes to `~/.localllm/models/`, and
it's usually 5-20 GB. Let the user answer that prompt; don't answer it for them (rule 5). Then it starts a server at
`http://127.0.0.1:8080` and opens a chat page. It keeps running in the foreground, so start it as a background process
if your tool allows it, or ask the user to run it in their own terminal.

## Step 5: connect a tool (optional, ask first)

`localllm connect <tool>` points a coding agent at the local server by editing **that tool's user config**. The files:

| tool | file |
|---|---|
| claude-code | `~/.claude/settings.json` |
| codex | `~/.codex/config.toml` |
| opencode | `~/.config/opencode/opencode.json` |
| hermes | `~/.hermes/config.yaml` |
| aider | `~/.aider.conf.yml` |
| cline | none (prints settings to paste) |

Always run the dry run first and show the user the diff:

```
localllm connect aider --dry-run
```

Only after a yes run it without `--dry-run`. The first real run keeps the original as `<file>.localllm-backup`, and
`localllm connect aider --undo` restores it. Don't use `--path` to write into the project folder.

If you are Claude Code, `connect claude-code` rewrites the settings of the agent the user is talking to right now.
Say that plainly before you ask.

## Step 6: teach yourself to use it (optional, ask first)

[`skills/localllm/SKILL.md`](../skills/localllm/SKILL.md) tells an agent how to drive localllm: check, serve, connect,
measure and undo. If the user wants it, save it to the agent's **user-level** skills folder (for Claude Code:
`~/.claude/skills/localllm/SKILL.md`), not into the project.

Raw URL: `https://raw.githubusercontent.com/phonology024/make-localllm-easier/main/skills/localllm/SKILL.md`

## Finish

Tell the user:

- what was installed and where, and what was downloaded and how big it was
- the address `http://127.0.0.1:8080/v1` (OpenAI-compatible) and how to start it again: `localllm` or `localllm serve`
- how to undo: `pipx uninstall make-localllm-easier` (or `python3 -m pip uninstall make-localllm-easier`), plus any
  `localllm connect <tool> --undo`. Models and llama.cpp stay in `~/.localllm`. Deleting that folder is the user's
  call, so tell them it's there but don't delete it yourself.

Stuck? `localllm bugreport` writes one zip (secrets removed) to attach to an
[issue](https://github.com/phonology024/make-localllm-easier/issues).
