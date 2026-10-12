"""`localllm connect <tool>`: point a coding agent at the local endpoint (#75).

  localllm connect hermes|opencode|claude-code|codex|aider|cline   write that tool's config (the old file is backed up)
  localllm connect TOOL --undo                                     put the old file back (or remove the one we made)
  localllm connect TOOL --dry-run                                  print what would be written, change nothing

Only the base URL and the model name change; nothing else in the file is touched. The original file is copied to
<file>.localllm-backup the first time, and ~/.localllm/connect.json remembers what to undo. Standard library only, so
YAML and TOML are edited as text: a top-level block we own is replaced, everything else stays byte for byte.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from . import runtime

AGENT_CTX = 65536   # Hermes asks for ~64k; long tool sessions want it too. gemma-4 26B holds ~78k on a 16 GB card
MARK_START, MARK_END = "# >>> localllm connect", "# <<< localllm connect"


def _home() -> Path:
    return Path(os.environ.get("LOCALLLM_CONNECT_HOME") or Path.home())


def model_id(url: str) -> str:
    """The id the running server reports in /v1/models, so tools that check it (Hermes) match; 'localllm' if none."""
    import urllib.request
    try:
        d = json.load(urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=3))
        return d["data"][0]["id"]
    except (OSError, ValueError, KeyError, IndexError):
        return "localllm"


# ---- text edits that keep the rest of a file as it is ----------------------------------------------------------------

def _yaml_set_block(text: str, key: str, block: str) -> str:
    """Replace the top-level `key:` block (the key line and its indented lines) with `block`, or append it."""
    lines = text.splitlines(keepends=True)
    start = next((i for i, l in enumerate(lines) if re.match(rf"{re.escape(key)}\s*:", l)), None)
    if start is None:
        sep = "" if not text or text.endswith("\n") else "\n"
        return text + sep + block
    end = start + 1
    while end < len(lines) and (lines[end].startswith((" ", "\t")) or not lines[end].strip()):
        end += 1
    tail = lines[end:]
    return "".join(lines[:start]) + block + ("\n" if tail and not block.endswith("\n\n") else "") + "".join(tail)


def _yaml_set_keys(text: str, values: dict) -> str:
    """Flat top-level `key: value` lines (aider's style): replace the ones that exist, append the rest."""
    for k, v in values.items():
        line = f"{k}: {v}\n"
        pat = re.compile(rf"^{re.escape(k)}\s*:.*\n?", re.M)
        if pat.search(text):
            text = pat.sub(lambda _m: line, text, count=1)
        else:
            text = text + ("" if not text or text.endswith("\n") else "\n") + line
    return text


def _marked(text: str, block: str) -> str:
    """Replace our marked block, or append it (TOML tables can go at the end of a file)."""
    new = f"{MARK_START}\n{block.rstrip()}\n{MARK_END}\n"
    if MARK_START in text and MARK_END in text:
        head, rest = text.split(MARK_START, 1)
        return head + new + rest.split(MARK_END, 1)[1].lstrip("\n")
    return text + ("\n" if text and not text.endswith("\n\n") else "") + new


# ---- one function per tool: (path, old text or None) -> new text ----------------------------------------------------

def _hermes(old: str | None, base: str, model: str) -> str:
    block = (f'model:\n  default: "{model}"\n  provider: "custom"\n  base_url: "{base}/v1"\n'
             f'  api_key: "local"\n  context_length: {AGENT_CTX}\n')
    return _yaml_set_block(old or "", "model", block)


def _opencode(old: str | None, base: str, model: str) -> str:
    cfg = json.loads(old) if old and old.strip() else {"$schema": "https://opencode.ai/config.json"}
    cfg.setdefault("provider", {})["localllm"] = {
        "npm": "@ai-sdk/openai-compatible", "name": "localllm (this PC)",
        "options": {"baseURL": f"{base}/v1", "apiKey": "local"},
        "models": {model: {"name": model, "limit": {"context": AGENT_CTX, "output": 8192}}}}
    cfg["model"] = f"localllm/{model}"
    return json.dumps(cfg, indent=2, ensure_ascii=False) + "\n"


def _claude_code(old: str | None, base: str, model: str) -> str:
    cfg = json.loads(old) if old and old.strip() else {}
    env = cfg.setdefault("env", {})
    env.update({"ANTHROPIC_BASE_URL": base, "ANTHROPIC_AUTH_TOKEN": "local", "ANTHROPIC_MODEL": model,
                "ANTHROPIC_DEFAULT_OPUS_MODEL": model, "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": model})
    return json.dumps(cfg, indent=2, ensure_ascii=False) + "\n"


def _codex(old: str | None, base: str, model: str) -> str:
    # a profile, so the user's default provider stays as it is: `codex --profile localllm`
    block = (f'[model_providers.localllm]\nname = "localllm (this PC)"\nbase_url = "{base}/v1"\n'
             f'wire_api = "responses"\n\n[profiles.localllm]\nmodel_provider = "localllm"\nmodel = "{model}"\n')
    return _marked(old or "", block)


def _aider(old: str | None, base: str, model: str) -> str:
    return _yaml_set_keys(old or "", {"openai-api-base": f"{base}/v1", "openai-api-key": "local",
                                      "model": f"openai/{model}"})


TOOLS = {
    "hermes": (lambda h: h / ".hermes" / "config.yaml", _hermes, "hermes"),
    "opencode": (lambda h: h / ".config" / "opencode" / "opencode.json", _opencode, "opencode"),
    "claude-code": (lambda h: h / ".claude" / "settings.json", _claude_code, "claude"),
    "codex": (lambda h: h / ".codex" / "config.toml", _codex, "codex --profile localllm"),
    "aider": (lambda h: h / ".aider.conf.yml", _aider, "aider"),
}
CLINE = """Cline keeps its settings inside VS Code, so there is no file to write. In Cline's settings:
  API Provider: OpenAI Compatible
  Base URL:     {base}/v1
  API Key:      local
  Model ID:     {model}
  Context window: {ctx}"""


def _state_file() -> Path:
    return runtime.HOME / "connect.json"


def _state() -> dict:
    try:
        return json.loads(_state_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_state(s: dict) -> None:
    _state_file().parent.mkdir(parents=True, exist_ok=True)
    _state_file().write_text(json.dumps(s, indent=1), encoding="utf-8")


def connect(tool: str, base: str, model: str, path: Path | None = None, dry_run: bool = False) -> str:
    """Write the config; returns what to tell the user."""
    if tool == "cline":
        return CLINE.format(base=base, model=model, ctx=AGENT_CTX)
    where, edit, run = TOOLS[tool]
    path = path or where(_home())
    old = path.read_text(encoding="utf-8") if path.exists() else None
    new = edit(old, base, model)
    if dry_run:
        return f"would write {path}:\n{new}"
    backup = path.with_name(path.name + ".localllm-backup")
    state = _state()
    if str(path) not in state:                         # first connect: remember the original, keep it forever
        if old is not None:
            shutil.copy2(path, backup)
        state[str(path)] = {"tool": tool, "backup": str(backup) if old is not None else None}
        _save_state(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new, encoding="utf-8", newline="")
    saved = f" (old file: {backup.name})" if state[str(path)]["backup"] else ""
    return (f"wrote {path}{saved}\n"
            f"start the model with room for agent sessions:  localllm serve --ctx {AGENT_CTX}\n"
            f"then run:  {run}\n"
            f"undo:  localllm connect {tool} --undo")


def undo(tool: str, path: Path | None = None) -> str:
    if tool == "cline":
        return "nothing to undo: Cline's settings are in VS Code"
    path = path or TOOLS[tool][0](_home())
    state = _state()
    entry = state.pop(str(path), None)
    if not entry:
        return f"nothing to undo for {path}"
    if entry["backup"]:
        shutil.move(entry["backup"], path)
        msg = f"restored {path}"
    else:
        path.unlink(missing_ok=True)
        msg = f"removed {path} (it didn't exist before)"
    _save_state(state)
    return msg
