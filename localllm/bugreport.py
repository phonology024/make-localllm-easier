"""`localllm bugreport`: what we need to help, in one zip, with secrets and personal paths taken out (#119).

  localllm bugreport           show what goes in, ask, then write ~/.localllm/bugreport-<date>.zip
  localllm bugreport --yes     write it without asking (still prints what is inside)

Inside: `doctor` output, versions (localllm, Python, OS, llama.cpp build, GPU driver), the last lines of the
llama-server and router logs, and localllm's own settings files (tune, route, connect). Before anything is written:
- API keys and tokens go: the LAN key (LOCALLLM_API_KEY), the cloud keys named in route.json (key_env), anything
  shaped like a key (sk-..., Bearer ..., hf_...), and the value of any field named key / token / secret / password;
- the home folder becomes ~ and the user name <user> (the same rule as `localllm report`).
Nothing is uploaded: attach the zip to an issue yourself (the "Bug report" form asks for it).
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import platform
import re
import sys
import time
import zipfile
from pathlib import Path

from . import __version__, report, runtime

LOG_LINES = 200
LOGS = ("llama-server.log", "router-embed.log")
SETTINGS = ("tune.json", "route.json", "connect.json")
GONE = "<redacted>"
KEY_SHAPES = [re.compile(p) for p in (
    r"\bsk-[A-Za-z0-9_-]{12,}",                         # OpenAI / Anthropic / OpenRouter style
    r"\bhf_[A-Za-z0-9]{16,}",                           # Hugging Face
    r"\bgh[pousr]_[A-Za-z0-9]{20,}",                    # GitHub
    r"\bAIza[0-9A-Za-z_-]{20,}",                        # Google
    r"(?i)(?<=bearer )[A-Za-z0-9._~+/=-]{8,}",          # Authorization: Bearer <token>
    r"(?i)(?<=--api-key )\S+",                          # a key on a command line
)]
JSON_FIELD = re.compile(r'("([^"\\]*)"\s*:\s*)"(?:[^"\\]|\\.)*"')
SECRET_NAME = re.compile(r"(?i)key|token|secret|password|authorization")


def _field(m: re.Match) -> str:
    """A JSON string field whose name says secret loses its value; `key_env` (the variable's name) is kept."""
    name = m.group(2)
    if SECRET_NAME.search(name) and not name.lower().endswith("_env"):
        return m.group(1) + f'"{GONE}"'
    return m.group(0)


def known_secrets(env=os.environ) -> list[str]:
    """Values we know are secret on this PC: the LAN key and every key_env that route.json names."""
    names = {"LOCALLLM_API_KEY"}
    try:
        cfg = json.loads((runtime.HOME / "route.json").read_text(encoding="utf-8"))
        names |= {p["key_env"] for p in (cfg.get("providers") or {}).values()
                  if isinstance(p, dict) and isinstance(p.get("key_env"), str)}
    except (OSError, ValueError, AttributeError):
        pass
    return sorted((env[n] for n in names if len(env.get(n) or "") >= 6), key=len, reverse=True)


def redact(text: str, secrets: list[str] | None = None, bits: list[str] | None = None) -> str:
    for s in secrets if secrets is not None else known_secrets():
        text = text.replace(s, GONE)
    for shape in KEY_SHAPES:
        text = shape.sub(GONE, text)
    text = JSON_FIELD.sub(_field, text)
    for b in report._user_bits() if bits is None else bits:          # home folder -> ~, user name -> <user>
        if "/" in b or "\\" in b:
            for form in {b, b.replace("\\", "/"), b.replace("\\", "\\\\")}:
                text = text.replace(form, "~")
        else:
            text = re.sub(rf"(?<![\w.-]){re.escape(b)}(?![\w.-])", "<user>", text)
    return text


def tail(path: Path, n: int = LOG_LINES) -> str | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 400 * n))                # enough bytes for n log lines without reading GBs
        lines = f.read().decode("utf-8", "replace").splitlines()
    return "\n".join(lines[-n:]) + "\n"


def collect(doctor) -> dict[str, str]:
    """{file name in the zip: text}, already redacted. doctor() prints the doctor output."""
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            doctor()
    except BaseException as e:                            # noqa: BLE001 - a broken doctor is part of the report
        out.write(f"\n[doctor failed: {type(e).__name__}: {e}]\n")
    server = report.installed_server()
    versions = {"localllm": __version__, "python": sys.version.split()[0], "os": f"{platform.system()} {platform.release()}",
                "arch": platform.machine(), "llama_cpp": server.parent.name if server else None,
                "driver": report.driver(), "date": time.strftime("%Y-%m-%d %H:%M")}
    files = {"doctor.txt": out.getvalue(), "versions.json": json.dumps(versions, indent=1) + "\n"}
    for name in LOGS:
        text = tail(runtime.HOME / name)
        if text is not None:
            files[name] = text
    for name in SETTINGS:
        p = runtime.HOME / name
        if p.exists():
            files[name] = p.read_text(encoding="utf-8", errors="replace")
    secrets, bits = known_secrets(), report._user_bits()
    return {k: redact(v, secrets, bits) for k, v in files.items()}


def summary(files: dict[str, str]) -> str:
    return "\n".join(f"  {name:20} {len(text.splitlines()):5} lines" for name, text in files.items())


def write(files: dict[str, str], folder: Path | None = None) -> Path:
    out = (folder or runtime.HOME) / f"bugreport-{time.strftime('%Y-%m-%d')}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in files.items():
            z.writestr(name, text)
    return out
