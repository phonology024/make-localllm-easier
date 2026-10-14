"""`localllm import` (#110): reuse GGUF models another app already downloaded - Ollama, LM Studio, Jan - instead of
downloading them again.

A file is matched to a catalog model by SHA-256 (Ollama names its blobs by their SHA-256, so those cost nothing; other
files are hashed only when their size is within 2% of a catalog model's). A match is hard-linked into
~/.localllm/models under the catalog's file name: no copy, no extra disk space, and the other app's files are never
moved, renamed or changed. Across drives a hard link is impossible, so it says to point LOCALLLM_MODELS at the folder
instead. GGUFs that match nothing are listed as unmeasured: we have no scores for them. Dry run unless --apply.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from . import catalog, runtime

SIZE_SLACK_GB = 0.06    # catalog sizes are rounded to 0.1 GB


@dataclass
class Found:
    app: str
    path: Path
    size: int
    sha256: str | None = None      # known without hashing (Ollama) or after hashing
    name: str = ""                 # the other app's name for it (Ollama tag, file name)


def _home() -> Path:
    return Path(os.environ.get("USERPROFILE") or Path.home())


def ollama_dir() -> Path:
    return Path(os.environ.get("OLLAMA_MODELS") or _home() / ".ollama" / "models")


def scan_ollama(root: Path | None = None) -> list[Found]:
    """Models from Ollama manifests: the model layer's digest is the blob's SHA-256."""
    root = root or ollama_dir()
    out = []
    for man in sorted((root / "manifests").glob("*/*/*/*")) if (root / "manifests").exists() else []:
        try:
            layers = json.loads(man.read_text(encoding="utf-8")).get("layers", [])
        except (OSError, ValueError):
            continue
        for layer in layers:
            if layer.get("mediaType") != "application/vnd.ollama.image.model":
                continue
            digest = layer.get("digest", "")
            blob = root / "blobs" / digest.replace(":", "-")
            if blob.exists() and digest.startswith("sha256:"):
                out.append(Found("Ollama", blob, blob.stat().st_size, digest.split(":", 1)[1],
                                 f"{man.parent.name}:{man.name}"))
    return out


def gguf_dirs() -> list[tuple[str, Path]]:
    h = _home()
    appdata = Path(os.environ.get("APPDATA") or h / "AppData" / "Roaming")
    return [("LM Studio", h / ".lmstudio" / "models"), ("LM Studio", h / ".cache" / "lm-studio" / "models"),
            ("Jan", h / "jan" / "models"), ("Jan", appdata / "Jan" / "data" / "models"),
            ("Jan", appdata / "Jan" / "data" / "llamacpp" / "models")]


def scan_folders(dirs: list[tuple[str, Path]] | None = None) -> list[Found]:
    out, seen = [], set()
    for app, d in dirs if dirs is not None else gguf_dirs():
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*.gguf")):
            if f.resolve() in seen or "mmproj" in f.name.lower():
                continue
            seen.add(f.resolve())
            out.append(Found(app, f, f.stat().st_size, None, f.name))
    return out


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def match(found: list[Found], hash_file=sha256_of) -> list[tuple[Found, str | None]]:
    """(file, catalog key or None). Files are hashed only if their size is near a catalog model's."""
    by_sha = {m["sha256"]: k for k, m in catalog.MODELS.items() if m.get("sha256")}
    out = []
    for f in found:
        if f.sha256 is None and any(abs(f.size / 2**30 - m["gb"]) <= 0.02 * m["gb"] + SIZE_SLACK_GB
                                    for m in catalog.MODELS.values()):
            f.sha256 = hash_file(f.path)
        out.append((f, by_sha.get(f.sha256) if f.sha256 else None))
    return out


def plan(matches: list[tuple[Found, str | None]], models_dir: Path | None = None) -> list[dict]:
    """What --apply would do, one row per file: link / have / other-drive / unmeasured."""
    models_dir = models_dir or runtime.HOME / "models"
    rows = []
    for f, key in matches:
        row = {"app": f.app, "name": f.name, "path": str(f.path), "gb": round(f.size / 2**30, 1), "key": key}
        if not key:
            row["action"] = "unmeasured"
        else:
            dest = models_dir / catalog.MODELS[key]["file"]
            row["dest"] = str(dest)
            if dest.exists():
                row["action"] = "have"
            elif _same_volume(f.path, models_dir):
                row["action"] = "link"
            else:
                row["action"] = "other-drive"
        rows.append(row)
    return rows


def _same_volume(a: Path, b: Path) -> bool:
    while not b.exists() and b.parent != b:      # models dir not made yet: its nearest existing parent decides
        b = b.parent
    try:
        return os.stat(a).st_dev == os.stat(b).st_dev
    except OSError:
        return False


def apply(rows: list[dict]) -> list[dict]:
    """Hard-link every 'link' row. The source file is only read: os.link adds a second name for the same data."""
    for r in rows:
        if r["action"] != "link":
            continue
        dest = Path(r["dest"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(r["path"], dest)
            r["action"] = "linked"
        except OSError as e:
            r["action"], r["error"] = "failed", str(e)
    return rows


def describe(rows: list[dict], applied: bool) -> str:
    if not rows:
        return "No models from Ollama, LM Studio or Jan found on this PC."
    lines = []
    verbs = {"link": "would link" if not applied else "link", "linked": "linked", "have": "already have",
             "other-drive": "on another drive", "unmeasured": "unmeasured", "failed": "could not link"}
    for r in rows:
        what = f"{r['app']}: {r['name']} ({r['gb']} GB)"
        if r["key"]:
            what += f" = {r['key']}"
        line = f"  {verbs[r['action']]:15} {what}"
        if r["action"] == "other-drive":
            line += f"\n                  set LOCALLLM_MODELS={Path(r['path']).parent} to use it where it is"
        if r["action"] == "failed":
            line += f"\n                  {r['error']}"
        lines.append(line)
    if not applied and any(r["action"] == "link" for r in rows):
        lines.append("dry run: nothing changed. `localllm import --apply` hard-links the matches (no copy, no extra "
                     "disk space; the other app's files stay as they are).")
    if any(r["action"] == "unmeasured" for r in rows):
        lines.append("unmeasured: not in localllm's catalog, so there are no scores for them; `localllm list` shows "
                     "the measured models.")
    return "\n".join(lines)
