"""`localllm serve --ui`: the local chat workspace (#76).

The page (ui.html, no external assets, no telemetry) is served by the gateway at `/`. Chats are plain JSON files in
~/.localllm/sessions/<id>.json, one per chat, readable and deletable by hand. Endpoints (same key rules as the API):

  GET    /localllm/catalog           models to pick from, with size, measured speed and the score in this PC's language
  GET    /localllm/sessions          [{id, title, updated}] newest first
  GET    /localllm/sessions/<id>     one chat
  PUT    /localllm/sessions/<id>     save it (whole document)
  DELETE /localllm/sessions/<id>
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from . import catalog, runtime

PAGE = Path(__file__).with_name("ui.html")
ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_BYTES = 8 * 2**20          # one chat file; a longer one is refused rather than filling the disk


class Sessions:
    def __init__(self, folder: Path | None = None):
        self.dir = folder or runtime.HOME / "sessions"

    def _path(self, sid: str) -> Path:
        if not ID.match(sid):
            raise ValueError("bad session id")
        return self.dir / f"{sid}.json"

    def list(self) -> list[dict]:
        out = []
        for p in self.dir.glob("*.json") if self.dir.exists() else []:
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append({"id": p.stem, "title": str(d.get("title") or "")[:80], "updated": d.get("updated", 0)})
        return sorted(out, key=lambda s: -s["updated"])

    def get(self, sid: str) -> dict | None:
        p = self._path(sid)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def put(self, sid: str, raw: bytes) -> dict:
        if len(raw) > MAX_BYTES:
            raise ValueError("chat too large to save (8 MB)")
        d = json.loads(raw or b"{}")
        if not isinstance(d, dict) or not isinstance(d.get("messages", []), list):
            raise ValueError("a session is an object with a messages list")
        d.update(id=sid, updated=time.time())
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self._path(sid).with_suffix(".tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self._path(sid))
        return {"id": sid, "updated": d["updated"]}

    def delete(self, sid: str) -> bool:
        p = self._path(sid)
        if not p.exists():
            return False
        p.unlink()
        return True


def catalog_info(keys: list[str], lang: str, choosable: bool) -> dict:
    """What the model picker shows: size, measured speed (RX 9070 XT) and the mean score in this PC's language."""
    models = []
    for k in keys:
        m = catalog.MODELS.get(k)
        if not m:
            continue
        mine = [v for t, v in m["scores"].items() if t.startswith(lang + "/") and not t.endswith("/translate")]
        models.append({"key": k, "gb": m["gb"], "tok_s": m["tok_s_9070xt"], "note": m["note"], "lang": lang,
                       "score": round(sum(mine) / len(mine), 1) if mine else None})
    return {"models": models, "choosable": choosable, "single": keys[0] if len(keys) == 1 else None}
