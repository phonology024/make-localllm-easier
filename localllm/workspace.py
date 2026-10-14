"""`localllm serve --ui`: the local chat workspace (#76).

The page (ui.html, no external assets, no telemetry) is served by the gateway at `/`. Chats are plain JSON files in
~/.localllm/sessions/<id>.json, one per chat, readable and deletable by hand. Endpoints (same key rules as the API):

  GET    /localllm/catalog           models to pick from, with size, measured speed and the score in this PC's language
  GET    /localllm/models            the Models tab (#107): every catalog model, whether it fits this PC, its scores in
                                     this PC's language, expected speed, downloaded / running / download progress
  POST   /localllm/models/<key>/download   fetch a catalog model in the background (checked against its SHA-256)
  GET    /localllm/sessions          [{id, title, updated}] newest first
  GET    /localllm/sessions/<id>     one chat
  PUT    /localllm/sessions/<id>     save it (whole document)
  DELETE /localllm/sessions/<id>
"""
from __future__ import annotations

import json
import re
import threading
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


def fit(key: str, vram_gb: float, ram_free_gb: float) -> tuple[str, int]:
    """("fits", 0) whole on the GPU with an 8k context; ("spill", n) a MoE with n layers' experts in system RAM
    (works, slower); ("no", 0) won't run on this PC. Same rule `localllm` uses to pick (catalog.cpu_moe_layers)."""
    n = catalog.cpu_moe_layers(key, vram_gb, ram_free_gb)
    return ("no", 0) if n is None else ("fits", 0) if n == 0 else ("spill", n)


FIT_ORDER = {"fits": 0, "spill": 1, "no": 2}


def browser_info(vram_gb: float, ram_free_gb: float, lang: str, active: list[str], have, downloads=None) -> dict:
    """The Models tab: every catalog model for *this* PC. Runnable ones first, best score in the user's language first
    within each group. have(key) -> bool: the file is on disk; active: the models this server runs."""
    models = []
    for k, m in catalog.MODELS.items():
        state, layers = fit(k, vram_gb, ram_free_gb)
        badges = {t.split("/", 1)[1]: v for t, v in m["scores"].items() if t.split("/", 1)[0] == lang}
        mine = [v for s, v in badges.items() if s != "translate"]          # chrF++ is not a % score
        models.append({
            "key": k, "gb": m["gb"], "note": m.get("note", ""), "fit": state, "experts_in_ram": layers,
            "tok_s": round(catalog.speed(k, vram_gb, ram_free_gb)) if state != "no" else None,
            "score": round(sum(mine) / len(mine), 1) if mine else None, "badges": badges,
            "below_floor": catalog.below_floor(k), "qat": bool(m.get("qat")),
            "downloaded": bool(have(k)), "active": k in active,
            "download": downloads.state(k) if downloads else None})
    models.sort(key=lambda d: (FIT_ORDER[d["fit"]], -(d["score"] if d["score"] is not None else -1.0), d["gb"]))
    best = catalog.pick(vram_gb, lang, ram_free_gb) if vram_gb else None
    return {"lang": lang, "vram_gb": round(vram_gb, 1), "ram_free_gb": round(ram_free_gb, 1), "best": best,
            "active": list(active), "models": models}


class Downloads:
    """Background model downloads for the Models tab, one per model. fetch(key) downloads and checks the file (raises
    or exits on failure); part(key) is the file being written, read for progress."""

    def __init__(self, fetch, part):
        self.fetch, self.part, self.jobs, self.lock = fetch, part, {}, threading.Lock()

    def start(self, key: str) -> dict:
        if key not in catalog.MODELS:
            raise ValueError("not a catalog model")
        with self.lock:
            if (self.jobs.get(key) or {}).get("state") != "downloading":
                self.jobs[key] = {"state": "downloading", "error": None}
                threading.Thread(target=self._run, args=(key,), daemon=True).start()
        return self.state(key)

    def _run(self, key: str) -> None:
        try:
            self.fetch(key)
            done = {"state": "done", "error": None}
        except BaseException as e:          # noqa: BLE001 - cli._fetch reports failures with sys.exit
            done = {"state": "error", "error": str(e) or type(e).__name__}
        with self.lock:
            self.jobs[key] = done

    def state(self, key: str) -> dict | None:
        with self.lock:
            job = dict(self.jobs[key]) if key in self.jobs else None
        if job is None:
            return None
        total = catalog.MODELS[key]["gb"] * 2**30
        part = self.part(key)
        got = part.stat().st_size if job["state"] == "downloading" and part.exists() else 0
        job["pct"] = 100.0 if job["state"] == "done" else round(min(99.9, 100 * got / total), 1)
        return job
