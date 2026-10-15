"""Update notices (#120): a newer localllm, or a better measured model for your language that now fits your PC.

At most once a day, one small request: catalog.json on the project website (the version on main and every measured
model with its sizes and scores; site/build.py publishes it with each deploy). It runs in the background, so starting
localllm never waits for it; the notice shows from the saved answer, as one line in the terminal and in the chat
page's status box. Nothing is ever downloaded or installed by this: the notice says what command would do it.

Off: `localllm updates off` (remembered in ~/.localllm/updates.json) or LOCALLLM_NO_UPDATE_CHECK=1.
Offline: the request fails quietly and is tried again the next day. Nothing about this PC is sent.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request
from pathlib import Path

from . import __version__, catalog, runtime

URL = "https://phonology024.github.io/make-localllm-easier/catalog.json"
EVERY_S = 24 * 3600
TIMEOUT_S = 5
MAX_BYTES = 2 * 2**20          # the catalog is a few KB; anything bigger is not ours


def _file() -> Path:
    return runtime.HOME / "updates.json"


def _load() -> dict:
    try:
        d = json.loads(_file().read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(d: dict) -> None:
    _file().parent.mkdir(parents=True, exist_ok=True)
    _file().write_text(json.dumps(d, indent=1), encoding="utf-8")


def enabled() -> bool:
    return not os.environ.get("LOCALLLM_NO_UPDATE_CHECK") and _load().get("enabled", True)


def set_enabled(on: bool) -> str:
    d = _load()
    d["enabled"] = on
    _save(d)
    return "update notices on (one request a day to GitHub)" if on else "update notices off: no requests are made"


def _version(v: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def fetch(opener=urllib.request.urlopen, now: float | None = None, force: bool = False) -> bool:
    """Ask GitHub once a day (force: now, for `localllm updates check`); True if a request was made. Failures
    (offline, timeouts, junk) are quiet."""
    now = time.time() if now is None else now
    d = _load()
    last = d.get("checked")                      # None: never asked
    if not enabled() or (not force and isinstance(last, (int, float)) and now - last < EVERY_S):
        return False
    d["checked"] = now
    try:
        with opener(URL, timeout=TIMEOUT_S) as r:
            remote = json.loads(r.read(MAX_BYTES + 1)[:MAX_BYTES])
        if isinstance(remote, dict) and isinstance(remote.get("models"), dict):
            d["remote"] = {"version": str(remote.get("version", "")), "models": remote["models"]}
    except (OSError, ValueError):
        pass
    _save(d)
    return True


def fetch_in_background() -> None:
    threading.Thread(target=fetch, daemon=True).start()


def _better_model(remote: dict, vram_gb: float, lang: str, ram_free_gb: float) -> tuple[str, float] | None:
    """A model the newer catalog has, this version doesn't, that fits this PC and scores clearly higher in `lang`."""
    new = {k: m for k, m in remote.items() if k not in catalog.MODELS and isinstance(m, dict)}
    if not new or not vram_gb:
        return None
    mine = catalog.pick(vram_gb, lang, ram_free_gb)
    cat = _private_catalog({**new, **catalog.MODELS})
    try:                                    # score and fit the new entries with the very same rules
        ok = [k for k in new if cat.cpu_moe_layers(k, vram_gb, ram_free_gb) is not None and not cat.below_floor(k)]
        if not ok:
            return None
        best = max(ok, key=lambda k: cat.score(k, lang))
        gain = cat.score(best, lang) - (cat.score(mine, lang) if mine else 0.0)
    except (KeyError, TypeError, ValueError):   # an entry this version can't read: no notice
        return None
    return (best, round(gain, 1)) if gain > catalog.TIE_POINTS else None


def _private_catalog(models: dict):
    """A separate copy of the catalog module whose MODELS is `models`. The real catalog.MODELS is never swapped, even
    for a moment: the server's routing threads read it while this runs at startup."""
    import importlib.util
    spec = importlib.util.find_spec(catalog.__name__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.MODELS = models
    return mod


def notice(vram_gb: float = 0.0, lang: str = "en", ram_free_gb: float = 0.0) -> str | None:
    """One line, or None. Uses only what fetch() saved; never makes a request itself."""
    if not enabled():
        return None
    remote = _load().get("remote") or {}
    newer = remote.get("version") and _version(remote["version"]) > _version(__version__)
    better = _better_model(remote.get("models") or {}, vram_gb, lang, ram_free_gb) if newer else None
    if better:
        return (f"a better measured model for {lang} now fits this PC: {better[0]} (+{better[1]:g} points) - "
                "update with: pip install -U make-localllm-easier")
    if newer:
        return f"localllm {remote['version']} is out (you have {__version__}): pip install -U make-localllm-easier"
    return None
