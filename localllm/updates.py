"""Update notices (#120): a newer localllm, or a better measured model for your language that now fits your PC.

At most once a day, two small requests to the project website, and nothing about you or this PC is sent:
latest.json (the localllm version on main) and catalog/v1/catalog.json (the published measured catalog, #164; the
same file other tools read). site/build.py publishes both with each deploy. It runs in the background, so starting
localllm never waits for it; the notice shows from the saved answer, as one line in the terminal and in the chat
page's status box. Nothing is ever downloaded or installed by this: the notice says what command would do it.

Off: `localllm updates off` (remembered in ~/.localllm/updates.json) or LOCALLLM_NO_UPDATE_CHECK=1.
Offline: the requests fail quietly and are tried again the next day.
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

SITE = "https://phonology024.github.io/make-localllm-easier"
LATEST_URL = f"{SITE}/latest.json"                 # {"version": "0.7.2"}
CATALOG_URL = f"{SITE}/catalog/v1/catalog.json"    # data/catalog/v1/schema.json
EVERY_S = 24 * 3600
TIMEOUT_S = 5
MAX_BYTES = 2 * 2**20          # the catalog is tens of KB; anything bigger is not ours


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
    return ("update notices on (two small requests a day to the project website, nothing about you is sent)" if on
            else "update notices off: no requests are made")


def _version(v: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def _get(opener, url: str):
    with opener(url, timeout=TIMEOUT_S) as r:
        return json.loads(r.read(MAX_BYTES + 1)[:MAX_BYTES])


def from_v1(entry: dict) -> tuple[str, dict]:
    """One catalog/v1 model entry -> (key, the catalog.MODELS shape the fit and score rules read). The inverse of
    tools/build_catalog.py model_entry; tests/test_updates.py checks every rule gives the same answer on both."""
    mem, speed = entry["memory"], entry["speed"][0]
    m = {"repo": entry["hf_repo"], "file": entry["file"], "rev": entry["rev"], "sha256": entry["sha256"],
         "bpw": entry["bits_per_weight"], "qat": entry["qat"], "max_ctx": entry["max_ctx"],
         "gb": mem["weights_gib"], "kv_kb_per_token": mem["kv_kib_per_token_q8"],
         "fixed_cache_gb": mem["fixed_cache_gib"], "checkpoint_gb": mem["checkpoint_gib"],
         "build": speed["llama_cpp"], "tok_s_9070xt": speed["decode_tok_s"], "mtp": bool(speed.get("mtp_draft_tokens")),
         "scores": dict(entry["scores"]), "note": entry.get("note", "")}
    if mem.get("cpu_mapped_gib") is not None:
        m["cpu_mapped_gb"] = mem["cpu_mapped_gib"]
    if entry.get("moe"):
        m["moe"] = {"layers": entry["moe"]["layers"], "expert_gb_per_layer": entry["moe"]["expert_gib_per_layer"]}
    if speed.get("moe_offload"):
        m["tok_s_offload"] = {o["layers_in_ram"]: o["decode_tok_s"] for o in speed["moe_offload"]}
    if entry.get("vision"):
        v = entry["vision"]
        m.update(mmproj=v["mmproj_file"], mmproj_gb=v["mmproj_gib"], mmproj_sha256=v["mmproj_sha256"])
    return entry["id"], m


def fetch(opener=urllib.request.urlopen, now: float | None = None, force: bool = False) -> bool:
    """Ask the website once a day (force: now, for `localllm updates check`); True if requests were made. Failures
    (offline, timeouts, junk, a catalog entry this version can't read) are quiet and keep what was saved before."""
    now = time.time() if now is None else now
    d = _load()
    last = d.get("checked")                      # None: never asked
    if not enabled() or (not force and isinstance(last, (int, float)) and now - last < EVERY_S):
        return False
    d["checked"] = now
    remote = dict(d.get("remote") or {})
    try:
        latest = _get(opener, LATEST_URL)
        if isinstance(latest, dict) and isinstance(latest.get("version"), str):
            remote["version"] = latest["version"]
    except (OSError, ValueError):
        pass
    try:
        cat = _get(opener, CATALOG_URL)
        if isinstance(cat, dict) and cat.get("catalog_version") == 1 and isinstance(cat.get("models"), list):
            remote["models"] = dict(from_v1(e) for e in cat["models"])
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
        pass
    if remote:
        d["remote"] = remote
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
