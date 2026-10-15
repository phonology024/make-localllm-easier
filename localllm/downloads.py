"""Background model download for the first run (#116): chat on a small starter model within about a minute while the
best model for this PC downloads behind it, resumable and checksum-verified like every other download.

state() is what `/status`, the chat page and `GET /localllm/status` show:
  {"key", "file", "total_gb", "done_gb", "mb_s", "eta_s", "status": "downloading" | "done" | "failed", "error"}
"""
from __future__ import annotations

import threading
import time

_lock = threading.Lock()
_state: dict | None = None


def state() -> dict | None:
    with _lock:
        return dict(_state) if _state else None


def _set(**kw) -> None:
    global _state
    with _lock:
        _state = {**(_state or {}), **kw}


def eta(seconds: float | None) -> str:
    """'about 3 min' / 'about 40 s' / '' for unknown."""
    if seconds is None or seconds < 0:
        return ""
    if seconds >= 90:
        return f"about {round(seconds / 60)} min"
    return f"about {max(5, round(seconds / 5) * 5)} s"


def line(st: dict | None) -> str | None:
    """One line for /status and the chat page, or None when nothing is downloading."""
    if not st:
        return None
    if st["status"] == "done":
        return f"{st['key']} is downloaded: restart localllm to switch to it (the best model for this PC)"
    if st["status"] == "switching":
        return f"{st['key']} is downloaded: switching to it after the answer in progress (the best model for this PC)"
    if st["status"] == "switched":
        return f"now running {st['key']}, the best model for this PC (switched in {st.get('switch_s', 0):g} s)"
    if st["status"] == "switch_failed":
        return (f"{st['key']} is downloaded but didn't start ({st.get('error')}); still on the starter model, "
                "restart localllm to try again")
    if st["status"] == "failed":
        return f"downloading {st['key']} stopped ({st.get('error')}); run localllm again to resume it"
    pct = 100 * st["done_gb"] / st["total_gb"] if st.get("total_gb") else 0
    left = eta(st.get("eta_s"))
    return (f"downloading {st['key']} in the background: {pct:.0f}% of {st.get('total_gb', 0):.1f} GB"
            + (f", {left} left" if left else ""))


def mark(**kw) -> None:
    """Record what happened after the download (#128: switching / switched / switch_failed)."""
    _set(**kw)


def start(key: str, fetch, on_done=None) -> threading.Thread:
    """Download catalog model `key` in a daemon thread. fetch(progress) does the work (cli._fetch with this progress
    callback, raising on failure); on_done(key) runs once the file is in place."""
    from . import catalog
    m = catalog.MODELS[key]
    _set(key=key, file=m["file"], total_gb=m["gb"], done_gb=0.0, mb_s=None, eta_s=None, status="downloading", error=None)

    def progress(done: int, total: int, bytes_per_s: float) -> None:
        left = (total - done) / bytes_per_s if bytes_per_s and total else None
        _set(done_gb=round(done / 2**30, 2), total_gb=round(total / 2**30, 2) if total else m["gb"],
             mb_s=round(bytes_per_s / 2**20, 1), eta_s=round(left) if left is not None else None)

    def work() -> None:
        try:
            fetch(progress)
        except BaseException as e:  # noqa: BLE001 - a daemon thread must report, not die silently (SystemExit too)
            _set(status="failed", error=str(e)[:200] or type(e).__name__)
            return
        _set(status="done", done_gb=m["gb"], eta_s=0, finished=time.time())
        if on_done:
            on_done(key)

    t = threading.Thread(target=work, daemon=True, name=f"download-{key}")
    t.start()
    return t
