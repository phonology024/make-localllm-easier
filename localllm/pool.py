"""Lazy-load model pool (0.6): several models, one GPU. Only one model is loaded at a time; when the router picks
another one for a message, the loaded llama-server is stopped and the new one started (the file usually comes back
from the OS page cache). Requests are served one at a time (llama-server runs with a single slot anyway), so a swap
never cuts off an answer in progress. Every swap is timed and reported in the X-Localllm-Model header.

Prefetch: llama-server loads with --load-mode none, so the loaded model doesn't need its file in the OS page cache once
it is on the GPU. Right after each swap we read the *other* models' files in the background (only when RAM is free),
so the next swap reads from RAM instead of disk - two models alternate warm even when RAM can't cache both at once.

Resident mode: when all models fit in VRAM together (24 GB+ cards for a 12 + 13 GB pair), every model stays loaded and
each message goes straight to its best model - no swap, so no hysteresis either.

Idle unload: after `idle_unload_s` without a request the loaded llama-server(s) stop, so the GPU and RAM are free for
games or other apps; the next message loads the model it needs (reported as "loaded after idle in X s"). GETs (health
checks, model lists, a polling web page) are not activity: they neither reset the idle clock nor load a model. Requests hold
the pool's lock, so nothing is ever unloaded in the middle of an answer.

Stay: a request with `stay=True` (header `X-Localllm-Stay: 1`, `/stay` in `localllm chat`) keeps the loaded model even
when another one scores better for it - for a conversation that should not pause for a swap.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from pathlib import Path

from . import router

IDLE_UNLOAD_S = 15 * 60
PREFETCH_CHUNK = 64 * 2**20
PREFETCH_SPARE_GB = 4.0     # leave this much RAM free for everything else


def prefetch(path: Path, stop: threading.Event, ram_free_gb=None) -> bool:
    """Read a file once so the OS keeps it in the page cache. Skips it when RAM is short; stops early on `stop`."""
    from . import runtime
    free = runtime.ram_available_gb() if ram_free_gb is None else ram_free_gb
    if free < path.stat().st_size / 2**30 + PREFETCH_SPARE_GB:
        return False
    with open(path, "rb", buffering=0) as f:
        while not stop.is_set() and f.read(PREFETCH_CHUNK):
            pass
    return not stop.is_set()


class Pool:
    def __init__(self, keys: list[str], launch, vram_gb: float, ram_free_gb: float = 0.0, first: str | None = None,
                 resident: bool = False, files: dict | None = None, idle_unload_s: float = IDLE_UNLOAD_S):
        """launch(key) -> (Popen, url) starts a llama-server for `key` and returns once it is healthy.
        files: {key: model path} to prefetch the models that aren't loaded (lazy mode only).
        idle_unload_s: stop the servers after this long without a request (0 = never)."""
        self.keys, self.launch, self.vram_gb, self.ram_free_gb = keys, launch, vram_gb, ram_free_gb
        self.files = files or {}
        self._stop_prefetch = threading.Event()
        self.prefetched: list[str] = []
        self.lock = threading.Lock()
        self.proc = self.url = self.current = None
        self.swaps: list[float] = []
        self.resident, self.first = resident, first or keys[0]
        self.loaded: dict[str, tuple] = {k: launch(k) for k in keys} if resident else {}
        if resident:
            self.current = self.first
            self.proc, self.url = self.loaded[self.current]
        else:
            self._swap(self.first)
        self.idle_unload_s, self.unloads, self.last_used = idle_unload_s, 0, time.time()
        self._closed = threading.Event()
        if idle_unload_s:
            threading.Thread(target=self._watch_idle, daemon=True).start()

    def _is_loaded(self) -> bool:
        return bool(self.loaded) or self.proc is not None

    def _watch_idle(self) -> None:
        while not self._closed.wait(min(30.0, self.idle_unload_s / 4)):
            with self.lock:                         # held by every request: never unloads mid-answer
                if self._is_loaded() and time.time() - self.last_used >= self.idle_unload_s:
                    self._unload()

    def _unload(self) -> None:
        self._stop_prefetch.set()
        for p, _url in list(self.loaded.values()) or [(self.proc, self.url)]:
            p.terminate()
            p.wait(30)
        self.loaded, self.proc, self.url = {}, None, None
        self.unloads += 1

    def _wake(self, key: str) -> str:
        """Load again after an idle unload: every model in resident mode, else `key`. Returns a label part."""
        t = time.time()
        if self.resident:
            self.loaded = {k: self.launch(k) for k in self.keys}
        else:
            self._swap(key)
            self.swaps.pop()                        # not a model switch
        return f"loaded after idle in {time.time() - t:.1f}s"

    def _swap(self, key: str) -> None:
        t = time.time()
        self._stop_prefetch.set()                 # the disk is needed for the load now
        if self.proc is not None:
            self.proc.terminate()
            self.proc.wait(30)
        self.proc, self.url = self.launch(key)
        self.current = key
        self.swaps.append(round(time.time() - t, 2))
        others = [self.files[k] for k in self.keys if k != key and k in self.files]
        if others:
            self._stop_prefetch = stop = threading.Event()
            threading.Thread(target=self._prefetch_all, args=(others, stop), daemon=True).start()

    def _prefetch_all(self, paths: list, stop: threading.Event) -> None:
        for p in paths:
            if prefetch(Path(p), stop):
                self.prefetched.append(Path(p).name)

    def peek_url(self) -> str | None:
        """URL of the loaded model, or None while unloaded. No lock, no wake-up, not activity: for GETs such as /health
        or a polling web page, which must neither reload a model nor keep one in memory."""
        if self.resident:
            return (self.loaded.get(self.current or self.first) or (None, None))[1]
        return self.url

    def ensure_url(self) -> str:
        """URL of the current model, loading it again if it was unloaded while idle (non-chat POSTs: completions,
        embeddings, token counts - real work, so it counts as activity)."""
        with self.lock:
            self.last_used = time.time()
            if not self._is_loaded():
                self._wake(self.current or self.first)
            if self.resident:
                self.proc, self.url = self.loaded[self.current or self.first]
            return self.url

    @contextmanager
    def use(self, body: dict, stay: bool = False):
        """Hold the GPU for one request: pick the model for it, swap if needed, yield (url, header label).
        stay: keep the current model even if another one is better for this message."""
        with self.lock:
            self.last_used = time.time()
            idle = not self._is_loaded()
            loaded_key = None if (self.resident or idle) else self.current
            key, why = router.pick_local(router._last_user(body), self.keys, loaded_key, self.vram_gb,
                                         self.ram_free_gb)
            if stay and self.current and key != self.current:
                key, why = self.current, f"stayed as asked; {key} would be picked ({why})"
            label = f"{key} ({why})"
            if idle:
                label += ", " + self._wake(key)
            if self.resident:
                self.current = key
                self.proc, self.url = self.loaded[key]
            elif key != self.current:
                self._swap(key)
                label += f", swapped in {self.swaps[-1]:.1f}s"
            try:
                yield self.url, label
            finally:
                self.last_used = time.time()        # idle time counts from the end of the answer

    def close(self) -> None:
        self._closed.set()
        self._stop_prefetch.set()
        for p, _url in self.loaded.values() or ([(self.proc, None)] if self.proc is not None else []):
            p.terminate()
