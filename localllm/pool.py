"""Lazy-load model pool (0.6): several models, one GPU. Only one model is loaded at a time; when the router picks
another one for a message, the loaded llama-server is stopped and the new one started (the file usually comes back
from the OS page cache). Requests are served one at a time (llama-server runs with a single slot anyway), so a swap
never cuts off an answer in progress. Every swap is timed and reported in the X-Localllm-Model header.

Prefetch: llama-server loads with --load-mode none, so the loaded model doesn't need its file in the OS page cache once
it is on the GPU. Right after each swap we read the *other* models' files in the background (only when RAM is free),
so the next swap reads from RAM instead of disk - two models alternate warm even when RAM can't cache both at once.

Resident mode: when all models fit in VRAM together (24 GB+ cards for a 12 + 13 GB pair), every model stays loaded and
each message goes straight to its best model - no swap, so no hysteresis either.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from pathlib import Path

from . import router

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
                 resident: bool = False, files: dict | None = None, classifier=None):
        """launch(key) -> (Popen, url) starts a llama-server for `key` and returns once it is healthy.
        files: {key: model path} to prefetch the models that aren't loaded (lazy mode only).
        classifier: taskclf.Classifier that tells the task (math, code, ...) when the keyword rules can't."""
        self.keys, self.launch, self.vram_gb, self.ram_free_gb = keys, launch, vram_gb, ram_free_gb
        self.classifier = classifier
        self.files = files or {}
        self._stop_prefetch = threading.Event()
        self.prefetched: list[str] = []
        self.lock = threading.Lock()
        self.proc = self.url = self.current = None
        self.swaps: list[float] = []
        self.loaded: dict[str, tuple] = {k: launch(k) for k in keys} if resident else {}
        if resident:
            self.current = first or keys[0]
            self.proc, self.url = self.loaded[self.current]
        else:
            self._swap(first or keys[0])

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

    @contextmanager
    def use(self, body: dict):
        """Hold the GPU for one request: pick the model for it, swap if needed, yield (url, header label)."""
        with self.lock:
            key, why = router.pick_local(router._last_user(body), self.keys, None if self.loaded else self.current,
                                         self.vram_gb, self.ram_free_gb, self.classifier)
            label = f"{key} ({why})"
            if self.loaded:
                self.current = key
                self.proc, self.url = self.loaded[key]
            elif key != self.current:
                self._swap(key)
                label += f", swapped in {self.swaps[-1]:.1f}s"
            yield self.url, label

    def close(self) -> None:
        self._stop_prefetch.set()
        if self.classifier is not None:
            self.classifier.close()
        for p, _url in self.loaded.values() or ([(self.proc, None)] if self.proc is not None else []):
            p.terminate()
