"""Switch the running model without a restart (#128): the first run starts on a small starter model while the best
model for the PC downloads (#116); when the download is done, the server moves to the best model between answers.

Switchable sits behind the gateway like the lazy-load pool does (`use`, `ensure_url`, `peek_url`, `keys`). Every
answer holds its lock, so a switch waits for the answer in progress and never cuts one. The switch stops the starter
first (one GPU: both may not fit), starts the new model, and times it. If the new model doesn't start, the starter
comes back and the error is kept, so chat keeps working.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager


class Switchable:
    def __init__(self, key: str, launch):
        """launch(key) -> (process, url) starts llama-server for `key` and returns once it answers (cli._launch)."""
        self.launch = launch
        self.key = key
        self.proc, self.url = launch(key)
        self.keys = [key]
        self.last: dict | None = None       # {"from", "to", "seconds"} or {"from", "to", "error"}
        self._note: str | None = None       # said once, on the first answer after a switch
        self._lock = threading.Lock()

    @contextmanager
    def use(self, body: dict, stay: bool = False):
        with self._lock:
            label = self.key
            if self._note:
                label, self._note = f"{self.key} ({self._note})", None
            yield self.url, label

    def ensure_url(self) -> str:
        return self.url

    def peek_url(self) -> str:
        return self.url

    def _stop(self, proc) -> None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:  # noqa: BLE001 - a process that won't stop in 30 s is killed
            proc.kill()

    def switch(self, key: str) -> dict:
        """Move to `key` once the answer in progress (if any) is done. Returns what happened (also in .last)."""
        with self._lock:
            if key == self.key:
                return self.last or {"from": key, "to": key, "seconds": 0.0}
            old, t0 = self.key, time.perf_counter()
            self._stop(self.proc)
            try:
                self.proc, self.url = self.launch(key)
            except BaseException as e:  # noqa: BLE001 - cli._launch reports with sys.exit; chat must keep working
                self.proc, self.url = self.launch(old)
                self.last = {"from": old, "to": key, "error": str(e)[:200] or type(e).__name__}
                return self.last
            seconds = round(time.perf_counter() - t0, 1)
            self.key, self.keys = key, [key]
            self.last = {"from": old, "to": key, "seconds": seconds}
            self._note = f"switched from {old} in {seconds:g} s: the best model for this PC"
            return self.last

    def close(self) -> None:
        self._stop(self.proc)
