"""Embedding task classifier (#35): is a message general, math, code or translate - in any language, in milliseconds.

multilingual-e5-small (MIT) as a Q8_0 GGUF (126 MB) runs in its own small llama-server on the CPU (--embedding
--pooling mean -ngl 0 -dev none, so it never takes VRAM from the chat model) on a free private port. It starts on the
first message that needs it, is reused for every message after that, and is stopped when localllm exits. A
logistic-regression head (labels, coef, intercept: tools/router/train_router.py --export) turns the embedding into
(label, probability) in pure Python. Standard library only.

Both files are read from ~/.localllm/models/ (or a $LOCALLLM_MODELS folder) and downloaded there on first use. Anything
that goes wrong raises Unavailable(reason); router.classify_task() then falls back to the keyword rules and says why.
"""
from __future__ import annotations

import atexit
import json
import math
import os
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from . import runtime

GGUF_FILE = "multilingual-e5-small-Q8_0.gguf"
HEAD_FILE = "router_head.json"
# TODO(#35): the maintainer uploads both files to Hugging Face and puts the links here. Until then the URLs are empty:
# nothing is downloaded and routing uses the keyword rules, unless both files are already in ~/.localllm/models/
# (the "router" CI workflow builds them as artifacts: tools/router/README.md).
GGUF_URL = ""   # e.g. https://huggingface.co/<user>/<repo>/resolve/main/multilingual-e5-small-Q8_0.gguf
HEAD_URL = ""   # e.g. https://huggingface.co/<user>/<repo>/resolve/main/router_head.json
PREFIX, MAX_CHARS = "query: ", 450   # e5 inputs need the prefix; the head was trained on the first 450 characters
START_TIMEOUT_S = 60


class Unavailable(Exception):
    """The embedding path can't answer this message; str(e) is a short ASCII reason (it goes into an HTTP header)."""


def _reason(msg) -> str:
    return " ".join(str(msg).split()).encode("ascii", "replace").decode()[:160]


def _show(p: Path) -> str:
    home = str(Path.home())
    return "~" + str(p)[len(home):] if str(p).startswith(home) else str(p)


# ---- head math (pure functions, unit-tested against numpy) -----------------------------------------------------------

def load_head(path) -> dict:
    h = json.loads(Path(path).read_text(encoding="utf-8"))
    n = len(h["labels"])
    if n < 2 or len(h["coef"]) != n or len(h["intercept"]) != n or len({len(r) for r in h["coef"]}) != 1:
        raise ValueError(f"bad router head {Path(path).name}: needs one coef row and one intercept per label")
    return h


def probabilities(head: dict, emb: list[float]) -> list[float]:
    """softmax(W x/|x| + b): sklearn's multinomial LogisticRegression.predict_proba on the L2-normalised embedding."""
    if len(emb) != len(head["coef"][0]):
        raise ValueError(f"embedding has {len(emb)} dims, the router head expects {len(head['coef'][0])}")
    norm = math.sqrt(sum(v * v for v in emb)) or 1.0
    z = [b + sum(w * v for w, v in zip(row, emb)) / norm for row, b in zip(head["coef"], head["intercept"])]
    top = max(z)
    e = [math.exp(v - top) for v in z]
    s = sum(e)
    return [v / s for v in e]


def predict(head: dict, emb: list[float]) -> tuple[str, float]:
    p = probabilities(head, emb)
    i = max(range(len(p)), key=p.__getitem__)      # first maximum, like numpy.argmax
    return head["labels"][i], p[i]


# ---- the embedding server --------------------------------------------------------------------------------------------

def server_args(gguf: Path, port: int) -> list[str]:
    # one slot of 512 tokens: e5-small's whole window, and "query: " + 450 characters always fits in one batch
    return ["-m", str(gguf), "--host", "127.0.0.1", "--port", str(port), "--embedding", "--pooling", "mean",
            "-ngl", "0", "-dev", "none", "-c", "512", "-b", "512", "-ub", "512", "-np", "1"]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(gguf: Path, server: Path | None = None) -> tuple[subprocess.Popen, str]:
    """Start the embedding llama-server on a free private port; return (process, url) once /health says ok."""
    server = server or runtime.find_server()
    port = _free_port()
    runtime.HOME.mkdir(parents=True, exist_ok=True)
    log = runtime.HOME / "router-embed.log"
    with open(log, "ab") as f:
        proc = subprocess.Popen([str(server), *server_args(gguf, port)], stdout=f, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + START_TIMEOUT_S
    while time.time() < deadline:
        if proc.poll() is not None:
            raise Unavailable(f"embedding server exited (code {proc.returncode}), log: {_show(log)}")
        try:
            if b'"ok"' in urllib.request.urlopen(url + "/health", timeout=2).read():
                return proc, url
        except OSError:
            pass
        time.sleep(0.05)
    proc.kill()
    raise Unavailable(f"embedding server did not answer within {START_TIMEOUT_S} s")


def model_dirs() -> list[Path]:
    """$LOCALLLM_MODELS folders first (like the chat models), then ~/.localllm/models, where downloads go."""
    return [Path(d) for d in os.environ.get("LOCALLLM_MODELS", "").split(os.pathsep) if d] + [runtime.HOME / "models"]


class Classifier:
    """classify(text) -> (label, probability). Nothing runs until the first call; then the files are found (or
    downloaded) and the server started once, and reused. A failure is remembered: later calls raise Unavailable at
    once instead of retrying on every message. start(gguf) -> (process, url) replaces the real server in tests."""

    def __init__(self, server: Path | None = None, dirs: list | None = None, start=None, timeout: float = 10.0):
        self.server, self.timeout = server, timeout
        self.dirs = [Path(d) for d in dirs] if dirs is not None else model_dirs()
        self._start = start or (lambda gguf: start_server(gguf, self.server))
        self.lock = threading.Lock()
        self.head = self.proc = self.url = None
        self.error: str | None = None
        self.starts = 0

    def files(self) -> tuple[Path, Path]:
        """(gguf, head) from the model folders; a missing one is downloaded into the last folder."""
        out = []
        for name, url in ((GGUF_FILE, GGUF_URL), (HEAD_FILE, HEAD_URL)):
            path = next((d / name for d in self.dirs if (d / name).is_file()), None)
            if path is None:
                if not url:
                    raise Unavailable(f"no {name} in {_show(self.dirs[-1])} (download link not set yet)")
                path = self.dirs[-1] / name
                print(f"[localllm] downloading the task router's {name} (one time) ...", flush=True)
                try:
                    from .cli import _download
                    _download(url, path, name)
                except Exception as e:
                    raise Unavailable(f"download of {name} failed: {e}") from None
            out.append(path)
        return out[0], out[1]

    def _ready(self) -> str:
        with self.lock:
            if self.error:
                raise Unavailable(self.error)
            if self.url is None:
                try:
                    gguf, head = self.files()
                    self.head = load_head(head)
                    self.starts += 1
                    self.proc, self.url = self._start(gguf)
                except Exception as e:      # a missing binary, a bad file, a crash: never break the chat itself
                    self.error = _reason(e if isinstance(e, Unavailable) else f"{type(e).__name__}: {e}")
                    raise Unavailable(self.error) from None
                if self.starts == 1:
                    atexit.register(self.close)
            return self.url

    def classify(self, text: str) -> tuple[str, float]:
        url = self._ready()
        inp = self.head.get("prefix", PREFIX) + text[:self.head.get("max_chars", MAX_CHARS)]
        req = urllib.request.Request(url + "/v1/embeddings", data=json.dumps({"input": inp}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            emb = json.load(urllib.request.urlopen(req, timeout=self.timeout))["data"][0]["embedding"]
        except Exception as e:      # one slow or failed request falls back for this message; a dead server for good
            if self.proc is not None and self.proc.poll() is not None:
                self.error = _reason(f"embedding server exited (code {self.proc.returncode})")
            raise Unavailable(self.error or _reason(f"embedding failed: {type(e).__name__}: {e}")) from None
        try:
            return predict(self.head, emb)
        except (ValueError, TypeError) as e:    # head and model don't match: no point trying again
            self.error = _reason(e)
            raise Unavailable(self.error) from None

    def close(self) -> None:
        proc, self.proc, self.url = self.proc, None, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
