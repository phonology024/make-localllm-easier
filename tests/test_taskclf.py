import atexit
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from localllm import pool, router, runtime, taskclf
from test_gateway import _gw, _post, _up

FIX = json.loads((Path(__file__).parent / "fixtures" / "taskclf_fixture.json").read_text(encoding="utf-8"))
HEAD, CASES = FIX["head"], FIX["cases"]       # made by fixtures/make_taskclf_fixture.py (numpy + scikit-learn)
Q, G = "qwen3.8-27b-q3", "gemma4-26b-a4b-qat"
POEM = "Write a poem about the sea."


class FakeEmbed(BaseHTTPRequestHandler):
    """Stands in for the e5-small llama-server: the fixture embedding of a known text, the first one otherwise."""
    seen = []

    def log_message(self, *a):
        pass

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self._send({"status": "ok"})

    def do_POST(self):
        inp = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["input"]
        FakeEmbed.seen.append(inp)
        emb = next((c["embedding"] for c in CASES if "query: " + c["text"] == inp), CASES[0]["embedding"])
        self._send({"data": [{"embedding": emb, "index": 0}]})


class FakeProc:
    def __init__(self, server):
        self.server, self.returncode = server, None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.server.shutdown(); self.server.server_close(); self.returncode = 0

    def wait(self, _t=None):
        return self.returncode


def _serve(handler):
    s = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, f"http://127.0.0.1:{s.server_port}"


def _clf(dirs):
    """A Classifier whose 'llama-server' is FakeEmbed; returns it and the list of models it was started with."""
    started = []

    def start(gguf):
        started.append(gguf)
        s, url = _serve(FakeEmbed)
        return FakeProc(s), url
    return taskclf.Classifier(dirs=dirs, start=start), started


@pytest.fixture
def models(tmp_path):
    (tmp_path / taskclf.GGUF_FILE).write_bytes(b"GGUF fake")
    (tmp_path / taskclf.HEAD_FILE).write_text(json.dumps(HEAD), encoding="utf-8")
    return tmp_path


@pytest.fixture(autouse=True)
def no_download_links(monkeypatch):
    monkeypatch.setattr(taskclf, "GGUF_URL", "")
    monkeypatch.setattr(taskclf, "HEAD_URL", "")


def test_head_math_matches_numpy_fixture():
    for c in CASES:
        assert taskclf.probabilities(HEAD, c["embedding"]) == pytest.approx(c["probs"], abs=1e-9)
        label, p = taskclf.predict(HEAD, c["embedding"])
        assert label == c["label"] and p == pytest.approx(c["p"], abs=1e-9)
    with pytest.raises(ValueError):
        taskclf.probabilities(HEAD, [0.1] * 10)                         # wrong embedding size


def test_server_args_cpu_only_on_a_private_port():
    a = taskclf.server_args(Path("e5.gguf"), 4321)
    assert "--embedding" in a and a[a.index("--pooling") + 1] == "mean" and a[a.index("-ngl") + 1] == "0"
    assert a[a.index("-dev") + 1] == "none" and a[a.index("--host") + 1] == "127.0.0.1" and a[a.index("--port") + 1] == "4321"


def test_server_starts_lazily_once_and_gets_prefixed_capped_text(models, monkeypatch):
    registered = []
    monkeypatch.setattr(atexit, "register", registered.append)
    clf, started = _clf([models])
    assert started == []                                                # nothing runs before the first message
    for c in CASES:
        label, p = clf.classify(c["text"])
        assert label == c["label"] and p == pytest.approx(c["p"], abs=1e-9)
    clf.classify("x" * 1000)
    assert FakeEmbed.seen[-1] == "query: " + "x" * 450
    assert started == [models / taskclf.GGUF_FILE] and registered == [clf.close]   # one server, stopped at exit
    clf.close()
    assert clf.proc is None


def test_missing_model_falls_back_to_keywords(tmp_path):
    clf, started = _clf([tmp_path])
    task, how = router.classify_task(POEM, clf)
    assert task == "general" and how.startswith(f"keyword fallback because no {taskclf.GGUF_FILE} in")
    assert router.classify_task("What is 17 * 23?", clf) == ("math", "keyword")   # keyword rules first: no embedding
    assert router.classify_task(POEM) == ("general", "keyword")                    # no classifier at all
    assert started == []


def test_order_is_easy_to_change(models):
    clf, _ = _clf([models])
    assert router.classify_task("What is 17 * 23?", clf) == ("math", "keyword")
    task, how = router.classify_task("What is 17 * 23?", clf, order=("embedding", "keyword"))
    assert (task, how) == (CASES[0]["label"], f"embedding p={CASES[0]['p']:.2f}")   # embedding first: it decides
    assert router.classify_task(CASES[1]["text"], clf, order=("keyword",)) == ("general", "keyword")
    assert router.classify_task(CASES[1]["text"], clf)[0] == "code"
    clf.close()


def test_server_that_exits_falls_back_and_is_not_restarted(models, tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "HOME", tmp_path / "home")
    calls, real = [], taskclf.start_server
    monkeypatch.setattr(taskclf, "start_server", lambda g, s=None: calls.append(g) or real(g, s))
    clf = taskclf.Classifier(server=Path(sys.executable), dirs=[models])   # `python -m <gguf> ...` exits with code 1
    for _ in range(2):
        task, how = router.classify_task(POEM, clf)
        assert task == "general" and "keyword fallback because embedding server exited (code 1)" in how
    assert len(calls) == 1 and (tmp_path / "home" / "router-embed.log").exists()


def test_server_that_dies_later_is_reported(models):
    clf, started = _clf([models])
    assert clf.classify(CASES[1]["text"])[0] == "code"
    clf.proc.terminate(); clf.proc.returncode = 3                       # crashed: the port no longer answers
    for _ in range(2):
        assert router.classify_task(POEM, clf) == ("general", "keyword fallback because embedding server exited (code 3)")
    assert len(started) == 1


def test_bad_head_falls_back(models):
    (models / taskclf.HEAD_FILE).write_text(json.dumps(dict(HEAD, intercept=HEAD["intercept"][:3])), encoding="utf-8")
    clf, started = _clf([models])
    assert "bad router head" in router.classify_task(POEM, clf)[1] and started == []


def test_downloads_missing_files_on_first_use(tmp_path, monkeypatch):
    files = {"/e5.gguf": b"GGUF fake", "/head.json": json.dumps(HEAD).encode()}

    class Files(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            b = files.get(self.path)
            self.send_response(200 if b else 404); self.send_header("Content-Length", str(len(b or b"")))
            self.end_headers(); self.wfile.write(b or b"")

    s, base = _serve(Files)
    monkeypatch.setattr(taskclf, "GGUF_URL", base + "/e5.gguf")
    monkeypatch.setattr(taskclf, "HEAD_URL", base + "/head.json")
    first, home = tmp_path / "first", tmp_path / "models"                 # downloads go to the last folder
    first.mkdir()
    clf, started = _clf([first, home])
    assert clf.classify(CASES[2]["text"])[0] == "translate"
    assert started == [home / taskclf.GGUF_FILE] and json.loads((home / taskclf.HEAD_FILE).read_text()) == HEAD
    clf.close()
    monkeypatch.setattr(taskclf, "HEAD_URL", base + "/missing.json")
    clf, started = _clf([tmp_path / "other"])
    assert f"download of {taskclf.HEAD_FILE} failed" in router.classify_task(POEM, clf)[1] and started == []
    s.shutdown(); s.server_close()


def test_gateway_reports_how_the_task_was_decided(models, tmp_path, monkeypatch):
    monkeypatch.setattr(router, "load_config", lambda: {"enabled": False})

    def launch(_key):
        s, url = _up()
        return FakeProc(s), url

    clf, _ = _clf([models])
    p = pool.Pool([Q, G], launch, 15.9, first=G, classifier=clf)
    g, gurl = _gw(p)
    r = _post(gurl + "/v1/chat/completions", {"messages": [{"role": "user", "content": CASES[1]["text"]}]})
    assert f"(th code by embedding p={CASES[1]['p']:.2f}: " in r.headers["X-Localllm-Model"]
    r = _post(gurl + "/v1/chat/completions", {"messages": [{"role": "user", "content": "What is 17 * 23?"}]})
    assert "(en math by keyword: " in r.headers["X-Localllm-Model"]
    p.close(); g.shutdown()
    assert clf.proc is None                                             # closing the pool stops the embedding server

    broken, _ = _clf([tmp_path / "โมเดล"])                              # no model there; the header stays ASCII
    p = pool.Pool([Q, G], launch, 15.9, first=G, classifier=broken)
    g, gurl = _gw(p)
    r = _post(gurl + "/api/chat", {"messages": [{"role": "user", "content": POEM}], "stream": False})
    hdr = r.headers["X-Localllm-Model"]
    assert hdr.startswith(G) and f"(en general by keyword fallback because no {taskclf.GGUF_FILE} in " in hdr
    assert hdr.isascii() and json.load(r)["message"]["content"] == "echo: " + POEM
    p.close(); g.shutdown()
