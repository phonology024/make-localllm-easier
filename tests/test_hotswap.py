"""#128: the starter model moves to the best one when its download is done - between answers, never mid-answer.
Fake llama-servers only (no model, no GPU)."""
import json
import threading
import time
import urllib.request

from localllm import cli, downloads, gateway, hotswap
from test_gateway import _up


class Proc:
    def __init__(self, server):
        self.server, self.stopped = server, False

    def terminate(self):
        self.stopped = True
        self.server.shutdown()

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.stopped = True


def launcher(log, fail=()):
    def launch(key):
        log.append(f"start {key}")
        if key in fail:
            raise SystemExit(f"[localllm] llama-server stopped (exit 1) for {key}")
        s, url = _up()
        return Proc(s), url
    return launch


def _chat(base):
    req = urllib.request.Request(base + "/v1/chat/completions", headers={"Content-Type": "application/json"},
                                 data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode())
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.headers.get("X-Localllm-Model"), json.load(r)


def test_switch_moves_the_gateway_and_says_so_once():
    log = []
    sw = hotswap.Switchable("starter-4b", launcher(log))
    g = gateway.serve(sw, port=0, model_name="starter-4b")
    base = f"http://127.0.0.1:{g.server_address[1]}"
    try:
        assert _chat(base)[0] == "starter-4b"
        first = sw.proc
        r = sw.switch("best-26b")
        assert r["from"] == "starter-4b" and r["to"] == "best-26b" and r["seconds"] >= 0 and first.stopped
        label, body = _chat(base)
        assert label.startswith("best-26b (switched from starter-4b in ") and "best model for this PC" in label
        assert body["choices"][0]["message"]["content"] == "echo: hi"
        assert _chat(base)[0] == "best-26b"                                          # the note is said once
        models = json.load(urllib.request.urlopen(base + "/v1/models", timeout=10))
        assert [m["id"] for m in models["data"]] == ["best-26b"]
        assert log == ["start starter-4b", "start best-26b"]
    finally:
        g.shutdown(); sw.close()


def test_a_switch_waits_for_the_answer_in_progress():
    sw = hotswap.Switchable("starter-4b", launcher([]))
    events = []
    with sw.use({}) as (url, _label):                                               # an answer is streaming
        t = threading.Thread(target=lambda: events.append(("switched", sw.switch("best-26b"))))
        t.start()
        time.sleep(0.3)
        assert events == [] and sw.key == "starter-4b"                              # not cut off
        events.append(("answer done", url))
    t.join(10)
    assert [e[0] for e in events] == ["answer done", "switched"] and sw.key == "best-26b"
    sw.close()


def test_a_model_that_does_not_start_brings_the_starter_back():
    log = []
    sw = hotswap.Switchable("starter-4b", launcher(log, fail={"best-26b"}))
    r = sw.switch("best-26b")
    assert "error" in r and "llama-server stopped" in r["error"] and sw.key == "starter-4b"
    assert log == ["start starter-4b", "start best-26b", "start starter-4b"]
    with sw.use({}) as (url, label):
        assert label == "starter-4b" and url.startswith("http://127.0.0.1:")
    sw.close()


def test_cli_switches_when_the_download_finishes_and_reports_it(monkeypatch):
    said = []
    monkeypatch.setattr(cli, "_say", said.append)
    sw = hotswap.Switchable("starter-4b", launcher([]))
    downloads.mark(key="best-26b", file="x.gguf", total_gb=13.3, done_gb=13.3, status="done", error=None)
    r = cli._switch_to(sw, "best-26b")
    st = downloads.state()
    assert st["status"] == "switched" and st["switch_s"] == r["seconds"]
    assert downloads.line(st).startswith("now running best-26b, the best model for this PC (switched in ")
    assert said[0].startswith("best-26b is downloaded - switching") and said[-1].startswith("now running best-26b")
    sw.close()

    bad = hotswap.Switchable("starter-4b", launcher([], fail={"best-26b"}))
    cli._switch_to(bad, "best-26b")
    assert downloads.state()["status"] == "switch_failed" and "restart localllm" in downloads.line(downloads.state())
    bad.close()
