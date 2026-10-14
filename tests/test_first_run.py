"""First run (#116): a small starter model first, the best model downloading behind it, progress with an ETA."""
import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

from localllm import catalog, chat, cli, downloads


def test_eta_wording():
    assert downloads.eta(None) == "" and downloads.eta(-1) == ""
    assert downloads.eta(42) == "about 40 s" and downloads.eta(2) == "about 5 s"
    assert downloads.eta(600) == "about 10 min"


def test_status_line_for_each_state():
    assert downloads.line(None) is None
    st = {"key": "big", "total_gb": 13.3, "done_gb": 6.65, "eta_s": 120, "status": "downloading"}
    assert downloads.line(st) == "downloading big in the background: 50% of 13.3 GB, about 2 min left"
    assert "restart localllm" in downloads.line({**st, "status": "done"})
    assert "run localllm again to resume" in downloads.line({**st, "status": "failed", "error": "reset"})


def test_starter_is_the_smallest_marked_model_that_fits(monkeypatch):
    models = {"big": {"gb": 13.3}, "s1": {"gb": 5.1, "starter": True}, "s2": {"gb": 3.2, "starter": True}}
    monkeypatch.setattr(catalog, "MODELS", models)
    monkeypatch.setattr(catalog, "cpu_moe_layers", lambda k, v, r: 0 if models[k]["gb"] < v else None)
    assert catalog.starter(16) == "s2"
    assert catalog.starter(3) is None


def _plan(monkeypatch, tmp_path, have: set, starter="small", best="big", **args):
    monkeypatch.setattr(cli.runtime, "HOME", tmp_path)
    monkeypatch.delenv("LOCALLLM_MODELS", raising=False)
    monkeypatch.delenv("LOCALLLM_NO_STARTER", raising=False)
    monkeypatch.setattr(cli, "_machine", lambda: (None, [], {"total_gb": 16.0, "id": "Vulkan0"}, 32.0))
    monkeypatch.setattr(cli.runtime, "ram_available_gb", lambda: 20.0)
    monkeypatch.setattr(cli.catalog, "pick", lambda *a, **k: best)
    monkeypatch.setattr(cli.catalog, "starter", lambda *a, **k: starter)
    monkeypatch.setattr(cli, "_have_model", lambda key: tmp_path / key if key in have else None)
    return cli._starter_plan(SimpleNamespace(model=args.get("model"), models=args.get("models")))


def test_first_run_starts_small_and_fetches_the_best_behind_it(monkeypatch, tmp_path):
    assert _plan(monkeypatch, tmp_path, have=set()) == ("small", "big")


def test_no_starter_when_the_best_is_already_here_or_asked_for(monkeypatch, tmp_path):
    assert _plan(monkeypatch, tmp_path, have={"big"}) == ("big", None)
    assert _plan(monkeypatch, tmp_path, have=set(), model="big") == ("big", None)
    assert _plan(monkeypatch, tmp_path, have=set(), starter=None) == ("big", None)
    assert _plan(monkeypatch, tmp_path, have=set(), starter="big") == ("big", None)
    monkeypatch.setenv("LOCALLLM_NO_STARTER", "1")
    assert cli._starter_plan(SimpleNamespace(model=None, models=None)) == (None, None)


BODY = b"GGUF" + bytes(range(256)) * 512


def _serve(chunks_first: int | None = None):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            rng = self.headers.get("Range")
            start = int(rng.split("=")[1].rstrip("-")) if rng else 0
            self.send_response(206 if rng else 200)
            self.send_header("Content-Length", str(len(BODY) - start))
            self.end_headers()
            end = len(BODY) if rng or chunks_first is None else chunks_first
            self.wfile.write(BODY[start:end])
    s = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, f"http://127.0.0.1:{s.server_port}/m.gguf"


def test_background_download_reports_progress_resumes_and_finishes(tmp_path, monkeypatch):
    """Kill-and-resume: the first reply stops short, the download resumes from the .part and checks the SHA-256."""
    s, url = _serve(chunks_first=40000)
    monkeypatch.setattr(cli.time, "sleep", lambda x: None)
    monkeypatch.setitem(catalog.MODELS, "tiny", {"file": "m.gguf", "gb": len(BODY) / 2**30})
    seen, done = [], []
    dest = tmp_path / "m.gguf"
    t = downloads.start("tiny", lambda progress: cli._fetch(url, dest, "m", hashlib.sha256(BODY).hexdigest(),
                                                               progress=lambda *a: (seen.append(a), progress(*a))),
                        on_done=done.append)
    t.join(10)
    st = downloads.state()
    assert st["status"] == "done" and done == ["tiny"] and dest.read_bytes() == BODY
    assert seen and seen[-1][0] == len(BODY) and not (tmp_path / "m.gguf.part").exists()
    s.shutdown(); s.server_close()


def test_background_download_failure_is_reported_not_raised(monkeypatch):
    monkeypatch.setitem(catalog.MODELS, "tiny", {"file": "m.gguf", "gb": 1.0})

    def boom(progress):
        progress(2**29, 2**30, 2**20)
        raise SystemExit("[localllm] download failed 5 times (reset); run the command again to resume it")
    downloads.start("tiny", boom).join(5)
    st = downloads.state()
    assert st["status"] == "failed" and "resume" in downloads.line(st)


def test_chat_status_shows_the_download(monkeypatch):
    monkeypatch.setattr(chat, "_get", lambda url, timeout=10: {"loaded": True, "running": "small",
                                                               "download": "downloading big in the background: 10%"}
                        if url.endswith("/localllm/status") else {})
    assert "downloading big in the background: 10%" in chat.status_lines("http://x", {})


def test_a_starter_fits_a_6_gb_card():
    """#116 needs a starter on small cards too: the smallest marked model must run whole on 6 GB (and none on 2 GB)."""
    key = catalog.starter(6.0, 8.0)
    assert key and catalog.MODELS[key]["starter"] and catalog.MODELS[key]["gb"] <= 3.0      # ~1 min at 50 MB/s
    assert catalog.starter(2.0, 8.0) is None
