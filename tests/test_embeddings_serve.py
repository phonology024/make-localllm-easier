"""#158: `localllm serve --embeddings` answers /v1/embeddings with the router's e5 server. Fake servers only."""
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from localllm import cli, gateway, taskclf
from test_gateway import _up


class FakeEmbed(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        out = json.dumps({"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
                          "model": "e5", "usage": {"prompt_tokens": len(str(body["input"]).split())}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)


def _embed_server():
    s = HTTPServer(("127.0.0.1", 0), FakeEmbed)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, f"http://127.0.0.1:{s.server_port}"


def _post(base, body):
    req = urllib.request.Request(base + "/v1/embeddings", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.headers.get("X-Localllm-Route"), json.load(r)


def test_embeddings_go_to_the_e5_server_and_chat_still_goes_to_the_model():
    chat, curl = _up()
    emb, eurl = _embed_server()
    g = gateway.serve(curl, port=0, model_name="m", embeddings=lambda: eurl)
    base = f"http://127.0.0.1:{g.server_address[1]}"
    try:
        route, out = _post(base, {"input": "passage: สวัสดี", "model": "anything"})
        assert route == "local (embeddings: multilingual-e5-small)" and out["data"][0]["embedding"] == [0.1, 0.2, 0.3]
        req = urllib.request.Request(base + "/v1/chat/completions", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode())
        assert json.load(urllib.request.urlopen(req, timeout=30))["choices"][0]["message"]["content"] == "echo: hi"
    finally:
        g.shutdown(); chat.shutdown(); emb.shutdown()


def test_unavailable_embeddings_say_why_with_503():
    chat, curl = _up()

    def down():
        raise taskclf.Unavailable("download of multilingual-e5-small-Q8_0.gguf failed: offline")

    g = gateway.serve(curl, port=0, model_name="m", embeddings=down)
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(f"http://127.0.0.1:{g.server_address[1]}", {"input": "x"})
        assert e.value.code == 503 and b"offline" in e.value.read()
    finally:
        g.shutdown(); chat.shutdown()


def _files(tmp_path):
    (tmp_path / taskclf.GGUF_FILE).write_bytes(b"gguf")
    (tmp_path / taskclf.HEAD_FILE).write_text(json.dumps(
        {"labels": ["general", "math"], "coef": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], "intercept": [0.0, 0.0]}))
    return [tmp_path]


def test_embeddings_url_waits_for_the_server_to_start(tmp_path):
    emb, eurl = _embed_server()

    def slow_start(gguf):
        time.sleep(0.6)                                  # longer than the router's own short wait
        return type("P", (), {"poll": lambda self: None, "terminate": lambda self: None,
                              "wait": lambda self, t=None: 0})(), eurl

    clf = taskclf.Classifier(dirs=_files(tmp_path), start=slow_start, wait=0.05)
    try:
        assert clf.embeddings_url(wait=10, poll=0.05) == eurl
    finally:
        clf.close(); emb.shutdown()


def test_embeddings_url_gives_up_on_a_permanent_failure(tmp_path):
    def broken(gguf):
        raise taskclf.Unavailable("embedding server exited (code 1)")

    clf = taskclf.Classifier(dirs=_files(tmp_path), start=broken, wait=0.05)
    t0 = time.time()
    with pytest.raises(taskclf.Unavailable, match="exited"):
        clf.embeddings_url(wait=30, poll=0.05)
    assert time.time() - t0 < 5                          # not the full wait


def test_flag_on_serve_and_the_default_command():
    ap = cli.build_parser()
    assert ap.parse_args(["serve", "--embeddings"]).embeddings and ap.parse_args(["--embeddings"]).embeddings
    assert not ap.parse_args(["serve"]).embeddings
