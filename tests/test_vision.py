"""Image input through all four APIs (#44), against a fake llama-server that may or may not have a projector."""
import io
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

from localllm import bench, cli, gateway, router

PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
JPEG = "/9j/4AAQSkZJRgABAQ"


class FakeVisionLlama(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    vision = True
    seen = []

    def log_message(self, *a):
        pass

    def _send(self, obj, ctype="application/json"):
        out = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)

    def do_GET(self):
        if self.path == "/props":
            return self._send({"modalities": {"vision": FakeVisionLlama.vision, "audio": False}})
        self._send({})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeVisionLlama.seen.append((self.path, body))
        if body.get("stream"):
            data = b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in (
                {"choices": [{"delta": {"content": "green"}}]}, {"choices": [{"delta": {}, "finish_reason": "stop"}]}))
            return self._send(data + b"data: [DONE]\n\n", "text/event-stream")
        self._send({"choices": [{"message": {"role": "assistant", "content": "green"}, "finish_reason": "stop"}]})


def _servers(monkeypatch, vision: bool):
    monkeypatch.setattr(router, "load_config", lambda: {"enabled": False})
    monkeypatch.setattr(FakeVisionLlama, "vision", vision)
    up = HTTPServer(("127.0.0.1", 0), FakeVisionLlama)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    g = gateway.serve(f"http://127.0.0.1:{up.server_port}", port=0, model_name="m")
    return up, g, f"http://127.0.0.1:{g.server_address[1]}"


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


REQUESTS = {   # the same "what colour?" question with one image, in each API's own format
    "openai": ("/v1/chat/completions", {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "colour?"}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}}]}]}),
    "anthropic": ("/v1/messages", {"model": "m", "max_tokens": 9, "messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}},
        {"type": "text", "text": "colour?"}]}]}),
    "ollama": ("/api/chat", {"model": "m", "stream": False,
                             "messages": [{"role": "user", "content": "colour?", "images": [PNG]}]}),
    "gemini": ("/v1beta/models/m:generateContent", {"contents": [{"role": "user", "parts": [
        {"text": "colour?"}, {"inlineData": {"mimeType": "image/png", "data": PNG}}]}]}),
}


def test_images_reach_llama_server_in_openai_form(monkeypatch):
    up, g, url = _servers(monkeypatch, vision=True)
    want = [{"type": "text", "text": "colour?"}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}}]
    try:
        for api, (path, body) in REQUESTS.items():
            code, out = _post(url + path, body)
            assert code == 200, (api, out)
            sent_path, sent = FakeVisionLlama.seen[-1]
            if api in ("openai", "anthropic"):
                assert (sent_path, sent) == (path, body)                    # passed through untouched
            else:
                assert sent_path == "/v1/chat/completions" and sent["messages"][0]["content"] == want, api
        code, out = _post(url + "/v1beta/models/m:streamGenerateContent", REQUESTS["gemini"][1])
        assert code == 200 and "green" in out
        code, out = _post(url + "/api/chat", {**REQUESTS["ollama"][1], "stream": True})
        assert code == 200 and "green" in out
    finally:
        g.shutdown(); up.shutdown()


def test_without_projector_each_api_gets_its_own_error_with_the_fix(monkeypatch):
    up, g, url = _servers(monkeypatch, vision=False)
    try:
        before = len(FakeVisionLlama.seen)
        for api, (path, body) in REQUESTS.items():
            code, out = _post(url + path, body)
            err = json.loads(out)
            msg = {"openai": lambda e: e["error"]["message"], "anthropic": lambda e: e["error"]["message"],
                   "ollama": lambda e: e["error"], "gemini": lambda e: e["error"]["message"]}[api](err)
            assert code == 400 and "--vision" in msg, api
        assert err["error"]["status"] == "INVALID_ARGUMENT"
        assert len(FakeVisionLlama.seen) == before                          # nothing reached the model
        code, _ = _post(url + "/api/chat", {"model": "m", "stream": False,
                                            "messages": [{"role": "user", "content": "hi"}]})
        assert code == 200                                                  # text still works
        monkeypatch.setattr(FakeVisionLlama, "vision", True)                # restarted with --vision, same port
        assert _post(url + "/api/chat", REQUESTS["ollama"][1])[0] == 200
    finally:
        g.shutdown(); up.shutdown()


def test_ollama_and_gemini_image_translation():
    out = gateway.ollama_to_openai({"prompt": "what is it?", "images": [JPEG], "stream": False}, chat=False)
    assert out["messages"][-1]["content"] == [{"type": "text", "text": "what is it?"},
                                              {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{JPEG}"}}]
    out = gateway.ollama_to_openai({"messages": [{"role": "user", "content": "", "images": ["UklGRabc"]}]}, chat=True)
    assert out["messages"][0]["content"] == [{"type": "image_url", "image_url": {"url": "data:image/webp;base64,UklGRabc"}}]
    out = gateway.gemini_to_openai({"contents": [{"parts": [
        {"inline_data": {"mime_type": "image/jpeg", "data": JPEG}}, {"text": "and this?"}]}]})
    assert [p["type"] for p in out["messages"][0]["content"]] == ["image_url", "text"]     # order kept
    assert gateway.has_image(out) and not gateway.has_image({"messages": [{"role": "user", "content": "hi"}]})
    try:
        gateway.gemini_to_openai({"contents": [{"parts": [{"fileData": {"fileUri": "gs://x", "mimeType": "image/png"}}]}]})
        raise AssertionError("fileData should be refused")
    except ValueError as e:
        assert "inlineData" in str(e)


def test_vqa_match():
    assert bench.vqa_match("Red.", ["red"]) and bench.vqa_match("It is red", ["red"])
    assert bench.vqa_match("แมว", ["แมว"]) and bench.vqa_match("สีแดง", ["แดง", "สีแดง"])
    assert not bench.vqa_match("blue", ["red"]) and not bench.vqa_match("", ["red"])
    assert not bench.vqa_match("red " + "x" * 40, ["red"])                  # a long reply doesn't count
    assert "vision" in bench.available("th", bench.SUITES) and "vision" not in bench.available("de", bench.SUITES)


def test_projector_file_is_named_per_repo(monkeypatch, tmp_path):
    key = "gemma4-26b-a4b-qat"
    name = "gemma-4-26B-A4B-it-qat-GGUF-mmproj-F16.gguf"
    (tmp_path / name).write_bytes(b"x")
    monkeypatch.setenv("LOCALLLM_MODELS", str(tmp_path))
    assert cli._mmproj_path(key) == tmp_path / name


def test_rows_stop_at_the_limit_and_retry_server_errors(monkeypatch):
    calls = []

    def fake_urlopen(url, timeout=0):
        calls.append(url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(url, 502, "Bad Gateway", {}, None)
        n = int(url.split("length=")[1])
        return io.BytesIO(json.dumps({"rows": [{"row": {"i": k}} for k in range(n)],
                                      "num_rows_total": 300}).encode())
    monkeypatch.setattr(bench.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(bench.time, "sleep", lambda s: None)
    rows = bench._rows("floschne/maxm", "default", "th", page=10, limit=25)
    assert len(rows) == 25 and len(calls) == 4                              # 1 retry + pages of 10, 10, 5
    assert calls[-1].endswith("length=5")
