import json
import socket
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from localllm import gateway, router


class FakeLlama(BaseHTTPRequestHandler):
    """Stands in for llama-server (and for a cloud provider): echoes the last user message."""
    protocol_version = "HTTP/1.1"
    seen = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        b = b'{"status":"ok"}'
        self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeLlama.seen.append((self.path, dict(self.headers), body))
        last = body.get("messages", [{}])[-1].get("content", "")
        if body.get("stream"):
            self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
            for c in [{"choices": [{"delta": {"content": "echo: "}}]}, {"choices": [{"delta": {"content": last}}]},
                      {"choices": [{"delta": {}, "finish_reason": "stop"}], "timings": {"predicted_n": 2, "predicted_ms": 10}}]:
                d = f"data: {json.dumps(c, ensure_ascii=False)}\n\n".encode()
                self.wfile.write(f"{len(d):x}\r\n".encode() + d + b"\r\n")
            d = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(d):x}\r\n".encode() + d + b"\r\n0\r\n\r\n")
            return
        out = json.dumps({"choices": [{"message": {"role": "assistant", "content": "echo: " + last},
                                       "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)


def _up():
    s = HTTPServer(("127.0.0.1", 0), FakeLlama)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, f"http://127.0.0.1:{s.server_port}"


def _gw(upstream):
    g = gateway.serve(upstream, port=0, model_name="test-model")
    return g, f"http://127.0.0.1:{g.server_address[1]}"


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=30)


def test_openai_passthrough_reports_local_route(monkeypatch):
    monkeypatch.setattr(router, "load_config", lambda: {"enabled": False})
    up, uurl = _up(); g, gurl = _gw(uurl)
    r = _post(gurl + "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
    assert json.load(r)["choices"][0]["message"]["content"] == "echo: hi" and r.headers["X-Localllm-Route"] == "local"
    g.shutdown(); up.shutdown()


def test_ollama_chat_non_stream_and_stream(monkeypatch):
    monkeypatch.setattr(router, "load_config", lambda: {"enabled": False})
    up, uurl = _up(); g, gurl = _gw(uurl)
    d = json.load(_post(gurl + "/api/chat", {"model": "x", "stream": False,
                                             "messages": [{"role": "user", "content": "สวัสดี"}]}))
    assert d["message"]["content"] == "echo: สวัสดี" and d["done"] is True and d["eval_count"] == 2
    lines = [json.loads(l) for l in _post(gurl + "/api/generate", {"prompt": "yo"}).read().decode().splitlines() if l]
    assert "".join(l.get("response", "") for l in lines) == "echo: yo" and lines[-1]["done"] is True
    tags = json.load(urllib.request.urlopen(gurl + "/api/tags"))
    assert tags["models"][0]["name"] == "test-model"
    g.shutdown(); up.shutdown()


def test_gemini_generate_and_stream(monkeypatch):
    monkeypatch.setattr(router, "load_config", lambda: {"enabled": False})
    up, uurl = _up(); g, gurl = _gw(uurl)
    body = {"contents": [{"role": "user", "parts": [{"text": "hello"}]}], "generationConfig": {"maxOutputTokens": 9}}
    d = json.load(_post(gurl + "/v1beta/models/m:generateContent", body))
    assert d["candidates"][0]["content"]["parts"][0]["text"] == "echo: hello"
    assert d["usageMetadata"]["totalTokenCount"] == 5 and FakeLlama.seen[-1][2]["max_tokens"] == 9
    raw = _post(gurl + "/v1beta/models/m:streamGenerateContent?alt=sse", body).read().decode()
    texts = [json.loads(l[5:])["candidates"][0]["content"]["parts"][0]["text"] for l in raw.splitlines() if l.startswith("data:")]
    assert "".join(texts) == "echo: hello"
    g.shutdown(); up.shutdown()


def test_router_local_by_default_and_rules(monkeypatch):
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert router.decide(body, "/v1/chat/completions", {"enabled": False}).provider is None
    monkeypatch.setenv("FAKE_KEY", "k")
    cfg = {"enabled": True, "provider": "c",
           "providers": {"c": {"kind": "openai", "url": "http://x", "key_env": "FAKE_KEY", "model": "m"}},
           "rules": {"max_local_prompt_tokens": 50, "language_floor": 70}, "local_model": "qwen3.8-27b-q3"}
    assert router.decide(body, "/v1/chat/completions", cfg).provider is None                       # stays local
    assert router.decide({"model": "gpt-5", **body}, "/v1/chat/completions", cfg).provider.name == "c"
    assert router.decide({"messages": [{"role": "user", "content": "x" * 400}]}, "/v1/chat/completions", cfg).provider
    thai = {"messages": [{"role": "user", "content": "ช่วยอธิบายเรื่องนี้หน่อย"}]}              # th score 67.1 < 70
    assert "th score" in router.decide(thai, "/v1/chat/completions", cfg).label
    assert router.decide(thai, "/v1/messages", cfg).provider is None                               # no anthropic key


def test_router_forwards_to_cloud_with_key_and_model(monkeypatch):
    up, uurl = _up(); cloud, curl = _up(); g, gurl = _gw(uurl)
    monkeypatch.setenv("FAKE_KEY", "secret")
    cfg = {"enabled": True, "providers": {"c": {"kind": "openai", "url": curl, "key_env": "FAKE_KEY", "model": "big"}}}
    monkeypatch.setattr(router, "load_config", lambda: cfg)
    r = _post(gurl + "/v1/chat/completions", {"model": "gpt-5", "messages": [{"role": "user", "content": "q"}]})
    path, headers, body = FakeLlama.seen[-1]
    assert r.headers["X-Localllm-Route"].startswith("cloud:c") and headers["Authorization"] == "Bearer secret"
    assert body["model"] == "big"
    g.shutdown(); up.shutdown(); cloud.shutdown()


def test_detect_language():
    assert router.detect_language("สวัสดีครับ") == "th" and router.detect_language("こんにちは") == "ja"
    assert router.detect_language("hello world") == "en"


def test_route_compare_local_and_cloud(monkeypatch):
    up, uurl = _up(); cloud, curl = _up()
    monkeypatch.setenv("FAKE_KEY", "k")
    cfg = {"enabled": True, "providers": {"c": {"kind": "openai", "url": curl, "key_env": "FAKE_KEY", "model": "big",
                                                "price_in": 3.0, "price_out": 15.0}}}
    rows = router.compare("hi", uurl, cfg)
    up.shutdown(); cloud.shutdown()
    assert [r["target"] for r in rows] == ["local", "c"]
    assert rows[1]["cost_usd"] == (3 * 3.0 + 2 * 15.0) / 1e6 and rows[0]["answer"] == "echo: hi"


def test_passthrough_forwards_request_headers(monkeypatch):
    # llama-server's web UI answers 415 unless the browser's Accept-Encoding reaches it
    monkeypatch.setattr(router, "load_config", lambda: {"enabled": False})
    up, uurl = _up(); g, gurl = _gw(uurl)
    req = urllib.request.Request(gurl + "/v1/chat/completions", data=json.dumps({"messages": [{"role": "user", "content": "x"}]}).encode(),
                                 headers={"Content-Type": "application/json", "Accept-Encoding": "gzip", "X-Custom": "1"})
    urllib.request.urlopen(req, timeout=30).read()
    hdrs = {k.lower(): v for k, v in FakeLlama.seen[-1][1].items()}
    g.shutdown(); up.shutdown()
    assert hdrs.get("accept-encoding") == "gzip" and hdrs.get("x-custom") == "1"


# ---- LAN mode: API key -----------------------------------------------------------------------------------------------

def _keyed_gw(upstream, key="lan-secret"):
    # trust_loopback=False makes this test client count as "another device on the network"
    g = gateway.serve(upstream, port=0, model_name="test-model", api_key=key, trust_loopback=False)
    return g, f"http://127.0.0.1:{g.server_address[1]}"


def _status(url, body=None, headers=None):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def test_lan_key_accepted_and_denied_on_all_four_apis(monkeypatch):
    monkeypatch.setattr(router, "load_config", lambda: {"enabled": False})
    up, uurl = _up(); g, gurl = _keyed_gw(uurl)
    chat = {"messages": [{"role": "user", "content": "hi"}], "stream": False}
    gem = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
    calls = [  # (path, body, the way that API's SDK sends a key)
        ("/v1/chat/completions", chat, {"Authorization": "Bearer lan-secret"}),        # OpenAI
        ("/v1/messages", {"max_tokens": 5, **chat}, {"x-api-key": "lan-secret"}),        # Anthropic
        ("/api/chat", chat, {"Authorization": "Bearer lan-secret"}),                    # Ollama
        ("/v1beta/models/m:generateContent", gem, {"x-goog-api-key": "lan-secret"}),    # Gemini (header)
        ("/v1beta/models/m:generateContent?key=lan-secret", gem, {}),                   # Gemini (query)
        ("/api/tags", None, {"Authorization": "Bearer lan-secret"}),
    ]
    try:
        for path, body, good in calls:
            bare = path.split("?")[0]
            assert _status(gurl + path, body, good) == 200, path
            assert _status(gurl + bare, body) == 401, bare                                      # no key
            assert _status(gurl + bare, body, {"Authorization": "Bearer nope", "x-api-key": "nope"}) == 401, bare
            assert _status(gurl + bare + "?key=nope", body) == 401, bare
    finally:
        g.shutdown(); up.shutdown()


def test_lan_key_never_reaches_llama_server_or_cloud(monkeypatch):
    up, uurl = _up(); cloud, curl = _up(); g, gurl = _keyed_gw(uurl)
    monkeypatch.setattr(router, "load_config", lambda: {"enabled": False})
    _status(gurl + "/v1/chat/completions?key=lan-secret", {"messages": [{"role": "user", "content": "x"}]},
            {"Authorization": "Bearer lan-secret", "x-api-key": "lan-secret", "x-goog-api-key": "lan-secret"})
    path, headers, _ = FakeLlama.seen[-1]
    assert "lan-secret" not in path and "lan-secret" not in json.dumps(headers)                 # local llama-server
    monkeypatch.setenv("FAKE_KEY", "cloud-secret")
    cfg = {"enabled": True, "providers": {"c": {"kind": "openai", "url": curl, "key_env": "FAKE_KEY", "model": "big"}}}
    monkeypatch.setattr(router, "load_config", lambda: cfg)
    _status(gurl + "/v1/chat/completions", {"model": "gpt-5", "messages": [{"role": "user", "content": "q"}]},
            {"Authorization": "Bearer lan-secret"})
    path, headers, _ = FakeLlama.seen[-1]
    g.shutdown(); up.shutdown(); cloud.shutdown()
    assert headers["Authorization"] == "Bearer cloud-secret" and "lan-secret" not in json.dumps(headers)


def test_lan_refuses_network_address_without_key_and_trusts_this_pc(monkeypatch):
    up, uurl = _up()
    for host in ("0.0.0.0", "::", "192.168.1.20"):
        try:
            gateway.serve(uurl, host=host, port=0)
            raise AssertionError(f"bound {host} without a key")
        except ValueError as e:
            assert "API key" in str(e)
    assert gateway.is_loopback("127.0.0.1") and gateway.is_loopback("::1") and gateway.is_loopback("localhost")
    monkeypatch.setattr(router, "load_config", lambda: {"enabled": False})
    g = gateway.serve(uurl, port=0, model_name="m", api_key="k")          # default: requests from this PC need no key
    code = _status(f"http://127.0.0.1:{g.server_address[1]}/api/tags")
    g.shutdown()
    g, gurl = _keyed_gw(uurl)                       # another device may load the chat page, not call the API
    page, api = _status(gurl + "/"), _status(gurl + "/v1/models")
    g.shutdown(); up.shutdown()
    assert code == 200 and page == 200 and api == 401


def test_split_key_keeps_other_query_parameters():
    assert gateway.split_key("/v1beta/models/m:streamGenerateContent?alt=sse&key=K") == \
        ("/v1beta/models/m:streamGenerateContent?alt=sse", "K")
    assert gateway.split_key("/v1/models") == ("/v1/models", None)


def _ipv6_ok() -> bool:
    try:
        with socket.socket(socket.AF_INET6) as s:
            s.bind(("::1", 0))
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _ipv6_ok(), reason="this machine has no IPv6 (GitHub runners do)")
def test_lan_dual_stack_host_takes_ipv6_and_ipv4_clients(monkeypatch):
    up, uurl = _up()
    monkeypatch.setattr(router, "load_config", lambda: {"enabled": False})
    g = gateway.serve(uurl, host="::", port=0, model_name="m", api_key="k")
    port = g.server_address[1]
    try:
        for host in ("[::1]", "127.0.0.1"):              # 127.0.0.1 arrives as ::ffff:127.0.0.1: still this PC
            assert _status(f"http://{host}:{port}/v1/models") == 200, host
    finally:
        g.shutdown()
    g = gateway.serve(uurl, host="::", port=0, model_name="m", api_key="k", trust_loopback=False)
    port = g.server_address[1]
    try:
        for host in ("[::1]", "127.0.0.1"):
            assert _status(f"http://{host}:{port}/v1/models") == 401, host
            assert _status(f"http://{host}:{port}/v1/models", headers={"Authorization": "Bearer k"}) == 200, host
    finally:
        g.shutdown(); up.shutdown()


def test_mapped_loopback_and_key_required_locally():
    assert gateway.is_loopback("::ffff:127.0.0.1") and not gateway.is_loopback("::ffff:192.168.1.5")
    try:
        gateway.serve("http://127.0.0.1:9", port=0, trust_loopback=False)
        raise AssertionError("served local clients that must send a key, with no key set")
    except ValueError as e:
        assert "key" in str(e)
