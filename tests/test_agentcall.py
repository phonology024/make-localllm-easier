"""#97 cost saver under agents: cheap calls stay local, hard ones may go to the user's own key, a failed local answer
can be retried there. Fake servers only."""
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from localllm import agentcall, gateway, meter, router
from test_gateway import FakeLlama, _up

TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {}}},
         {"type": "function", "function": {"name": "edit_file", "parameters": {}}}]


def _body(*msgs, tools=None, **kw):
    return {"messages": [{"role": r, "content": c} for r, c in msgs], **({"tools": tools} if tools else {}), **kw}


@pytest.mark.parametrize("body,kind", [
    (_body(("user", "hi")), "chat"),
    (_body(("system", "You are Hermes."), ("user", "fix the failing test in parser.py"), tools=TOOLS), "hard"),
    (_body(("system", "agent"), ("user", "task"), ("assistant", "ok"), ("tool", "file contents"), tools=TOOLS), "cheap"),
    (_body(("user", "a"), ("assistant", "b"), ("user", "Summarize the conversation above in 5 bullets for memory")), "cheap"),
    (_body(("user", "a"), ("assistant", "b"), ("user", "Please save this to memory: the user likes tabs")), "cheap"),
    (_body(("user", "task"), ("assistant", "call"), ("user", "Error: invalid JSON in tool call arguments"), tools=TOOLS), "cheap"),
    (_body(("user", "a"), ("assistant", "b"), ("user", "what's the capital of Peru"), max_tokens=64), "cheap"),
    (_body(("user", "task"), ("assistant", "done"), ("user", "now also update the README"), tools=TOOLS), "hard"),
])
def test_classify(body, kind):
    assert agentcall.classify(body)[0] == kind


def test_failed_check():
    ok = json.dumps({"choices": [{"message": {"content": "", "tool_calls": [
        {"function": {"name": "f", "arguments": "{\"a\": 1}"}}]}, "finish_reason": "tool_calls"}]}).encode()
    bad = json.dumps({"choices": [{"message": {"content": "", "tool_calls": [
        {"function": {"name": "f", "arguments": "{\"a\": 1"}}]}}]}).encode()
    assert agentcall.failed_check("openai", 200, ok) is None
    assert agentcall.failed_check("openai", 200, bad) == "tool call arguments are not JSON"
    assert agentcall.failed_check("openai", 200, b'{"choices":[{"message":{"content":""},"finish_reason":"length"}]}') \
        == "cut off before answering"
    assert agentcall.failed_check("openai", 502, b"") == "local server error 502"
    assert agentcall.failed_check("openai", 400, b"{}") is None                # our request was wrong: don't pay for it
    assert agentcall.failed_check("anthropic", 200, b'{"content":[{"type":"text","text":" "}]}') == "empty answer"
    assert agentcall.failed_check("anthropic", 200, b'{"content":[{"type":"tool_use","name":"f","input":{}}]}') is None


def _cfg(cloud_url, **rules):
    return {"enabled": True, "provider": "c", "rules": rules,
            "providers": {"c": {"kind": "openai", "url": cloud_url, "key_env": "TEST_CLOUD_KEY", "model": "cloud-m"}}}


def test_decide_hard_to_cloud_only_when_asked(monkeypatch):
    monkeypatch.setenv("TEST_CLOUD_KEY", "sk-test")
    hard = _body(("user", "build a CLI for this repo"), tools=TOOLS)
    cheap = _body(("user", "a"), ("assistant", "b"), ("user", "Summarize the conversation so far"))
    d = router.decide(hard, "/v1/chat/completions", _cfg("http://x"))
    assert d.provider is None and d.label.startswith("local (agent call, hard")         # off by default
    d = router.decide(hard, "/v1/chat/completions", _cfg("http://x", agent_hard_to_cloud=True))
    assert d.provider.name == "c" and "agent call, hard: starts a task" in d.label
    d = router.decide(cheap, "/v1/chat/completions", _cfg("http://x", agent_hard_to_cloud=True, escalate_failed_local=True))
    assert d.provider is None and d.escalate.name == "c" and "cheap: housekeeping" in d.label
    assert router.decide(_body(("user", "hi")), "/v1/chat/completions", _cfg("http://x")).label == "local"


class BrokenToolCall(BaseHTTPRequestHandler):
    """A local model that answers with a tool call whose arguments are not JSON."""
    protocol_version = "HTTP/1.1"
    calls = 0

    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        BrokenToolCall.calls += 1
        out = json.dumps({"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "edit_file", "arguments": "{\"path\": "}}]},
            "finish_reason": "tool_calls"}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)


def _post(base, body):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.headers["X-Localllm-Route"], json.load(r)


def test_a_failed_local_answer_is_retried_on_the_cloud_key_with_the_cost(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_CLOUD_KEY", "sk-test")
    local = HTTPServer(("127.0.0.1", 0), BrokenToolCall)
    threading.Thread(target=local.serve_forever, daemon=True).start()
    cloud, curl = _up()
    m = meter.Meter(tmp_path / "savings.json")
    m.set_price(3, 15, "test prices")
    monkeypatch.setattr(router, "load_config", lambda: _cfg(curl, escalate_failed_local=True))
    g = gateway.serve(f"http://127.0.0.1:{local.server_port}", port=0, model_name="m", meter=m)
    base = f"http://127.0.0.1:{g.server_address[1]}"
    try:
        cont = _body(("user", "task"), ("assistant", "reading"), ("tool", "content of a.py"), tools=TOOLS, max_tokens=500)
        route, out = _post(base, cont)
        assert route.startswith("cloud:c (escalated: local answer failed the check - tool call arguments are not JSON")
        assert "est. up to $" in route and out["choices"][0]["message"]["content"].startswith("echo: ")
        sent = FakeLlama.seen[-1]
        assert sent[2]["model"] == "cloud-m" and sent[1].get("Authorization") == "Bearer sk-test"
        monkeypatch.setattr(router, "load_config", lambda: _cfg(curl))                     # escalation off
        route, out = _post(base, cont)
        assert route.startswith("local (agent call") and out["choices"][0]["message"]["tool_calls"]
    finally:
        g.shutdown(); local.shutdown(); cloud.shutdown()


def test_a_good_local_answer_stays_local_and_is_metered(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_CLOUD_KEY", "sk-test")
    up, uurl = _up()                                       # echoes: a usable answer
    m = meter.Meter(tmp_path / "savings.json")
    monkeypatch.setattr(router, "load_config", lambda: _cfg("http://127.0.0.1:9", escalate_failed_local=True))
    g = gateway.serve(uurl, port=0, model_name="m", meter=m)
    try:
        route, out = _post(f"http://127.0.0.1:{g.server_address[1]}",
                           _body(("user", "a"), ("assistant", "b"), ("user", "Summarize the conversation so far")))
        assert route == "local (agent call, cheap: housekeeping (summary, memory, title))"
        assert out["choices"][0]["message"]["content"].startswith("echo: ")
        assert m.month()["requests"] == 1 and m.month()["output"] > 0          # the savings meter still counts it
    finally:
        g.shutdown(); up.shutdown()
