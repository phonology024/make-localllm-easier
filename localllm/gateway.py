"""One local endpoint that speaks the APIs apps already use, in front of llama-server.

  OpenAI     /v1/chat/completions /v1/completions /v1/models /v1/embeddings   -> passed through
  Anthropic  /v1/messages /v1/messages/count_tokens                          -> passed through (llama-server native)
  Ollama     /api/chat /api/generate /api/tags /api/version                  -> translated to/from OpenAI chat
  Gemini     /v1beta/models/{m}:generateContent and :streamGenerateContent   -> translated to/from OpenAI chat

Chat requests go through `router.decide()` first: local by default; forwarded to the user's own cloud key only when the
user enabled it in ~/.localllm/route.json and a rule says so. The decision is reported in the X-Localllm-Route header.
Standard library only.
"""
from __future__ import annotations

import json
import re
import threading
import time
from contextlib import nullcontext
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import router

PASSTHROUGH = ("/v1/chat/completions", "/v1/completions", "/v1/models", "/v1/embeddings", "/v1/messages",
               "/v1/messages/count_tokens", "/health", "/props", "/slots")
HOP_BY_HOP = {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
              "proxy-authorization", "proxy-connection"}
GEMINI = re.compile(r"^/v1beta/models/([^/:]+):(generateContent|streamGenerateContent)")


def _post(url: str, body: dict, headers: dict | None = None, timeout: int = 3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json",
                                                                              **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)


def _sse_chunks(resp):
    """Yield parsed OpenAI stream chunks from an SSE response."""
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if line.startswith("data:") and line != "data: [DONE]":
            yield json.loads(line[5:])


# ---- format translation (pure functions, unit-tested) ----------------------------------------------------------------

def ollama_to_openai(body: dict, chat: bool) -> dict:
    msgs = body.get("messages") or []
    if not chat:
        msgs = ([{"role": "system", "content": body["system"]}] if body.get("system") else []) + \
               [{"role": "user", "content": body.get("prompt", "")}]
    opts = body.get("options") or {}
    out = {"messages": msgs, "stream": bool(body.get("stream", True))}
    for src, dst in (("temperature", "temperature"), ("top_p", "top_p"), ("num_predict", "max_tokens"), ("seed", "seed")):
        if src in opts:
            out[dst] = opts[src]
    return out


def openai_to_ollama(choice_text: str, model: str, chat: bool, done: bool, timings: dict | None = None) -> dict:
    d = {"model": model, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "done": done}
    if chat:
        d["message"] = {"role": "assistant", "content": choice_text}
    else:
        d["response"] = choice_text
    if done and timings:
        d["eval_count"] = timings.get("predicted_n", 0)
        d["eval_duration"] = int(timings.get("predicted_ms", 0) * 1e6)
        d["prompt_eval_count"] = timings.get("prompt_n", 0)
    return d


def gemini_to_openai(body: dict) -> dict:
    msgs = []
    sys_inst = body.get("systemInstruction") or body.get("system_instruction")
    if sys_inst:
        msgs.append({"role": "system", "content": "".join(p.get("text", "") for p in sys_inst.get("parts", []))})
    for c in body.get("contents", []):
        role = "assistant" if c.get("role") == "model" else "user"
        msgs.append({"role": role, "content": "".join(p.get("text", "") for p in c.get("parts", []))})
    cfg = body.get("generationConfig") or {}
    out = {"messages": msgs}
    for src, dst in (("temperature", "temperature"), ("topP", "top_p"), ("maxOutputTokens", "max_tokens")):
        if src in cfg:
            out[dst] = cfg[src]
    return out


def openai_to_gemini(text: str, finish: str | None = None, usage: dict | None = None) -> dict:
    cand = {"content": {"role": "model", "parts": [{"text": text}]}, "index": 0}
    if finish:
        cand["finishReason"] = {"stop": "STOP", "length": "MAX_TOKENS"}.get(finish, "STOP")
    d = {"candidates": [cand]}
    if usage:
        d["usageMetadata"] = {"promptTokenCount": usage.get("prompt_tokens", 0),
                              "candidatesTokenCount": usage.get("completion_tokens", 0),
                              "totalTokenCount": usage.get("total_tokens", 0)}
    return d


IDLE_NOTE = "the model was unloaded while idle to free the GPU; it loads again on your next message"
IDLE_PAGE = ("<!doctype html><meta charset=utf-8><title>localllm</title><p style='font:16px sans-serif;margin:2em'>"
             f"localllm: {IDLE_NOTE}. Send a message from your app or <code>localllm chat</code>, then reload.</p>")


# ---- HTTP server ------------------------------------------------------------------------------------------------------

def make_handler(upstream, model_name: str):
    """`upstream` is a llama-server URL, or a pool.Pool that picks (and lazy-loads) a model per request."""
    def local(body: dict, stay: bool = False):
        return nullcontext((upstream, None)) if isinstance(upstream, str) else upstream.use(body, stay=stay)

    def base_url(method: str) -> str | None:
        """Where a non-chat request goes. A pool wakes for POSTs (work) but not for GETs (health checks, model lists,
        polling pages): those get None while it is unloaded and must not reset its idle clock."""
        if isinstance(upstream, str):
            return upstream
        return upstream.ensure_url() if method == "POST" else upstream.peek_url()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _stay(self) -> bool:
            """X-Localllm-Stay: 1 keeps the loaded model for this request (no swap)."""
            return (self.headers.get("X-Localllm-Stay") or "").strip().lower() in ("1", "true", "yes")

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def _json(self, code: int, obj: dict, extra: dict | None = None):
            data = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def _stream_start(self, ctype: str, extra: dict | None = None):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Transfer-Encoding", "chunked")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()

        def _chunk(self, data: bytes):
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        def _stream_end(self):
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        def _proxy(self, method: str):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) if method == "POST" else None
            route_hdr, hdrs, cloud = "local", {}, None
            if method == "POST" and self.path in ("/v1/chat/completions", "/v1/messages"):
                d = router.decide(json.loads(body or b"{}"), self.path)
                route_hdr, cloud = d.label, d.provider
                if cloud:
                    hdrs, body = cloud.headers(self.path), cloud.adapt(body, self.path)
                    return self._send(method, cloud.url_base, body, hdrs, route_hdr, None)
                with local(json.loads(body or b"{}"), self._stay()) as (url, model_hdr):
                    return self._send(method, url, body, hdrs, route_hdr, model_hdr)
            url = base_url(method)
            if url is None:
                return self._idle()
            try:
                self._send(method, url, body, hdrs, route_hdr, None)
            except OSError:                             # URLError, or the connection dropped as the server stopped
                if method != "GET" or isinstance(upstream, str) or upstream.peek_url() == url:
                    raise
                self._idle()                            # unloaded between the look and the request

        def _idle(self):
            """A GET while the pool has unloaded its model(s): answer here instead of loading one."""
            if self.path == "/health":
                return self._json(200, {"status": "ok", "model": "unloaded"})
            if self.path == "/":
                html = IDLE_PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                return self.wfile.write(html)
            self._json(503, {"error": IDLE_NOTE}, {"Retry-After": "5"})

        def _send(self, method: str, target: str, body, hdrs: dict, route_hdr: str, model_hdr: str | None):
            fwd = {k: v for k, v in self.headers.items()
                   if k.lower() not in HOP_BY_HOP and not (hdrs and k.lower() in ("authorization", "x-api-key"))}
            req = urllib.request.Request(target.rstrip("/") + self.path, data=body, method=method, headers={**fwd, **hdrs})
            try:
                r = urllib.request.urlopen(req, timeout=3600)
            except urllib.error.HTTPError as e:
                r = e
            self.send_response(r.status if hasattr(r, "status") else r.code)
            for k, v in r.headers.items():
                if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                    self.send_header(k, v)
            self.send_header("X-Localllm-Route", route_hdr)
            if model_hdr:
                self.send_header("X-Localllm-Model", model_hdr)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while chunk := r.read(8192) if not hasattr(r, "read1") else r.read1(8192):
                self._chunk(chunk)
            self._stream_end()

        def do_GET(self):
            if self.path == "/v1/models" and not isinstance(upstream, str):   # the pool's names, without waking it
                return self._json(200, {"object": "list", "data": [
                    {"id": k, "object": "model", "owned_by": "localllm"} for k in upstream.keys]})
            if self.path == "/api/tags":
                return self._json(200, {"models": [{"name": model_name, "model": model_name, "size": 0,
                                                    "details": {"format": "gguf"}}]})
            if self.path == "/api/version":
                return self._json(200, {"version": "0.14.0-localllm"})
            if self.path.startswith(PASSTHROUGH) or self.path == "/":
                return self._proxy("GET")
            self._json(404, {"error": f"not supported: {self.path}"})

        def do_POST(self):
            if self.path in ("/api/chat", "/api/generate"):
                return self._ollama(self.path == "/api/chat")
            m = GEMINI.match(self.path)
            if m:
                return self._gemini(m.group(2) == "streamGenerateContent")
            if self.path.startswith(PASSTHROUGH):
                return self._proxy("POST")
            self._json(404, {"error": f"not supported: {self.path}"})

        def _ollama(self, chat: bool):
            req = ollama_to_openai(self._body(), chat)
            stream = req["stream"]
            req["stream"] = True
            with local(req, self._stay()) as (url, model_hdr):
                self._ollama_reply(_post(url + "/v1/chat/completions", req), chat, stream,
                                   {"X-Localllm-Model": model_hdr} if model_hdr else None)

        def _ollama_reply(self, r, chat: bool, stream: bool, extra: dict | None):
            if stream:
                self._stream_start("application/x-ndjson", extra)
                timings = {}
                for c in _sse_chunks(r):
                    timings = c.get("timings", timings)
                    text = (c.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
                    if text:
                        self._chunk((json.dumps(openai_to_ollama(text, model_name, chat, False), ensure_ascii=False)
                                     + "\n").encode())
                self._chunk((json.dumps(openai_to_ollama("", model_name, chat, True, timings)) + "\n").encode())
                return self._stream_end()
            text, timings = [], {}
            for c in _sse_chunks(r):
                timings = c.get("timings", timings)
                text.append((c.get("choices") or [{}])[0].get("delta", {}).get("content") or "")
            self._json(200, openai_to_ollama("".join(text), model_name, chat, True, timings), extra)

        def _gemini(self, stream: bool):
            req = gemini_to_openai(self._body())
            with local(req, self._stay()) as (url, model_hdr):
                self._gemini_reply(url, req, stream, {"X-Localllm-Model": model_hdr} if model_hdr else None)

        def _gemini_reply(self, url: str, req: dict, stream: bool, extra: dict | None):
            if not stream:
                r = json.load(_post(url + "/v1/chat/completions", {**req, "stream": False}))
                ch = r["choices"][0]
                return self._json(200, openai_to_gemini(ch["message"].get("content") or "", ch.get("finish_reason"),
                                                        r.get("usage")), extra)
            r = _post(url + "/v1/chat/completions", {**req, "stream": True})
            self._stream_start("text/event-stream", extra)
            for c in _sse_chunks(r):
                ch = (c.get("choices") or [{}])[0]
                text = ch.get("delta", {}).get("content") or ""
                if text or ch.get("finish_reason"):
                    self._chunk(f"data: {json.dumps(openai_to_gemini(text, ch.get('finish_reason')), ensure_ascii=False)}\r\n\r\n"
                                .encode())
            self._stream_end()

    return Handler


def serve(upstream, host: str = "127.0.0.1", port: int = 8080, model_name: str = "local") -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), make_handler(upstream, model_name))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
