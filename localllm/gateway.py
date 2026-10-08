"""One local endpoint that speaks the APIs apps already use, in front of llama-server.

  OpenAI     /v1/chat/completions /v1/completions /v1/models /v1/embeddings   -> passed through
  Anthropic  /v1/messages /v1/messages/count_tokens                          -> passed through (llama-server native)
  Ollama     /api/chat /api/generate /api/tags /api/version                  -> translated to/from OpenAI chat
  Gemini     /v1beta/models/{m}:generateContent and :streamGenerateContent   -> translated to/from OpenAI chat

Chat requests go through `router.decide()` first: local by default; forwarded to the user's own cloud key only when the
user enabled it in ~/.localllm/route.json and a rule says so. The decision is reported in the X-Localllm-Route header.

LAN mode: listening on anything but loopback needs an API key. Clients send it the way their SDK does -
`Authorization: Bearer KEY` (OpenAI, Ollama), `x-api-key` (Anthropic), `x-goog-api-key` or `?key=` (Gemini). It is
checked in constant time and stripped before forwarding, so it never reaches llama-server or a cloud provider. Requests
from this PC itself need no key: any local program can already reach llama-server's private port.
Standard library only.
"""
from __future__ import annotations

import hmac
import ipaddress
import json
import re
import socket
import threading
import time
from contextlib import nullcontext
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qsl, urlencode, urlsplit

from . import router

PASSTHROUGH = ("/v1/chat/completions", "/v1/completions", "/v1/models", "/v1/embeddings", "/v1/messages",
               "/v1/messages/count_tokens", "/health", "/props", "/slots")
HOP_BY_HOP = {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
              "proxy-authorization", "proxy-connection"}
GEMINI = re.compile(r"^/v1beta/models/([^/:]+):(generateContent|streamGenerateContent)")
KEY_HEADERS = ("authorization", "x-api-key", "x-goog-api-key")   # where clients put the gateway's key


def is_loopback(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host == "localhost"
    mapped = getattr(ip, "ipv4_mapped", None)        # an IPv4 client of a dual-stack socket: ::ffff:127.0.0.1
    return (mapped or ip).is_loopback


def split_key(path: str) -> tuple[str, str | None]:
    """Path without its ?key= parameter (Gemini's way of sending a key), and that key."""
    url = urlsplit(path)
    if not url.query:
        return path, None
    params = parse_qsl(url.query, keep_blank_values=True)
    key = next((v for k, v in params if k == "key"), None)
    rest = urlencode([(k, v) for k, v in params if k != "key"])
    return url.path + ("?" + rest if rest else ""), key


def presented_keys(headers, query_key: str | None) -> list[str]:
    auth = headers.get("Authorization") or ""
    keys = [auth[7:].strip() if auth.lower().startswith("bearer ") else "",
            headers.get("x-api-key") or "", headers.get("x-goog-api-key") or "", query_key or ""]
    return [k for k in keys if k]


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


# ---- HTTP server ------------------------------------------------------------------------------------------------------

def make_handler(upstream, model_name: str, api_key: str | None = None, trust_loopback: bool = True):
    """`upstream` is a llama-server URL, or a pool.Pool that picks (and lazy-loads) a model per request.
    With `api_key`, every request from another machine must carry it (loopback clients too unless trust_loopback)."""
    def local(body: dict):
        return nullcontext((upstream, None)) if isinstance(upstream, str) else upstream.use(body)

    def base_url() -> str:
        return upstream if isinstance(upstream, str) else upstream.url

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _authorized(self) -> bool:
            """Strip ?key= from the path; check the key when one is required."""
            self.path, query_key = split_key(self.path)
            if not api_key or (trust_loopback and is_loopback(self.client_address[0])):
                return True
            want = api_key.encode()
            return any(hmac.compare_digest(k.encode(), want) for k in presented_keys(self.headers, query_key))

        def _deny(self):
            self.close_connection = True                        # the unread request body must not become the next request
            self._json(401, {"error": {"type": "authentication_error", "message": "missing or wrong API key for this "
                                       "localllm gateway (Authorization: Bearer, x-api-key or ?key=)"}},
                       {"WWW-Authenticate": "Bearer", "Connection": "close"})

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
                with local(json.loads(body or b"{}")) as (url, model_hdr):
                    return self._send(method, url, body, hdrs, route_hdr, model_hdr)
            self._send(method, base_url(), body, hdrs, route_hdr, None)

        def _send(self, method: str, target: str, body, hdrs: dict, route_hdr: str, model_hdr: str | None):
            fwd = {k: v for k, v in self.headers.items()          # our own key never leaves this gateway
                   if k.lower() not in HOP_BY_HOP and not ((hdrs or api_key) and k.lower() in KEY_HEADERS)}
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
            if not self._authorized() and self.path != "/":      # the chat page is static; its API calls need the key
                return self._deny()
            if self.path == "/api/tags":
                return self._json(200, {"models": [{"name": model_name, "model": model_name, "size": 0,
                                                    "details": {"format": "gguf"}}]})
            if self.path == "/api/version":
                return self._json(200, {"version": "0.14.0-localllm"})
            if self.path.startswith(PASSTHROUGH) or self.path == "/":
                return self._proxy("GET")
            self._json(404, {"error": f"not supported: {self.path}"})

        def do_POST(self):
            if not self._authorized():
                return self._deny()
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
            with local(req) as (url, model_hdr):
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
            with local(req) as (url, model_hdr):
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


class _Server(ThreadingHTTPServer):
    """IPv4 or IPv6 by the address; `::` also takes IPv4 clients (dual stack) where the OS allows it."""
    def __init__(self, addr, handler):
        if ":" in addr[0]:
            self.address_family = socket.AF_INET6
        super().__init__(addr, handler)

    def server_bind(self):
        if self.address_family == socket.AF_INET6 and self.server_address[0] == "::":
            try:
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            except (AttributeError, OSError):
                pass
        super().server_bind()


def serve(upstream, host: str = "127.0.0.1", port: int = 8080, model_name: str = "local", api_key: str | None = None,
          trust_loopback: bool = True) -> ThreadingHTTPServer:
    if not api_key and (not is_loopback(host) or not trust_loopback):
        raise ValueError(f"refusing to listen on {host} without an API key: anyone on the network could use this PC"
                         if not is_loopback(host) else "a key is required for local clients but none was given")
    srv = _Server((host, port), make_handler(upstream, model_name, api_key, trust_loopback))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
