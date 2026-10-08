"""Image questions through each of localllm's four APIs, and what the vision projector costs in memory (#44).

  python tools/vision_check.py --server PATH/llama-server --model M.gguf --mmproj P.gguf [--name NAME] [--json out.jsonl]

Starts llama-server twice with localllm's own launch flags: text-only (no projector), then with --mmproj. Text-only: an
image request must get each API's error with the `--vision` hint, and text must still work. With the projector: a plain
green picture must be answered "green" through OpenAI, Anthropic, Ollama and Gemini (streaming too for the two
translated APIs). Prints one row per check and the server's resident memory (VmRSS / peak VmHWM) in both modes.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from localllm import gateway, router, runtime  # noqa: E402

ASK = "What colour is this picture? Answer with one word."


def png(w: int, h: int, rgb: tuple[int, int, int]) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    rows = b"".join(b"\0" + bytes(rgb) * w for _ in range(h))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + \
        chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


IMG = base64.b64encode(png(256, 256, (40, 170, 60))).decode()


def requests(img: str) -> list[tuple[str, str, dict]]:
    return [
        ("openai", "/v1/chat/completions", {"max_tokens": 256, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}, {"type": "text", "text": ASK}]}]}),
        ("anthropic", "/v1/messages", {"model": "local", "max_tokens": 256, "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}},
            {"type": "text", "text": ASK}]}]}),
        ("ollama", "/api/chat", {"model": "local", "stream": False, "options": {"num_predict": 256},
                                 "messages": [{"role": "user", "content": ASK, "images": [img]}]}),
        ("ollama-stream", "/api/chat", {"model": "local", "stream": True, "options": {"num_predict": 256},
                                        "messages": [{"role": "user", "content": ASK, "images": [img]}]}),
        ("gemini", "/v1beta/models/local:generateContent", {"generationConfig": {"maxOutputTokens": 256}, "contents": [
            {"role": "user", "parts": [{"inlineData": {"mimeType": "image/png", "data": img}}, {"text": ASK}]}]}),
        ("gemini-stream", "/v1beta/models/local:streamGenerateContent", {"generationConfig": {"maxOutputTokens": 256},
            "contents": [{"role": "user", "parts": [{"inlineData": {"mimeType": "image/png", "data": img}},
                                                     {"text": ASK}]}]}),
    ]


def post(url: str, body: dict) -> tuple[int, str]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def answer(api: str, raw: str) -> str:
    """The reply text, whatever the API's shape (JSON, NDJSON or SSE)."""
    if api.startswith("ollama"):
        return "".join(json.loads(x)["message"]["content"] for x in raw.splitlines() if x.strip())
    if api.startswith("gemini"):
        chunks = [json.loads(x[5:]) for x in raw.splitlines() if x.startswith("data:")] or [json.loads(raw)]
        return "".join(p.get("text", "") for c in chunks for p in c["candidates"][0]["content"]["parts"])
    d = json.loads(raw)
    if api == "anthropic":
        return "".join(b.get("text", "") for b in d["content"])
    return d["choices"][0]["message"].get("content") or ""


def mem(pid: int) -> dict:
    out = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        k, _, v = line.partition(":")
        if k in ("VmRSS", "VmHWM"):
            out[k] = round(int(v.split()[0]) / 2**20, 2)          # kB -> GiB
    return out


def start(server: str, model: str, mmproj: str | None, port: int) -> subprocess.Popen:
    args = runtime.server_args(Path(model), None, port, 4096, False, ram_total_gb=16)
    args += ["-t", str(os.cpu_count() or 4), "--reasoning-budget", "0", "--jinja"] + (["--mmproj", mmproj] if mmproj else [])
    log = open(f"server-{'vision' if mmproj else 'text'}.log", "wb")
    p = subprocess.Popen([server, *args], stdout=log, stderr=subprocess.STDOUT)
    for _ in range(600):
        if p.poll() is not None:
            sys.exit(f"llama-server exited ({p.returncode}): see {log.name}")
        try:
            if b'"ok"' in urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read():
                return p
        except OSError:
            pass
        time.sleep(0.5)
    p.kill()
    sys.exit("llama-server did not start")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--mmproj", required=True); ap.add_argument("--name", default="model")
    ap.add_argument("--json")
    a = ap.parse_args()
    router.load_config = lambda: {"enabled": False}
    rows, failed = [], 0
    for mode, mmproj in (("text-only", None), ("vision", a.mmproj)):
        p = start(a.server, a.model, mmproj, 8091)
        g = gateway.serve("http://127.0.0.1:8091", port=0, model_name="local")
        url = f"http://127.0.0.1:{g.server_address[1]}"
        try:
            loaded = mem(p.pid)
            code, raw = post(url + "/v1/chat/completions", {"max_tokens": 8, "messages": [
                {"role": "user", "content": "Say hello."}]})
            checks = [("text", code == 200, f"HTTP {code}")]
            for api, path, body in requests(IMG):
                t = time.time()
                code, raw = post(url + path, body)
                if mode == "text-only":
                    ok, note = code == 400 and "--vision" in raw, f"HTTP {code}: {raw[:90]}"
                else:
                    text = answer(api, raw) if code == 200 else raw[:90]
                    ok, note = code == 200 and "green" in text.lower(), f"{time.time() - t:.1f}s {text.strip()[:60]!r}"
                checks.append((api, ok, note))
            used = mem(p.pid)
        finally:
            g.shutdown(); p.terminate(); p.wait(30)
        for api, ok, note in checks:
            failed += not ok
            print(f"{'OK  ' if ok else 'FAIL'} {a.name:14} {mode:9} {api:14} {note}", flush=True)
        row = {"model": a.name, "mode": mode, "loaded_rss_gb": loaded.get("VmRSS"), "rss_gb": used.get("VmRSS"),
               "peak_gb": used.get("VmHWM"), "ok": sum(ok for _, ok, _ in checks), "checks": len(checks)}
        print(f"MEM  {a.name:14} {mode:9} RSS after load {row['loaded_rss_gb']} GiB, after the checks "
              f"{row['rss_gb']} GiB, peak {row['peak_gb']} GiB", flush=True)
        rows.append(row)
    if a.json:
        with open(a.json, "a", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"projector cost on CPU: +{rows[1]['peak_gb'] - rows[0]['peak_gb']:.2f} GiB peak RSS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
