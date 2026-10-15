"""A 30-turn tool-calling session at 20k+ tokens: RAM, VRAM, prefill and tok/s per turn (#77; feeds #49 and #96).

  python tools/agent_session.py --model gemma4-26b-a4b-qat                    # every setup x both APIs
  python tools/agent_session.py --model qwen3.8-27b-q3 --setups localllm-max --apis anthropic --turns 10

What a coding agent does to a local endpoint: a long system prompt with tool definitions, file reads that stay in the
history, a growing context with a stable prefix (so the prompt cache matters), several tool rounds per user turn.
The tools are real and read-only (list_files / read_file / search over this repository), so the session is the same
every run for the same model. When the history passes 80% of the context the oldest turns are dropped, the way agents
compact - which is also what breaks prompt caches.

Setups, all at the same context (--ctx, default 32768):
  llama.cpp-defaults   llama-server -m M -c CTX -ngl 999 (stock caches: f16 KV, 8 GB prompt cache, mmap)
  localllm-lean        localllm's launch flags, lean profile, at CTX
  localllm-max         localllm's launch flags, max profile (bigger prompt cache, more checkpoints), at CTX
APIs: openai (/v1/chat/completions) and anthropic (/v1/messages through the localllm gateway).

Per turn: the server's private bytes / working set (Windows), dedicated VRAM and shared (spilled) GPU memory, prompt
tokens, prefill ms, cached tokens, decode tok/s, tool calls and malformed tool calls. One JSON line per session in
tools/agent_session_results.jsonl.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from localllm import catalog, cli, gateway, router, runtime  # noqa: E402
from measure_footprint import proc_stats  # noqa: E402

PORT = 18124
SYSTEM = ("You are a coding assistant working in a Python repository (make-localllm-easier: a CLI that runs local LLMs "
          "with llama.cpp). You can only read the repository, through the tools. Always use a tool to look at the "
          "code before you answer; never guess file contents. Answer in at most five sentences and name the files "
          "and functions you rely on. Paths are relative to the repository root, with forward slashes.")
TOOLS = [
    ("list_files", "List the files in a directory of the repository.",
     {"type": "object", "properties": {"path": {"type": "string", "description": "directory, e.g. localllm"}},
      "required": ["path"]}),
    ("read_file", "Read a text file of the repository (at most 6000 characters from `start`).",
     {"type": "object", "properties": {"path": {"type": "string"}, "start": {"type": "integer", "description": "offset"}},
      "required": ["path"]}),
    ("search", "Search the repository's Python and Markdown files for a regular expression; returns file:line: text.",
     {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}),
]
QUESTIONS = [
    "What does localllm/pool.py do when a request names a model?",
    "How does the gateway decide which endpoints are metered for the savings counter?",
    "Where is the context length chosen for the max profile, and what caps it?",
    "How is a llama.cpp download verified before it is used?",
    "What happens in `localllm connect obsidian` when the vault has no Copilot plugin?",
    "Which files does the --ui chat page store, and where?",
    "How does bench.py score the long-context needle test?",
    "What does tools/runner.py write for a failed job, and is it retried?",
    "How does `doctor` estimate the RAM the prompt cache uses?",
    "อธิบายว่า localllm/router.py เลือกโมเดลตามภาษาอย่างไร",
    "Where does the README doctor sample test get its fixture GPU from?",
    "What does the Sessions class refuse to save, and why?",
    "How are Ollama requests translated to llama-server's API?",
    "Which environment variable pins the llama.cpp build, and what does 'latest' do?",
    "How does the gateway handle a 401 for LAN clients?",
    "สรุปว่า localllm/meter.py นับ token ที่ตอบในเครื่องอย่างไร",
    "What does tools/smoke_build.py compare between two builds?",
    "Where are Gemini requests translated, and how are tool calls mapped?",
    "How does the idle unload in pool.py avoid unloading in the middle of an answer?",
    "Which flags does runtime.server_args always pass to llama-server?",
    "How does catalog.pick choose a model for a language?",
    "What does `localllm savings --price` store?",
    "How does chat.py show the /status line?",
    "What is checked by tests/test_stability.py?",
    "How does the task classifier start its own server?",
    "อธิบายการทำงานของ --profile max ใน doctor",
    "Where does report.py scrub personal data from a report?",
    "How does the leaderboard page escape report text?",
    "What does packaging/build_exe.py smoke-test after building?",
    "Which parts of the website are generated from the catalog?",
]


def tool_run(name: str, args: dict) -> str:
    def inside(p: str) -> Path:
        q = (ROOT / p.strip("/")).resolve()
        if ROOT not in q.parents and q != ROOT:
            raise ValueError("outside the repository")
        return q
    try:
        if name == "list_files":
            d = inside(args.get("path") or ".")
            return "\n".join(sorted(x.name + ("/" if x.is_dir() else "") for x in d.iterdir()
                                    if not x.name.startswith((".", "__"))))
        if name == "read_file":
            start = int(args.get("start") or 0)
            return inside(args["path"]).read_text(encoding="utf-8", errors="replace")[start:start + 6000]
        if name == "search":
            rx, hits = re.compile(args["pattern"]), []
            for f in sorted(ROOT.glob("**/*")):
                if f.suffix in (".py", ".md") and ".git" not in f.parts and "site" not in f.parts[len(ROOT.parts):][:1]:
                    for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                        if rx.search(line):
                            hits.append(f"{f.relative_to(ROOT).as_posix()}:{i}: {line.strip()[:160]}")
            return "\n".join(hits[:40]) or "no matches"
        return f"error: unknown tool {name}"
    except Exception as e:  # noqa: BLE001 - the model sees the error, as in a real agent
        return f"error: {type(e).__name__}: {e}"


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


class OpenAISession:
    def __init__(self, url: str):
        self.url, self.msgs = url + "/v1/chat/completions", [{"role": "system", "content": SYSTEM}]
        self.tools = [{"type": "function", "function": {"name": n, "description": d, "parameters": p}}
                      for n, d, p in TOOLS]

    def user(self, text: str) -> None:
        self.msgs.append({"role": "user", "content": text})

    def step(self) -> tuple[dict, list]:
        d = post(self.url, {"messages": self.msgs, "tools": self.tools, "max_tokens": 400, "temperature": 0,
                            "chat_template_kwargs": {"enable_thinking": False}})
        m = d["choices"][0]["message"]
        self.msgs.append({k: v for k, v in m.items() if k in ("role", "content", "tool_calls")})
        t = d.get("timings", {})
        u = d.get("usage", {})
        stats = {"prompt_tokens": u.get("prompt_tokens"), "prefill_ms": t.get("prompt_ms"), "cached": t.get("cache_n"),
                 "tok_s": t.get("predicted_per_second"), "out_tokens": u.get("completion_tokens")}
        calls = []
        for c in m.get("tool_calls") or []:
            try:
                args, ok = json.loads(c["function"].get("arguments") or "{}"), True
            except ValueError:
                args, ok = {}, False
            calls.append((c["function"]["name"], args, ok))
            self.msgs.append({"role": "tool", "tool_call_id": c.get("id", ""),
                              "content": tool_run(c["function"]["name"], args) if ok else "error: arguments are not JSON"})
        return stats, calls

    def estimate_tokens(self) -> int:
        return len(json.dumps(self.msgs, ensure_ascii=False)) // 3     # ~3 characters per token, on the safe side

    def drop_oldest(self) -> bool:
        """Drop the oldest user turn with its replies and tool results (keeps the system prompt and the current turn)."""
        nxt = next((i for i in range(2, len(self.msgs)) if self.msgs[i]["role"] == "user"
                    and isinstance(self.msgs[i].get("content"), str)), None)
        if nxt:
            del self.msgs[1:nxt]
        return bool(nxt)


class AnthropicSession:
    def __init__(self, url: str):
        self.url, self.msgs = url + "/v1/messages", []
        self.tools = [{"name": n, "description": d, "input_schema": p} for n, d, p in TOOLS]

    def user(self, text: str) -> None:
        self.msgs.append({"role": "user", "content": [{"type": "text", "text": text}]})

    def step(self) -> tuple[dict, list]:
        t0 = time.time()
        d = post(self.url, {"model": "local", "system": SYSTEM, "messages": self.msgs, "tools": self.tools,
                            "max_tokens": 400, "temperature": 0})
        self.msgs.append({"role": "assistant", "content": d["content"]})
        u = d.get("usage", {})
        stats = {"prompt_tokens": (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0),
                 "cached": u.get("cache_read_input_tokens"), "out_tokens": u.get("output_tokens"),
                 "wall_s": round(time.time() - t0, 2)}
        calls, results = [], []
        for b in d["content"]:
            if b.get("type") == "tool_use":
                ok = isinstance(b.get("input"), dict)
                calls.append((b["name"], b.get("input") or {}, ok))
                results.append({"type": "tool_result", "tool_use_id": b["id"],
                                "content": tool_run(b["name"], b["input"]) if ok else "error: input is not an object"})
        if results:
            self.msgs.append({"role": "user", "content": results})
        return stats, calls

    def estimate_tokens(self) -> int:
        return (len(json.dumps(self.msgs, ensure_ascii=False)) + len(SYSTEM)) // 3

    def drop_oldest(self) -> bool:
        firsts = [i for i, m in enumerate(self.msgs) if m["role"] == "user"
                  and any(b.get("type") == "text" for b in m["content"])]
        if len(firsts) > 1:
            del self.msgs[:firsts[1]]
        return len(firsts) > 1


def start_server(setup: str, key: str, ctx: int) -> tuple[subprocess.Popen, list[str]]:
    server, _devs, dev, _ram = cli._machine()
    model = cli._model_path(key)
    if setup == "llama.cpp-defaults":
        args = ["-m", str(model), "--host", "127.0.0.1", "--port", str(PORT), "-c", str(ctx), "-ngl", "999",
                "--jinja", "-dev", dev["id"]]
    else:
        profile = setup.split("-")[1]
        args = runtime.server_args(model, dev["id"], PORT, ctx, catalog.MODELS[key]["mtp"], profile=profile)
        args += ["--jinja"]
    env = runtime.server_env(None, catalog.MODELS[key].get("vk_fix", True))
    log = open(Path(__file__).with_name("agent_session_server.log"), "ab")
    proc = subprocess.Popen([str(server), *args], env=env, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{PORT}"
    for _ in range(1200):
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited ({proc.returncode}), see agent_session_server.log")
        try:
            if b'"ok"' in urllib.request.urlopen(url + "/health", timeout=2).read():
                return proc, args
        except OSError:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError("llama-server did not start")


def session(key: str, setup: str, api: str, turns: int, ctx: int) -> dict:
    proc, args = start_server(setup, key, ctx)
    g = None
    try:
        url = f"http://127.0.0.1:{PORT}"
        if api == "anthropic":
            router.load_config = lambda: {"enabled": False}
            g = gateway.serve(url, port=0, model_name="local")
            url = f"http://127.0.0.1:{g.server_address[1]}"
        s = OpenAISession(url) if api == "openai" else AnthropicSession(url)
        loaded = {**proc_stats(proc.pid), **runtime.process_memory(proc.pid)}
        rows, t_start = [], time.time()
        for i in range(turns):
            s.user(QUESTIONS[i % len(QUESTIONS)])
            reqs, calls, bad, t0 = [], 0, 0, time.time()
            overflow = False
            for _ in range(6):                                  # tool rounds per user turn
                # compact before every request, not only between turns: a turn's own tool results can add ~9k tokens
                while s.estimate_tokens() > 0.7 * ctx and s.drop_oldest():
                    pass
                if s.estimate_tokens() > 0.9 * ctx:             # this turn alone doesn't fit: stop it, count it
                    overflow = True
                    break
                stats, cs = s.step()
                reqs.append(stats)
                calls += len(cs)
                bad += sum(not ok or n not in [t[0] for t in TOOLS] for n, _a, ok in cs)
                if not cs:
                    break
            if not reqs:
                reqs.append({})
            mem = {**proc_stats(proc.pid), **runtime.process_memory(proc.pid)}
            row = {"turn": i + 1, "requests": len(reqs), "tool_calls": calls, "bad_calls": bad, "overflow": overflow,
                   "turn_s": round(time.time() - t0, 1), "prompt_tokens": reqs[-1].get("prompt_tokens"),
                   "prefill_ms": sum(r.get("prefill_ms") or 0 for r in reqs) or None,
                   "cached": reqs[-1].get("cached"), "tok_s": reqs[-1].get("tok_s"),
                   **{k: round(v, 3) for k, v in mem.items() if isinstance(v, float)}}
            rows.append(row)
            print(f"  {setup:18} {api:9} turn {i + 1:2}: ctx {row['prompt_tokens']!s:>6}  tools {calls} (bad {bad})  "
                  f"private {row.get('private_gb', 0):.2f} GB  vram {row.get('vram_gb') or 0:.2f} GB  "
                  f"shared {row.get('spill_gb') or 0:.2f} GB  {row['turn_s']} s", flush=True)
    finally:
        if g:
            g.shutdown()
        proc.terminate()
        proc.wait(30)
    calls = sum(r["tool_calls"] for r in rows)
    return {"date": time.strftime("%Y-%m-%d"), "model": key, "setup": setup, "api": api, "ctx": ctx,
            "llama_cpp": runtime.build_of(runtime.find_server()), "args": args[2:],
            "loaded": {k: round(v, 3) for k, v in loaded.items() if isinstance(v, float)},
            "max_private_gb": max(r.get("private_gb", 0) for r in rows),
            "max_vram_gb": max(r.get("vram_gb") or 0 for r in rows),
            "max_shared_gb": max(r.get("spill_gb") or 0 for r in rows),
            "max_prompt_tokens": max(r["prompt_tokens"] or 0 for r in rows),
            "tool_calls": calls, "bad_call_rate": round(sum(r["bad_calls"] for r in rows) / max(1, calls), 3),
            "total_s": round(time.time() - t_start, 1), "turns": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="gemma4-26b-a4b-qat", choices=list(catalog.MODELS))
    ap.add_argument("--setups", default="llama.cpp-defaults,localllm-lean,localllm-max")
    ap.add_argument("--apis", default="openai,anthropic")
    ap.add_argument("--turns", type=int, default=30)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--json", type=Path, default=ROOT / "tools" / "agent_session_results.jsonl")
    a = ap.parse_args()
    for setup in a.setups.split(","):
        for api in a.apis.split(","):
            try:
                res = session(a.model, setup, api, a.turns, a.ctx)
            except (RuntimeError, OSError, urllib.error.URLError) as e:
                print(f"!! {setup} {api}: {e}", flush=True)
                continue
            with open(a.json, "a", encoding="utf-8") as f:
                f.write(json.dumps(res) + "\n")
            print(f"== {a.model} {setup} {api}: max private {res['max_private_gb']} GB, VRAM {res['max_vram_gb']} GB, "
                  f"shared {res['max_shared_gb']} GB, ctx {res['max_prompt_tokens']}, {res['tool_calls']} tool calls "
                  f"({res['bad_call_rate']:.0%} bad), {res['total_s']} s", flush=True)


if __name__ == "__main__":
    main()
