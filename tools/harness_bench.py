"""Does a verify-and-retry loop make a local model better at code? (#88, the first measurement of the harness idea)

  python tools/harness_bench.py --model gemma4-26b-a4b-qat                 # bare vs loop on HumanEval+ and MBPP+
  python tools/harness_bench.py --model qwen3.8-27b-q3 --tries 3 --limit 100

The loop is the smallest harness that can win: write the function, run the checks the task itself shows (HumanEval's
docstring examples, MBPP's one given assert) in the Docker sandbox, and when they fail send the error back and ask for
a fix, up to --tries answers. Bare = the first answer. Both are then judged by the hidden EvalPlus tests the model never
saw, so the loop gets no help from them. Records pass rate, wall time and generated tokens for both, so the gain can be
weighed against its cost. One JSON line per run in tools/harness_results.jsonl.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from localllm import bench, catalog, cli, runtime, sandbox  # noqa: E402

PORT = 18160


def visible_checks(item: dict) -> str:
    """The checks a user would have from the task text alone: doctest examples or the one given assert."""
    if "Your code should pass this test:" in item["prompt"]:
        return item["prompt"].split("Your code should pass this test:", 1)[1].strip()
    out, lines = [], item["prompt"].splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"\s*>>>\s*(.+)", line)
        if m and i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].strip().startswith(">>>"):
            out.append(f"assert ({m.group(1)}) == ({lines[i + 1].strip()}), {m.group(1)!r}")
    return "\n".join(out)


def program(item: dict, reply: str, tests: str) -> str:
    code = bench.code_block(reply)
    head = item["head"] if item["id"].startswith("HumanEval") else "import math\n"
    return "\n".join([head, code, "", tests])


def chat(url: str, msgs: list[dict]) -> tuple[str, int]:
    body = {"messages": msgs, "max_tokens": 1200, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=900))
    return d["choices"][0]["message"].get("content") or "", d.get("usage", {}).get("completion_tokens", 0)


def solve(url: str, item: dict, tries: int) -> dict:
    msgs = [{"role": "system", "content": bench.CODEGEN_SYSTEM}, {"role": "user", "content": item["prompt"]}]
    checks, replies, toks, t0 = visible_checks(item), [], 0, time.time()
    first_s = None
    for n in range(tries):
        reply, k = chat(url, msgs)
        replies.append(reply)
        toks += k
        if first_s is None:
            first_s = time.time() - t0
        if not checks:
            break
        res = sandbox.run([program(item, reply, checks)], 20.0)[0]
        if res["ok"]:
            break
        err = "timed out" if res.get("err") == "timeout" else (res.get("err") or f"exit code {res.get('rc')}")[-1500:]
        msgs += [{"role": "assistant", "content": reply},
                 {"role": "user", "content": f"Running the checks from the task failed:\n```\n{err}\n```\n"
                                             "Fix the function. Reply with the complete corrected code block."}]
    return {"first": replies[0], "last": replies[-1], "answers": len(replies), "tokens": toks,
            "first_s": first_s, "s": time.time() - t0, "has_checks": bool(checks)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="gemma4-26b-a4b-qat", choices=list(catalog.MODELS))
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if not sandbox.docker():
        sys.exit("needs Docker (Linux containers): model-written code only runs in the sandbox")
    items = bench.load("codegen", "en")[:a.limit or None]
    server, _devs, dev, _ram = cli._machine()
    args = runtime.server_args(cli._model_path(a.model), dev["id"], PORT, 16384, catalog.MODELS[a.model]["mtp"])
    log = open(Path(__file__).with_name("harness_server.log"), "ab")
    proc = subprocess.Popen([str(server), *args], env=runtime.server_env(None, catalog.MODELS[a.model].get("vk_fix", True)),
                            stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{PORT}"
    for _ in range(1200):
        try:
            if b'"ok"' in urllib.request.urlopen(url + "/health", timeout=2).read():
                break
        except OSError:
            time.sleep(0.5)
    rows = []
    try:
        for i, it in enumerate(items, 1):
            rows.append(solve(url, it, a.tries))
            if i % 50 == 0:
                print(f"  {i}/{len(items)}", flush=True)
    finally:
        proc.terminate()
        proc.wait(30)
    hidden = sandbox.run([bench.codegen_program(it, r["first"]) for it, r in zip(items, rows)] +
                         [bench.codegen_program(it, r["last"]) for it, r in zip(items, rows)], bench.CODEGEN_TIMEOUT)
    bare, loop = hidden[:len(items)], hidden[len(items):]
    split = {"humaneval": [i for i, it in enumerate(items) if it["id"].startswith("HumanEval")],
             "mbpp": [i for i, it in enumerate(items) if it["id"].startswith("Mbpp")]}
    res = {}
    for name, idx in split.items():
        if idx:
            res[name] = {"n": len(idx), "bare": round(100 * sum(bare[i]["ok"] for i in idx) / len(idx), 1),
                         "loop": round(100 * sum(loop[i]["ok"] for i in idx) / len(idx), 1),
                         "retried": sum(rows[i]["answers"] > 1 for i in idx),
                         "no_checks": sum(not rows[i]["has_checks"] for i in idx)}
    cost = {"bare_s": round(sum(r["first_s"] for r in rows), 1), "loop_s": round(sum(r["s"] for r in rows), 1),
            "loop_tokens": sum(r["tokens"] for r in rows), "answers": sum(r["answers"] for r in rows)}
    rec = {"date": time.strftime("%Y-%m-%d"), "model": a.model, "tries": a.tries,
           "llama_cpp": runtime.build_of(server), "results": res, "cost": cost}
    with open(ROOT / "tools" / "harness_results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    for name, r in res.items():
        print(f"{a.model} {name}: bare {r['bare']}%  loop {r['loop']}%  ({r['retried']} retried, n={r['n']})")
    print(f"time: bare {cost['bare_s']} s, loop {cost['loop_s']} s; {cost['answers']} answers")


if __name__ == "__main__":
    main()
