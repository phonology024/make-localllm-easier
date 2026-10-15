"""More compute at answer time vs a bigger model (#85 best-of-N / majority vote, #87 mixture-of-agents / debate).

  python tools/more_compute.py single --model gemma4-26b-a4b-qat                 # the baseline: one answer, temp 0
  python tools/more_compute.py vote   --file E4B.gguf --n 5                       # 5 samples, majority answer
  python tools/more_compute.py moa    --file E2B.gguf --file E4B.gguf             # both answer, the last one combines
  python tools/more_compute.py debate --file E2B.gguf --file E4B.gguf             # both answer, see each other, revise
  python tools/more_compute.py passn  --file E4B.gguf --n 5                       # code: pass@1 vs pass@N (sandbox)

Math is MGSM (checkable: a number) in --langs, the first --limit problems of each; code is HumanEval+ in the Docker
sandbox. Every run records accuracy, wall-clock seconds and generated tokens, so methods compare at equal cost, not
only at equal accuracy. All models are started with localllm's launch flags and stay loaded together (the small pair
fits in 16 GB). One JSON line per run in tools/more_compute_results.jsonl.

pass@N is an upper bound: it counts a problem as solved when any of N programs passes the hidden tests, which a real
user can't check. It says how much a verifier could win, not what --think-more would deliver.
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from localllm import bench, catalog, cli, runtime  # noqa: E402

RESULTS = ROOT / "tools" / "more_compute_results.jsonl"
MOA_SYSTEM = ("You are given a math problem and answers from other assistants. They may be wrong. Check their reasoning, "
              "solve the problem yourself, briefly. End with a last line of the form 'Answer: <number>'.")


class Server:
    def __init__(self, model: Path, port: int, slots: int, mtp=False):
        server, _devs, dev, _ram = cli._machine()
        # 8k per slot: the MoA / debate prompt holds the problem plus every model's worked answer (3k overflowed: HTTP 400)
        args = runtime.server_args(model, dev["id"] if dev else None, port, 8192 * slots, mtp) + ["-np", str(slots)]
        self.name, self.url = model.stem, f"http://127.0.0.1:{port}"
        log = open(Path(__file__).with_name("more_compute_server.log"), "ab")
        self.proc = subprocess.Popen([str(server), *args], env=runtime.server_env(None, False), stdout=log,
                                     stderr=subprocess.STDOUT)
        for _ in range(1200):
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited for {model.name}, see more_compute_server.log")
            try:
                if b'"ok"' in urllib.request.urlopen(self.url + "/health", timeout=2).read():
                    return
            except OSError:
                time.sleep(0.5)
        raise RuntimeError(f"{model.name} did not load")

    def chat(self, messages: list[dict], temp: float, max_tokens: int = 600, seed: int = 0) -> tuple[str, int]:
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": temp, "seed": seed,
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=1800))
        return d["choices"][0]["message"].get("content") or "", d.get("usage", {}).get("completion_tokens", 0)

    def close(self):
        self.proc.terminate()
        self.proc.wait(30)


def solve(s: Server, q: str, temp=0.0, seed=0):
    return s.chat([{"role": "system", "content": bench.MATH_SYSTEM}, {"role": "user", "content": q}], temp, seed=seed)


def majority(nums: list) -> float | None:
    got = [n for n in nums if n is not None]
    return collections.Counter(got).most_common(1)[0][0] if got else None


def math_item(method: str, servers: list[Server], item: dict, n: int) -> tuple[bool, int]:
    q, toks = item["q"], 0
    if method == "single":
        text, toks = solve(servers[0], q)
        final = bench.final_number(text)
    elif method == "vote":
        with ThreadPoolExecutor(n) as ex:
            outs = list(ex.map(lambda i: solve(servers[0], q, 0.7, seed=i + 1), range(n)))
        toks = sum(t for _, t in outs)
        final = majority([bench.final_number(t) for t, _ in outs])
    elif method == "moa":
        with ThreadPoolExecutor(len(servers)) as ex:
            outs = list(ex.map(lambda s: solve(s, q), servers))
        toks = sum(t for _, t in outs)
        shown = "\n\n".join(f"Assistant {i + 1}:\n{t}" for i, (t, _) in enumerate(outs))
        text, t2 = servers[-1].chat([{"role": "system", "content": MOA_SYSTEM},
                                     {"role": "user", "content": f"Problem:\n{q}\n\n{shown}"}], 0.0)
        toks += t2
        final = bench.final_number(text)
    elif method == "debate":
        with ThreadPoolExecutor(len(servers)) as ex:
            first = list(ex.map(lambda s: solve(s, q), servers))
            toks = sum(t for _, t in first)

            def revise(i):
                others = "\n\n".join(f"Another assistant:\n{t}" for j, (t, _) in enumerate(first) if j != i)
                return servers[i].chat([{"role": "system", "content": MOA_SYSTEM},
                                        {"role": "user", "content": f"Problem:\n{q}\n\nYour first answer:\n{first[i][0]}"
                                                                    f"\n\n{others}\n\nGive your final answer."}], 0.0)
            second = list(ex.map(revise, range(len(servers))))
        toks += sum(t for _, t in second)
        nums = [bench.final_number(t) for t, _ in second]
        final = majority(nums) if len(set(nums)) < len(nums) else nums[-1]     # tie: the larger (last) model
    else:
        raise ValueError(method)
    return final is not None and abs(final - item["ans"]) < 1e-6, toks


def run_math(method: str, servers: list[Server], langs: list[str], limit: int, n: int) -> dict:
    res = {}
    for lang in langs:
        items = bench.load("math", lang)[:limit]
        t0 = time.time()
        with ThreadPoolExecutor(2 if method == "single" else 1) as ex:
            got = list(ex.map(lambda it: math_item(method, servers, it, n), items))
        acc = round(100 * sum(ok for ok, _ in got) / len(got), 1)
        res[lang] = {"acc": acc, "n": len(got), "seconds": round(time.time() - t0, 1),
                     "tokens": sum(t for _, t in got)}
        print(f"  {method:6} {lang:3} {acc:5.1f}%  {res[lang]['seconds']:7.1f} s  {res[lang]['tokens']:7} tokens",
              flush=True)
    return res


def run_passn(server: Server, n: int, limit: int) -> dict:
    from localllm import sandbox
    items = [it for it in bench.load("codegen", "en") if it["id"].startswith("HumanEval")][:limit or None]
    t0, toks, progs = time.time(), 0, []
    for it in items:
        msgs = [{"role": "system", "content": bench.CODEGEN_SYSTEM}, {"role": "user", "content": it["prompt"]}]
        with ThreadPoolExecutor(n) as ex:
            outs = list(ex.map(lambda i: server.chat(msgs, 0.0 if i == 0 else 0.7, 1200, seed=i), range(n)))
        toks += sum(t for _, t in outs)
        progs.append([bench.codegen_program(it, text) for text, _ in outs])
    secs = round(time.time() - t0, 1)
    flat = sandbox.run([p for ps in progs for p in ps], bench.CODEGEN_TIMEOUT)
    ok = [[flat[i * n + j]["ok"] for j in range(n)] for i in range(len(items))]
    res = {"pass@1": round(100 * sum(o[0] for o in ok) / len(ok), 1),
           f"pass@{n}": round(100 * sum(any(o) for o in ok) / len(ok), 1), "n": len(ok), "seconds": secs,
           "tokens": toks}
    print(f"  passn  HumanEval+ pass@1 {res['pass@1']}%  pass@{n} {res[f'pass@{n}']}%  {secs} s", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("method", choices=["single", "vote", "moa", "debate", "passn"])
    ap.add_argument("--model", action="append", default=[], help="catalog key (repeatable)")
    ap.add_argument("--file", action="append", default=[], type=Path, help="GGUF path (repeatable)")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--langs", default="en,th,ja,zh,bn,sw")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    models = [(cli._model_path(k), catalog.MODELS[k]["mtp"]) for k in a.model] + [(f, False) for f in a.file]
    slots = a.n if a.method in ("vote", "passn") else 2
    servers = []
    try:
        for i, (m, mtp) in enumerate(models):
            servers.append(Server(m, 18140 + i, slots, mtp))
        if a.method == "passn":
            scores = {"en/codegen": run_passn(servers[0], a.n, a.limit)}
        else:
            scores = run_math(a.method, servers, a.langs.split(","), a.limit, a.n)
    finally:
        for s in servers:
            s.close()
    rec = {"date": time.strftime("%Y-%m-%d"), "method": a.method, "label": a.label,
           "models": [s.name for s in servers], "n": a.n if a.method in ("vote", "passn") else None,
           "limit": a.limit, "llama_cpp": runtime.build_of(runtime.find_server()), "scores": scores}
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
