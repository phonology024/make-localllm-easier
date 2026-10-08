"""Measure the task router exactly as localllm ships it: localllm.taskclf (its own CPU llama-server started lazily,
pure-Python head) + router.classify_task, on the shared test set. Standard library + localllm only.

Prints accuracy per combination order, per-message latency, file sizes, and the X-Localllm-Model reasons a gateway in
front of two (fake) chat models reports with and without the embedding model.
usage: eval_package.py [--test shared_test.jsonl] [--json out.json]
  model + head: ~/.localllm/models (or $LOCALLLM_HOME/models, $LOCALLLM_MODELS); llama-server: $LOCALLLM_LLAMA_SERVER
"""
import argparse, json, statistics, sys, tempfile, threading, time, urllib.request
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D.parents[1]))
from localllm import gateway, pool, router, taskclf  # noqa: E402

SHIPPED = "keyword first + embedding (shipped)"
ORDERS = {"keyword rules alone": ("keyword",), "embedding only": ("embedding",), SHIPPED: ("keyword", "embedding"),
          "embedding first + keyword": ("embedding", "keyword")}


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, round(q / 100 * len(xs) + 0.5) - 1))]     # nearest rank


class Echo(BaseHTTPRequestHandler):
    """A chat model stand-in: the routing decision is what we look at, not the answer."""
    def log_message(self, *a):
        pass

    def do_GET(self):
        self._send({"status": "ok"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self._send({"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                    "echo": body.get("messages", [{}])[-1].get("content")})

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)


class Proc:
    def __init__(self, s):
        self.s = s

    def terminate(self):
        self.s.shutdown()

    def wait(self, _t=None):
        return 0


def launch(_key):
    s = HTTPServer(("127.0.0.1", 0), Echo)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return Proc(s), f"http://127.0.0.1:{s.server_port}"


def gateway_headers(clf, texts):
    p = pool.Pool(["qwen3.8-27b-q3", "gemma4-26b-a4b-qat"], launch, 15.9, first="gemma4-26b-a4b-qat", classifier=clf)
    g = gateway.serve(p, port=0, model_name="localllm-auto")
    out = []
    for t in texts:
        req = urllib.request.Request(f"http://127.0.0.1:{g.server_address[1]}/v1/chat/completions",
                                     data=json.dumps({"messages": [{"role": "user", "content": t}]}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            r.read()
            out.append(r.headers["X-Localllm-Model"])
    p.close(); g.shutdown()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", type=Path, default=D / "shared_test.jsonl")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    test = [json.loads(l) for l in open(a.test, encoding="utf-8") if l.strip()]
    res = {"messages": len(test)}

    clf = taskclf.Classifier(wait=taskclf.START_TIMEOUT_S)     # the first message waits for the start: cold start time
    gguf, head = clf.files()
    res["gguf_mb"], res["head_kb"] = round(gguf.stat().st_size / 2**20, 1), round(head.stat().st_size / 1024, 1)
    print(f"model {gguf} ({res['gguf_mb']} MB), head {head} ({res['head_kb']} KB), server args: "
          f"{' '.join(taskclf.server_args(gguf, 0)[2:])}")
    t = time.perf_counter()
    first = clf.classify(test[0]["text"])
    res["cold_start_ms"] = round(1000 * (time.perf_counter() - t))
    print(f"first message (starts the embedding server): {res['cold_start_ms']} ms -> {first}")

    res["accuracy"] = {}
    for name, order in ORDERS.items():
        got = [router.classify_task(r["text"], clf, order) for r in test]
        acc = 100 * sum(g[0] == r["label"] for g, r in zip(got, test)) / len(test)
        per = defaultdict(list)
        for g, r in zip(got, test):
            per[r["label"]].append(g[0] == r["label"])
        hows = defaultdict(int)
        for g in got:
            hows[g[1].split(" p=")[0]] += 1
        res["accuracy"][name] = round(acc, 1)
        print(f"  {name:38} {acc:5.1f}%  per class { {k: round(100 * sum(v) / len(v), 1) for k, v in per.items()} }"
              f"  decided by {dict(hows)}")

    for name in ("embedding only", SHIPPED):
        lat = []
        for r in test:
            t = time.perf_counter()
            router.classify_task(r["text"], clf, ORDERS[name])
            lat.append(1000 * (time.perf_counter() - t))
        res.setdefault("latency_ms", {})[name] = {"median": round(statistics.median(lat), 2),
                                                  "p95": round(pct(lat, 95), 2), "max": round(max(lat), 2)}
        print(f"  latency per message, {name}: median {statistics.median(lat):.1f} ms, p95 {pct(lat, 95):.1f} ms, "
              f"max {max(lat):.1f} ms")
    h = taskclf.load_head(head)
    emb = [0.01 * (i % 7) for i in range(len(h["coef"][0]))]
    t = time.perf_counter()
    for _ in range(200):
        taskclf.predict(h, emb)
    res["head_math_ms"] = round(1000 * (time.perf_counter() - t) / 200, 3)
    print(f"  of which the pure-Python head: {res['head_math_ms']} ms")

    probe = ["เขียนฟังก์ชัน Python เรียงลำดับรายการให้หน่อย", "What is 17 * 23?",
             "Comment dit-on « merci » en japonais ?", "请用三句话解释为什么天空是蓝色的。"]
    proc = clf.proc
    res["gateway_with_model"] = gateway_headers(clf, probe)
    with tempfile.TemporaryDirectory() as empty:
        res["gateway_without_model"] = gateway_headers(taskclf.Classifier(dirs=[empty]), probe)
    print("\nX-Localllm-Model with the embedding model:", *res["gateway_with_model"], sep="\n  ")
    print("X-Localllm-Model without it:", *res["gateway_without_model"], sep="\n  ")
    res["server_stopped_by_pool_close"] = proc is not None and proc.poll() is not None
    print(f"embedding server stopped when the pool closed: {res['server_stopped_by_pool_close']}")
    print("\nRESULT " + json.dumps(res, ensure_ascii=False))
    if a.json:
        a.json.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
