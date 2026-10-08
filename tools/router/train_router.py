"""Embed the router data with e5-small (llama.cpp), train a logistic-regression head, evaluate vs the keyword rules,
and export the head the package loads (localllm/taskclf.py).

usage: train_router.py MODEL.gguf [--ngl 0|99] [--server llama-server] [--data DIR] [--with-generated [FILE]]
                       [--test shared_test.jsonl] [--export router_head.json]
  --data            folder with train.jsonl + heldout.jsonl from build_data.py (default: data/ next to this script)
  --with-generated  add gen_data.py's short requests (default: generated.jsonl next to this script), 85% to training
  --server          default: $LOCALLLM_LLAMA_SERVER, the maintainer's Vulkan build if present, else localllm's llama.cpp
"""
import argparse, hashlib, json, os, random, subprocess, sys, time, urllib.error, urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D.parents[1]))          # this repo, so `localllm` imports without installing it
from localllm import router, runtime, taskclf  # noqa: E402

LABELS = ["general", "math", "code", "translate"]
WIN_SERVER = r"C:\Users\user\tools\llama.cpp-b11457\vulkan\llama-server.exe"
ORDERS = {"embedding only": ("embedding",), "keyword first + embedding": ("keyword", "embedding"),
          "keyword rules alone": ("keyword",)}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("model")
ap.add_argument("--ngl", default="0")
ap.add_argument("--server", default=os.environ.get("LOCALLLM_LLAMA_SERVER")
                or (WIN_SERVER if Path(WIN_SERVER).exists() else None))
ap.add_argument("--data", type=Path, default=D / "data")
ap.add_argument("--with-generated", type=Path, nargs="?", const=D / "generated.jsonl", dest="generated")
ap.add_argument("--test", type=Path, default=D / "shared_test.jsonl")
ap.add_argument("--export", type=Path, default=D / "router_head.json")
A = ap.parse_args()
PORT = taskclf._free_port()


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def post(texts):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/embeddings", data=json.dumps({"input": texts}).encode(),
                                 headers={"Content-Type": "application/json"})
    return [d["embedding"] for d in json.load(urllib.request.urlopen(req, timeout=600))["data"]]


def embed(recs, tag):
    key = hashlib.sha1("\n".join(r["text"] for r in recs).encode()).hexdigest()[:8]   # new data -> new cache
    cache = A.data / f"emb-{tag}-{Path(A.model).stem}-{key}.npy"
    if cache.exists():
        return np.load(cache)
    out = []
    t = time.time()
    for i in range(0, len(recs), 32):
        batch = [taskclf.PREFIX + r["text"][:taskclf.MAX_CHARS] for r in recs[i:i + 32]]
        try:
            out += post(batch)
        except urllib.error.HTTPError:
            for one in batch:                   # find and shorten the text the server rejected
                try:
                    out += post([one])
                except urllib.error.HTTPError:
                    print("  shortened:", repr(one[:80])); out += post([one[:150]])
    print(f"  embedded {len(recs)} {tag} texts in {time.time() - t:.1f}s")
    x = np.array(out, dtype=np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    A.data.mkdir(parents=True, exist_ok=True)
    np.save(cache, x)
    return x


def report(name, y, pred, recs):
    acc = 100 * np.mean(np.array(y) == np.array(pred))
    per = {l: round(float(100 * np.mean([p == l for t, p in zip(y, pred) if t == l])), 1) for l in LABELS if l in y}
    bylang = defaultdict(list)
    for t, p, r in zip(y, pred, recs):
        bylang[r.get("lang", "?")].append(t == p)
    worst = sorted(((round(float(100 * np.mean(v)), 1), k, len(v)) for k, v in bylang.items()))[:4]
    print(f"  {name:40} {acc:5.1f}%  per class {per}  weakest langs {worst}")
    return round(float(acc), 1)


def export_head(clf, model, **meta) -> dict:
    """The JSON localllm/taskclf.py loads: softmax(coef . x/|x| + intercept) over `labels`."""
    return {"labels": [str(c) for c in clf.classes_], "coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist(),
            "model": Path(model).name, "prefix": taskclf.PREFIX, "max_chars": taskclf.MAX_CHARS, **meta}


class Precomputed:
    """taskclf.Classifier with the embeddings already computed: the shipped head math and combination code."""
    def __init__(self, head, recs, x):
        self.head, self.emb = head, {r["text"]: v for r, v in zip(recs, x.tolist())}

    def classify(self, text):
        return taskclf.predict(self.head, self.emb[text])


server = A.server or str(runtime.find_server())
srv = subprocess.Popen([server, "-m", A.model, "--embedding", "--pooling", "mean", "-ngl", A.ngl, "--port", str(PORT),
                        "-c", "8192", "-b", "8192", "-ub", "8192", "-np", "16"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(300):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1); break
        except OSError:
            time.sleep(0.2)
    train, held, test = load(A.data / "train.jsonl"), load(A.data / "heldout.jsonl"), load(A.test)
    if A.generated:                             # short everyday requests written by the local LLM
        gen = load(A.generated); random.Random(1).shuffle(gen)
        cut = len(gen) * 85 // 100
        train, held = train + gen[:cut], held + gen[cut:]
    print(f"train {len(train)} {dict(Counter(r['label'] for r in train))}  generated: {A.generated or 'no'}")
    print(f"held-out {len(held)}, shared test {len(test)} ({A.test})")
    xtr, xho, xte = embed(train, "train"), embed(held, "heldout"), embed(test, "test")
    ytr, yho, yte = [r["label"] for r in train], [r["label"] for r in held], [r["label"] for r in test]

    best = None
    for c in (0.5, 2, 8, 32):
        clf = LogisticRegression(C=c, max_iter=3000, class_weight="balanced").fit(xtr, ytr)
        a = np.mean(clf.predict(xho) == np.array(yho))
        print(f"  C={c}: held-out {100 * a:.1f}%")
        if best is None or a > best[0]:
            best = (a, c, clf)
    _, c, clf = best
    print(f"\nC={c}")
    report("held-out: embedding router", yho, clf.predict(xho), held)
    report("held-out: keyword rules", yho, [router.detect_task(r["text"]) for r in held], held)
    head = export_head(clf, A.model, C=c, train=dict(Counter(ytr)))
    pc = Precomputed(head, test, xte)
    same = sum(taskclf.predict(head, v)[0] == p for v, p in zip(xte.tolist(), clf.predict(xte)))
    print(f"  package head math agrees with sklearn on {same}/{len(test)} shared-test messages")
    shared = {name: report(f"shared test: {name}", yte, [router.classify_task(r["text"], pc, o)[0] for r in test], test)
              for name, o in ORDERS.items()}
    wrong = [(r["lang"], r["label"], str(p), r["text"][:60]) for r, p in zip(test, clf.predict(xte)) if p != r["label"]]
    print("\nshared-test mistakes (embedding only):", *wrong, sep="\n  ")

    # single-message latency (what a user feels): embed one text + classify with the package's head
    lat = []
    for r in test[:40]:
        t = time.perf_counter()
        taskclf.predict(head, post([taskclf.PREFIX + r["text"][:taskclf.MAX_CHARS]])[0])
        lat.append(1000 * (time.perf_counter() - t))
    print(f"\nlatency per message (ngl={A.ngl}, this training server): median {np.median(lat):.1f} ms, "
          f"p95 {np.percentile(lat, 95):.1f} ms")
    A.export.parent.mkdir(parents=True, exist_ok=True)
    A.export.write_text(json.dumps({**head, "shared_test": shared}), encoding="utf-8")
    print(f"head written to {A.export} ({A.export.stat().st_size / 1024:.1f} KB)")
finally:
    srv.terminate()
