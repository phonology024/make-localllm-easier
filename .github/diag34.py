# TEMPORARY diagnostics for issue #34 (removed before the PR): sizes, peak memory and run time of the canonical
# solutions that failed in the sandbox.
import ast, json, os, subprocess, sys, time
from localllm import bench, sandbox

he = bench._rows("evalplus/humanevalplus", "default")
mb = bench._rows("evalplus/mbppplus", "default")
print("columns", sorted(he[0]), sorted(mb[0]))
sizes = sorted([(len(r["test"]), r["task_id"]) for r in he] + [(len(r["test"]), f"Mbpp/{r['task_id']}") for r in mb])
print("largest tests (bytes):", sizes[-8:], "total", sum(s for s, _ in sizes))
want = {"Mbpp/255", "Mbpp/599", "HumanEval/32"}
progs = {}
for r in mb:
    i = f"Mbpp/{r['task_id']}"
    if i in want:
        print("=====", i, "test_imports", r.get("test_imports"), "len", len(r["test"]))
        print(r["code"])
        print(r["test"][:1500], "\n ... \n", r["test"][-800:])
        progs[i] = r["code"] + "\n\n" + r["test"]
for r in he:
    if r["task_id"] in want:
        print("=====", r["task_id"])
        print(r["test"][:600], "\n ... \n", r["test"][-600:])
os.makedirs("/tmp/diag", exist_ok=True)
for i, src in progs.items():
    f = f"/tmp/diag/{i.replace('/', '_')}.py"
    open(f, "w").write(src)
    t = time.time()
    p = subprocess.run(["/usr/bin/time", "-v", sys.executable, f], capture_output=True, text=True, timeout=900)
    rss = [l for l in p.stderr.splitlines() if "Maximum resident" in l or "Elapsed" in l]
    print(f"HOST {i}: rc={p.returncode} wall={time.time() - t:.1f}s {rss} {p.stderr[-300:] if p.returncode else ''}")
res = sandbox.run(list(progs.values()), timeout_s=300)
for i, r in zip(progs, res):
    print("SANDBOX(300s)", i, r)
