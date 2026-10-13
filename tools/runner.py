"""Measurement runner: a queue of benchmark jobs that runs unattended, e.g. overnight (#91).

  python tools/runner.py MANIFEST.json                 run every job not finished yet (resumes after a crash)
  python tools/runner.py MANIFEST.json --dry-run       print the launches, start nothing
  python tools/runner.py MANIFEST.json --reports DIR   also write each finished job as a report file into DIR

Manifest (see tools/runner_example.json):

  {"llama_cpp": "b11476",                    # refuse to run on any other build: results must be comparable
   "jobs": [{"id": "kv-q4-th",               # unique; a finished id is skipped next time
             "model": "qwen3.8-27b-q3",      # catalog key (downloaded if needed), or "file": "path/to/x.gguf"
             "ctx": 8192, "args": ["-ctk", "q4_0", "-ctv", "q4_0"], "env": {},   # env: exactly these vars
             "suites": "global,regional", "langs": "en,th", "limit": 0, "speed": true,
             "vision": false,                        # true: load the catalog model's projector (--mmproj)
             "memory": false}]}                      # true: record the server's RAM / VRAM / spill after load and at the end

Each finished job is one line in tools/runner_results.jsonl, written right away. A line is a valid `localllm report`
(schema 1: gpu, vram_gb, ram_gb, os, llama_cpp, tune, results) plus the job and the machine state before it started
(free VRAM, free RAM, driver), so tools/build_leaderboard.py can read it. A job that fails is recorded with the error
and retried on the next run.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from localllm import __version__, bench, catalog, report, runtime  # noqa: E402

RESULTS = Path(__file__).with_name("runner_results.jsonl")
PORT = 18301


def done_ids(results: Path) -> set[str]:
    if not results.exists():
        return set()
    out = set()
    for line in results.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("job", {}).get("status") == "ok":
                out.add(r["job"]["id"])
    return out


def validate(manifest: dict) -> list[str]:
    why = []
    if not manifest.get("llama_cpp"):
        why.append("manifest needs llama_cpp (the build every job runs on)")
    ids = [j.get("id") for j in manifest.get("jobs", [])]
    if len(ids) != len(set(ids)) or not all(ids):
        why.append("every job needs a unique id")
    for j in manifest.get("jobs", []):
        if not (j.get("model") in catalog.MODELS or j.get("file")):
            why.append(f"{j.get('id')}: model must be a catalog key, or give a file")
    return why


def state(server: Path) -> dict:
    """What the machine looks like right before a job: free VRAM, free RAM, driver."""
    devs = runtime.devices(server)
    dev = runtime.best_device(devs)
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "gpu": dev["name"] if dev else None,
            "vram_gb": round(dev["total_gb"], 1) if dev else 0.0,
            "vram_free_gb": round(dev["free_gb"], 1) if dev else 0.0, "dev_id": dev["id"] if dev else None,
            "ram_gb": round(runtime.ram_gb(), 1), "ram_free_gb": round(runtime.ram_available_gb(), 1),
            "driver": report.driver() if dev else None}


def launch_args(job: dict, model: Path, dev_id: str | None) -> list[str]:
    key = job.get("model")
    mtp = catalog.MODELS[key]["mtp"] if key in catalog.MODELS else False
    mtp = job.get("mtp", mtp)                                    # 0 = off, N = draft N tokens (models with an MTP head)
    args = runtime.server_args(model, dev_id, PORT, job.get("ctx", 8192), mtp)
    return args + [str(a) for a in job.get("args", [])]          # later flags win in llama.cpp


def wait_up(proc, url: str, timeout_s: float = 600) -> None:
    import urllib.request
    t = time.time()
    while time.time() - t < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with {proc.returncode} while loading")
        try:
            if b'"ok"' in urllib.request.urlopen(url + "/health", timeout=2).read():
                return
        except OSError:
            pass
        time.sleep(0.3)
    raise RuntimeError("model did not load in time")


def decode_speed(url: str) -> float:
    import urllib.request
    speeds = []
    for prompt in ("Warm up.", "Write a short paragraph about the ocean and the moon.",
                   "Explain in a few sentences how a bicycle stays upright."):
        body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 160, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        speeds.append(json.load(urllib.request.urlopen(req, timeout=600))["timings"]["predicted_per_second"])
    return round(sum(speeds[1:]) / len(speeds[1:]), 1)


def run_job(job: dict, build: str, server: Path, model_path, popen=subprocess.Popen) -> dict:
    """One job -> one report-shaped dict. Raises on failure (the caller records it)."""
    before = state(server)
    model = Path(job["file"]).expanduser() if job.get("file") else model_path(job["model"])
    args = launch_args(job, model, before["dev_id"])
    if job.get("vision"):                                         # the catalog model's projector, as --vision loads it
        from localllm.cli import _mmproj_path
        args += ["--mmproj", str(_mmproj_path(job["model"]))]
    env = runtime.server_env({"env": job.get("env", {})})
    url = f"http://127.0.0.1:{PORT}"
    proc = popen([str(server), *args], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        wait_up(proc, url)
        pid = getattr(proc, "pid", None)
        memory = {"loaded": runtime.process_memory(pid)} if job.get("memory") and pid else {}
        tok_s = decode_speed(url) if job.get("speed", True) else None
        scores = {}
        if job.get("suites"):
            langs = job.get("langs", "en").split(",")
            scores = bench.run(url, job["id"], langs, job.get("limit", 0), tuple(job["suites"].split(",")))
        if memory:
            memory["after"] = runtime.process_memory(pid)
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
    return {
        "schema": report.SCHEMA, "localllm": __version__, "date": time.strftime("%Y-%m-%d"),
        "os": f"{platform.system()} {platform.release()}", "arch": platform.machine(),
        "ram_gb": before["ram_gb"], "llama_cpp": build, "gpu": before["gpu"], "vram_gb": before["vram_gb"],
        "driver": before["driver"],
        "tune": {model.name: {build: {"gpu": before["gpu"], "env": job.get("env", {}), "mtp": 0, "tok_s": tok_s}}}
        if tok_s else {},
        "results": {job["id"]: scores} if scores else {},
        "job": {**job, "status": "ok", "args_full": report.scrub(args), "state_before": before,
                **({"memory": memory} if memory else {})},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reports", type=Path, help="also write each finished job here as <id>.json")
    ap.add_argument("--results", type=Path, default=RESULTS)
    a = ap.parse_args()
    manifest = json.loads(a.manifest.read_text(encoding="utf-8"))
    why = validate(manifest)
    if why:
        sys.exit("manifest: " + "; ".join(why))
    build = manifest["llama_cpp"]
    import os
    os.environ.setdefault("LOCALLLM_LLAMA_BUILD", build)
    server = runtime.find_server()
    if server.parent.name != build and not os.environ.get("LOCALLLM_LLAMA_SERVER"):
        sys.exit(f"llama-server is {server.parent.name}, the manifest pins {build}: results would not compare. "
                 f"Delete {server.parent} or set LOCALLLM_LLAMA_SERVER to a {build} build.")
    finished = done_ids(a.results)
    todo = [j for j in manifest["jobs"] if j["id"] not in finished]
    print(f"{len(todo)} of {len(manifest['jobs'])} jobs to run ({len(finished)} done) on llama.cpp {build}")
    from localllm.cli import _model_path
    for i, job in enumerate(todo, 1):
        if a.dry_run:
            model = Path(job["file"]).expanduser() if job.get("file") else runtime.HOME / "models" / catalog.MODELS[job["model"]]["file"]
            print(f"[{i}/{len(todo)}] {job['id']}: {server.name} {' '.join(launch_args(job, model, 'GPU'))}")
            continue
        print(f"[{i}/{len(todo)}] {job['id']} ...", flush=True)
        t0 = time.time()
        try:
            rec = run_job(job, build, server, _model_path)
        except (Exception, SystemExit) as e:  # noqa: BLE001 - record it (a failed download exits) and go on
            rec = {"job": {**job, "status": "failed", "error": f"{type(e).__name__}: {e}",
                           "trace": traceback.format_exc(limit=3)}, "date": time.strftime("%Y-%m-%d")}
            print(f"  failed: {rec['job']['error']}")
        rec["job"]["seconds"] = round(time.time() - t0)
        with open(a.results, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if a.reports and rec["job"]["status"] == "ok":
            a.reports.mkdir(parents=True, exist_ok=True)
            (a.reports / f"{job['id']}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
