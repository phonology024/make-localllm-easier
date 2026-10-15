"""Time to first answer on a clean machine profile (#116): an empty LOCALLLM_HOME, so llama.cpp and the starter model
are downloaded, then `localllm serve` starts and the first chat answer arrives.

  python tools/ttfa.py [--port 18555] [--keep]

Reports the time to each milestone (server answering /health, first answer) from the command start, and kills the
server right after the first answer (the best model's background download is not waited for). The temporary home is
deleted unless --keep. One JSON line in tools/ttfa_results.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def get(url: str, timeout: float = 2):
    try:
        return urllib.request.urlopen(url, timeout=timeout).read()
    except OSError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=18555)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    home = Path(tempfile.mkdtemp(prefix="localllm-clean-"))
    env = {**os.environ, "LOCALLLM_HOME": str(home), "LOCALLLM_YES": "1", "PYTHONIOENCODING": "utf-8"}
    env.pop("LOCALLLM_MODELS", None)
    log = open(ROOT / "tools" / "ttfa_server.log", "w", encoding="utf-8")
    t0 = time.time()
    proc = subprocess.Popen([sys.executable, "-m", "localllm", "serve", "--port", str(a.port)], cwd=ROOT, env=env,
                            stdout=log, stderr=subprocess.STDOUT)
    url, marks = f"http://127.0.0.1:{a.port}", {}
    try:
        while "first_answer_s" not in marks and time.time() - t0 < 1800:
            if proc.poll() is not None:
                raise SystemExit(f"localllm exited ({proc.returncode}); see tools/ttfa_server.log")
            if "health_s" not in marks and get(url + "/health"):
                marks["health_s"] = round(time.time() - t0, 1)
            if "health_s" in marks:
                # what the chat page sends: no thinking switch, so the model may reason first (hidden), then answer
                body = {"messages": [{"role": "user", "content": "Say hello in one short sentence."}], "max_tokens": 1024}
                req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                try:
                    d = json.load(urllib.request.urlopen(req, timeout=300))
                    msg = d["choices"][0]["message"]
                    if not (msg.get("content") or "").strip():
                        raise SystemExit(f"empty answer (finish_reason {d['choices'][0].get('finish_reason')}): not a "
                                         "first answer a person would see")
                    marks["first_answer_s"] = round(time.time() - t0, 1)
                    marks["answer"] = msg["content"].strip()[:80]
                    marks["reasoning_chars"] = len(msg.get("reasoning_content") or "")
                    marks["model"] = Path(d.get("model") or "").name
                except OSError:
                    pass
            time.sleep(0.5)
        st = json.loads(get(url + "/localllm/status", 5) or b"{}")
        marks["download_line"] = st.get("download")
    finally:
        if os.name == "nt":       # the whole tree first: once the python parent is gone its llama-server is orphaned
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
    sizes = {p.name: round(p.stat().st_size / 2**30, 2) for p in home.rglob("*") if p.is_file() and p.stat().st_size > 2**26}
    rec = {"date": time.strftime("%Y-%m-%d %H:%M"), **marks, "downloaded_gb": sizes, "note": a.note}
    with open(ROOT / "tools" / "ttfa_results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    if not a.keep:
        time.sleep(3)
        shutil.rmtree(home, ignore_errors=True)


if __name__ == "__main__":
    main()
