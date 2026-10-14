"""Model swap time with and without the page-cache prefetch (#63, step 1 of #32).

  python tools/swap_bench.py                          # qwen3.8-27b-q3 <-> gemma4-26b-a4b-qat, 6 swaps each way
  python tools/swap_bench.py --swaps 6 --pause 20 --json tools/swap_results.jsonl

Uses localllm's own Pool and launcher, the same as `localllm serve --models`: one model on the GPU at a time, a swap stops
the loaded llama-server and starts the other one. Each round asks one short question (named model, so the router can't
decide), waits --pause seconds (the time a person reads the answer; prefetch runs then), and records the swap time.
Runs "prefetch off" (files=None) first, then "prefetch on" (files passed to Pool, as serve does), each with a fresh Pool.
Windows can't drop the page cache without admin rights, so the first swap of each mode may be warm from earlier runs;
it is reported but left out of the medians.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from localllm import catalog, cli, pool, runtime  # noqa: E402


def ask(url: str) -> None:
    body = {"messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 4,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=600).read()


def disk_kind(path: Path) -> str:
    """Media type of the disk holding `path` (Windows: SSD / HDD via Get-PhysicalDisk), best effort."""
    if platform.system() != "Windows":
        return "unknown"
    import subprocess
    ps = (f"(Get-Partition -DriveLetter {str(path.resolve())[0]} | Get-Disk | Get-PhysicalDisk)."
          "MediaType + ' ' + (Get-Partition -DriveLetter " + str(path.resolve())[0] + " | Get-Disk).BusType")
    try:
        return subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                              timeout=30).stdout.strip() or "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def run_mode(keys: list[str], prefetch_on: bool, swaps: int, pause: float) -> dict:
    server, _devs, dev, _ram = cli._machine()
    ram_free = runtime.ram_available_gb()
    ctxs = {k: catalog.context_for(k, dev["total_gb"], ram_free, "lean") for k in keys}
    files = {k: cli._model_path(k) for k in keys} if prefetch_on else None
    p = pool.Pool(keys, lambda k: cli._launch(k, server, dev, ctxs[k], ram_free, False, "lean"), dev["total_gb"],
                  ram_free, keys[0], False, files=files, idle_unload_s=0)
    try:
        times = []
        for i in range(swaps * 2):
            key = keys[(i + 1) % 2]
            with p.use({"model": key, "messages": [{"role": "user", "content": "Say OK."}]}) as (url, label):
                ask(url)
            times.append({"to": key, "s": p.swaps[-1]})
            print(f"  prefetch {'on ' if prefetch_on else 'off'} -> {key:20} {p.swaps[-1]:5.1f} s  {label}", flush=True)
            time.sleep(pause)
    finally:
        p.close()
    rest = [t["s"] for t in times[1:]]
    return {"prefetch": prefetch_on, "swaps": times, "median_s": round(statistics.median(rest), 2),
            "max_s": max(rest), "prefetched": sorted(set(p.prefetched))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", default="qwen3.8-27b-q3,gemma4-26b-a4b-qat")
    ap.add_argument("--swaps", type=int, default=6, help="swaps each way per mode")
    ap.add_argument("--pause", type=float, default=20.0, help="seconds between messages")
    ap.add_argument("--json", type=Path, default=ROOT / "tools" / "swap_results.jsonl")
    a = ap.parse_args()
    keys = a.models.split(",")
    rows = [run_mode(keys, on, a.swaps, a.pause) for on in (False, True)]
    model = cli._model_path(keys[0])
    rec = {"date": time.strftime("%Y-%m-%d"), "llama_cpp": runtime.build_of(runtime.find_server()),
           "ram_gb": round(runtime.ram_gb(), 1), "disk": disk_kind(model), "models": keys,
           "model_gb": {k: round(cli._model_path(k).stat().st_size / 2**30, 1) for k in keys}, "modes": rows}
    with open(a.json, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    for r in rows:
        print(f"prefetch {'on ' if r['prefetch'] else 'off'}: median {r['median_s']} s, max {r['max_s']} s "
              f"(first swap left out)")


if __name__ == "__main__":
    main()
