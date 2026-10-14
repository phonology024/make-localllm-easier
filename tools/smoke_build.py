"""Smoke test a llama.cpp build before the pin in localllm/runtime.py moves to it (#95).

  python tools/smoke_build.py b11520                 every catalog model that fits this GPU, on build b11520
  python tools/smoke_build.py b11520 --models gemma4-26b-a4b-qat --tolerance 0.1

For each model: download the build (checked against GitHub's digest) and the model, load it with the launch localllm
uses, answer one short chat, measure decode speed, check the spill counter. Then the same on the pinned build, so both
speeds come from this PC on the same day. The bump passes when every model loads, answers, doesn't spill and is no more
than --tolerance slower than on the pinned build. One line per model goes to tools/smoke_results.jsonl.

What to do with the verdict is in docs/upgrading-llama-cpp.md.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from localllm import catalog, runtime  # noqa: E402
import runner  # noqa: E402

RESULTS = Path(__file__).with_name("smoke_results.jsonl")
CHAT = {"messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
        "max_tokens": 16, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}


def verdict(new: dict, old: dict, tolerance: float) -> list[str]:
    """Reasons the new build fails for one model; empty = pass. Each dict: loaded, answer, tok_s, spill_gb."""
    why = []
    if not new.get("loaded"):
        return [f"does not load: {new.get('error', 'unknown error')}"]
    if "paris" not in (new.get("answer") or "").lower():
        why.append(f"wrong or empty answer to the smoke question: {new.get('answer')!r}")
    if (new.get("spill_gb") or 0) > runtime.SPILL_WARN_GB:
        why.append(f"{new['spill_gb']:.1f} GB spilled into system RAM (pinned build: {old.get('spill_gb') or 0:.1f})")
    if new.get("tok_s") and old.get("tok_s") and new["tok_s"] < old["tok_s"] * (1 - tolerance):
        why.append(f"{new['tok_s']} tok/s vs {old['tok_s']} on the pinned build "
                   f"({100 * (new['tok_s'] / old['tok_s'] - 1):+.0f}%, allowed -{100 * tolerance:.0f}%)")
    return why


def server_for(build: str) -> Path:
    os.environ["LOCALLLM_LLAMA_BUILD"] = build
    os.environ.pop("LOCALLLM_LLAMA_SERVER", None)
    return runtime.find_server()


def smoke(server: Path, key: str, dev: dict, model_path) -> dict:
    """Load `key` with localllm's launch, one chat, decode speed, spill. Never raises: errors go in the dict."""
    import urllib.request
    m = catalog.MODELS[key]
    cpu_moe = catalog.cpu_moe_layers(key, dev["total_gb"], runtime.ram_available_gb()) or 0
    args = runtime.server_args(model_path(key), dev["id"], runner.PORT, catalog.MIN_CTX, m["mtp"], cpu_moe=cpu_moe)
    url = f"http://127.0.0.1:{runner.PORT}"
    t0 = time.time()
    proc = subprocess.Popen([str(server), *args], env=runtime.server_env(None, m.get("vk_fix", True)),
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    out = {"loaded": False}
    try:
        runner.wait_up(proc, url)
        out["load_s"] = round(time.time() - t0, 1)
        req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(CHAT).encode(),
                                     headers={"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=300))
        out.update(loaded=True, answer=d["choices"][0]["message"].get("content") or "",
                   tok_s=runner.decode_speed(url), spill_gb=runtime.gpu_spill_gb(proc.pid))
    except Exception as e:  # noqa: BLE001 - a smoke test reports, it doesn't crash
        out["error"] = f"{type(e).__name__}: {e}"
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("build", help="the candidate build, e.g. b11520")
    ap.add_argument("--models", help="comma-separated catalog keys (default: every model that fits this GPU)")
    ap.add_argument("--tolerance", type=float, default=0.05, help="allowed decode slowdown (default 0.05 = 5%%)")
    a = ap.parse_args()
    if runtime.build_number(a.build) is None:
        sys.exit("give the build as bNNNNN")
    from localllm.cli import _model_path
    pinned = server_for(runtime.LLAMA_BUILD)
    dev = runtime.best_device(runtime.devices(pinned))
    if not dev:
        sys.exit("no GPU found")
    keys = a.models.split(",") if a.models else \
        [k for k in catalog.MODELS if catalog.cpu_moe_layers(k, dev["total_gb"], runtime.ram_available_gb()) is not None]
    candidate = server_for(a.build)
    failed = 0
    for key in keys:
        old = smoke(pinned, key, dev, _model_path)
        new = smoke(candidate, key, dev, _model_path)
        why = verdict(new, old, a.tolerance)
        failed += bool(why)
        rec = {"date": time.strftime("%Y-%m-%d"), "gpu": dev["name"], "model": key, "pinned": runtime.LLAMA_BUILD,
               "candidate": a.build, "old": old, "new": new, "pass": not why, "why": why}
        with open(RESULTS, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{key}: {'PASS' if not why else 'FAIL'}  {old.get('tok_s')} -> {new.get('tok_s')} tok/s"
              + "".join(f"\n  - {w}" for w in why))
    print(f"\n{a.build}: {'ready to pin' if not failed else f'{failed} model(s) failed - do not bump'}"
          " (next steps: docs/upgrading-llama-cpp.md)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
