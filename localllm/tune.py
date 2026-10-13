"""Per-machine tuning, done once and cached (~/.localllm/tune.json): measure instead of guessing.

  small-BAR detection   AMD/Intel cards with Resizable BAR off expose a tiny DEVICE_LOCAL|HOST_VISIBLE heap; llama.cpp's
                        Vulkan buffers then land in system RAM (1.7x slower decode on RX 9070 XT, 2.7x on RX 7900 XTX,
                        llama.cpp#27097). Set GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1 there, confirmed by an A/B run.
  MTP draft length      keep drafting only where it measures > 1.1x (it is slower on Apple Metal).
  CUDA graphs           GGML_CUDA_GRAPH_OPT=1 on single-GPU NVIDIA (+17-42% reported on RTX 4090/5090), A/B confirmed.
  batch / ubatch        llama-bench sweep for prompt processing.
  energy                GPU board watts and joules per token for the kept settings (power.py); --prefer-energy keeps
                        a setting that saves > 15% energy at < 5% speed loss.
Everything here only *proposes* settings; `measure()` keeps a change only if it is faster on this PC.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from .runtime import HOME

CACHE = HOME / "tune.json"
SMALL_BAR_GB = 1.0
KEEP_IF_FASTER = 1.10


def vulkan_heaps(vulkaninfo_text: str, device_name: str) -> list[dict]:
    """Parse `vulkaninfo` text: memory heaps (size GiB, device_local) and which heaps have a DEVICE_LOCAL|HOST_VISIBLE type."""
    m = re.search(r"deviceName\s*=\s*" + re.escape(device_name), vulkaninfo_text)
    if not m:
        return []
    block = vulkaninfo_text[m.start():]
    nxt = re.search(r"\nGPU\d+:", block)          # the next device's section starts here
    block = block[: nxt.start()] if nxt else block
    heaps = []
    for m in re.finditer(r"memoryHeaps\[(\d+)\]:\s*\n\s*size\s*=\s*(\d+).*?\n(?:.*\n){0,3}?\s*flags:.*?\n((?:\s+MEMORY_HEAP_\w+\n)*)",
                         block):
        heaps.append({"index": int(m[1]), "gb": int(m[2]) / 2**30, "device_local": "DEVICE_LOCAL" in m[3]})
    hv_heaps = set()
    for m in re.finditer(r"memoryTypes\[\d+\]:\s*\n\s*heapIndex\s*=\s*(\d+)\s*\n\s*propertyFlags.*?\n((?:\s+MEMORY_PROPERTY_\w+\n)*)",
                         block):
        if "DEVICE_LOCAL" in m[2] and "HOST_VISIBLE" in m[2]:
            hv_heaps.add(int(m[1]))
    for h in heaps:
        h["host_visible_device_local"] = h["index"] in hv_heaps
    return heaps


def small_bar(heaps: list[dict]) -> bool | None:
    """True when the only host-visible VRAM is a small window (Resizable BAR off); None when unknown."""
    hv = [h for h in heaps if h.get("host_visible_device_local")]
    if not hv:
        return None
    return max(h["gb"] for h in hv) < SMALL_BAR_GB


def detect_small_bar(device_name: str) -> bool | None:
    exe = shutil.which("vulkaninfo") or shutil.which("vulkaninfoSDK")
    if not exe:
        return None
    try:
        out = subprocess.run([exe], capture_output=True, text=True, errors="replace", timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    return small_bar(vulkan_heaps(out, device_name))


def keep(base_tok_s: float, new_tok_s: float) -> bool:
    return base_tok_s > 0 and new_tok_s / base_tok_s >= KEEP_IF_FASTER


def best_mtp(results: dict[int, float]) -> int:
    """results: {draft_len: tok/s} with 0 = no drafting. Returns the draft length to use (0 = off)."""
    base = results.get(0, 0.0)
    best = max(results, key=results.get)
    return best if best and keep(base, results[best]) else 0


def bench_tg(llama_bench: Path, model: Path, env: dict, extra: list[str] | None = None, n: int = 128) -> float:
    """Decode tok/s from llama-bench (tg128); 0.0 on failure."""
    cmd = [str(llama_bench), "-m", str(model), "-ngl", "99", "-fa", "1", "-p", "0", "-n", str(n), "-r", "2", "-o", "json",
           *(extra or [])]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900).stdout
        rows = json.loads(out[out.find("["):])
        return float(rows[-1]["avg_ts"])
    except (OSError, ValueError, KeyError, IndexError, subprocess.TimeoutExpired):
        return 0.0


def load() -> dict:
    try:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save(key: str, value: dict) -> None:
    data = load()
    data[key] = value
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(data, indent=1), encoding="utf-8")


def machine_key(gpu_name: str, model_file: str, server: Path) -> str:
    return f"{gpu_name}|{model_file}|{server.parent.name}"


def server_decode(server: Path, args: list[str], env: dict, port: int = 18234) -> float:
    """Mean decode tok/s of a real llama-server run with `args` (two short chats, the first one warms up)."""
    return server_stats(server, args, env, port)["tok_s"]


def server_stats(server: Path, args: list[str], env: dict, port: int = 18234, power=None) -> dict:
    """{"tok_s", "watts", "j_per_token"} of a real llama-server run with `args`: two short chats after a warm-up.
    power: a reader from power.reader(); without one (or when it reads nothing) watts and j_per_token are None."""
    import time
    import urllib.request
    from . import power as pw
    proc = subprocess.Popen([str(server), *args, "--port", str(port)], env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    out = {"tok_s": 0.0, "watts": None, "j_per_token": None}
    try:
        for _ in range(900):
            if proc.poll() is not None:
                return out
            try:
                if b'"ok"' in urllib.request.urlopen(url + "/health", timeout=2).read():
                    break
            except OSError:
                time.sleep(0.3)
        speeds, joules, secs, tokens = [], 0.0, 0.0, 0
        for prompt in ("Warm up.", "Write a short paragraph about the ocean and the moon.",
                       "Explain in a few sentences how a bicycle stays upright."):
            body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 160, "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False}}
            req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with pw.Sampler(power or (lambda: None)) as sampler:
                timings = json.load(urllib.request.urlopen(req, timeout=600))["timings"]
            speeds.append(timings["predicted_per_second"])
            if prompt != "Warm up." and sampler.watts:
                joules += sampler.watts * sampler.seconds
                secs += sampler.seconds
                tokens += timings.get("predicted_n", 0)
        out["tok_s"] = sum(speeds[1:]) / len(speeds[1:])
        if secs and tokens:
            out["watts"], out["j_per_token"] = round(joules / secs, 1), round(joules / tokens, 3)
        return out
    except (OSError, ValueError, KeyError):
        return out
    finally:
        proc.terminate()
        proc.wait(30)


def run(server: Path, model: Path, gpu_name: str, base_args: list[str], mtp_capable: bool, log=print,
        prefer_energy: bool = False) -> dict:
    """Measure the candidate settings on this PC; keep only clear wins. Returns and caches the chosen settings.
    Energy (GPU board W and J/token, #42) is measured along where this vendor/OS has a power reader. prefer_energy:
    also keep a setting that saves > 15% energy per token at < 5% speed loss."""
    import os
    from . import power
    read = power.reader(gpu_name)
    env0 = {**os.environ}
    env0.pop("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM", None)
    chosen = {"env": {}, "mtp": 0}
    best = server_stats(server, base_args, env0, power=read)
    base = best["tok_s"]
    log(f"baseline: {base:.1f} tok/s" + _energy(best))
    is_nvidia = re.search(r"nvidia|geforce|rtx|gtx", gpu_name, re.I) is not None
    trials = []
    if not is_nvidia and detect_small_bar(gpu_name) is not False:
        trials.append(("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM", "1"))
    if is_nvidia:
        trials.append(("GGML_CUDA_GRAPH_OPT", "1"))
    for k, v in trials:
        st = server_stats(server, base_args, {**env0, **chosen["env"], k: v}, power=read)
        t = st["tok_s"]
        log((f"{k}={v}: {t:.1f} tok/s ({t / base:.2f}x)" if base else f"{k}={v}: {t:.1f} tok/s") + _energy(st))
        if keep(base, t) or (prefer_energy and power.prefer_energy(best, st)):
            chosen["env"][k] = v
            base, best = t, st
    if mtp_capable:
        res = {0: best}
        for n in (2, 3):
            res[n] = server_stats(server, base_args + ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(n)],
                                  {**env0, **chosen["env"]}, power=read)
            log(f"MTP draft {n}: {res[n]['tok_s']:.1f} tok/s" + _energy(res[n]))
        chosen["mtp"] = best_mtp({n: r["tok_s"] for n, r in res.items()})
        if prefer_energy:
            saver = min((n for n in res if power.prefer_energy(res[chosen["mtp"]], res[n])),
                        key=lambda n: res[n]["j_per_token"], default=None)
            chosen["mtp"] = chosen["mtp"] if saver is None else saver
        best = res[chosen["mtp"]]
        base = best["tok_s"]
    chosen["tok_s"] = round(base, 1)
    if best.get("j_per_token"):
        chosen["watts"], chosen["j_per_token"] = best["watts"], best["j_per_token"]
    save(machine_key(gpu_name, model.name, server), chosen)
    return chosen


def _energy(st: dict) -> str:
    return f", {st['watts']:.0f} W, {st['j_per_token']:.2f} J/token" if st.get("j_per_token") else ""
