"""`localllm doctor --fix` (#111): find the first-run failures people uninstall over, say what to do in plain words,
and apply the safe fixes when you agree.

  no-gpu         no GPU device: the Vulkan loader is missing, or the driver doesn't expose the card -> CPU only, slow
  small-bar      Resizable BAR is off (AMD/Intel): llama.cpp's Vulkan buffers land in system RAM -> `localllm tune`
  vram-busy      other apps hold the VRAM the model needs, so it would spill -> name them; or a model that fits
  ram-low        not enough free RAM for the model's working set -> lean profile, smaller model
  port-busy      something already listens on the port -> name it; or another port
  garbled        a running localllm answers with chat-template tokens or loops -> llama.cpp build / template
A fix that is applied writes ~/.localllm/defaults.json (model, port, profile), which `localllm` reads when the flag is
not given on the command line; `localllm doctor --fix --reset` removes it. Nothing else on the PC is changed.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import catalog, runtime

DEFAULTS = runtime.HOME / "defaults.json"
SETTABLE = {"model": None, "port": 8080, "profile": "lean"}     # what a fix may change, with the parser defaults
GARBLED = re.compile(r"<\|im_start\|>|<\|im_end\|>|<start_of_turn>|<end_of_turn>|<\|eot_id\|>|<\|start_header_id\|>|"
                     r"\[INST\]|<\|channel\|>|<\|endoftext\|>")


@dataclass
class Problem:
    id: str
    what: str                      # one plain sentence: what is wrong
    fix: str                       # what to do about it
    apply: dict = field(default_factory=dict)   # safe fix: defaults.json changes, applied with consent
    run: list | None = None        # or a localllm command to run with consent


def load_defaults() -> dict:
    try:
        d = json.loads(DEFAULTS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in d.items() if k in SETTABLE}


def save_defaults(changes: dict) -> dict:
    d = {**load_defaults(), **{k: v for k, v in changes.items() if k in SETTABLE}}
    DEFAULTS.parent.mkdir(parents=True, exist_ok=True)
    DEFAULTS.write_text(json.dumps(d, indent=1), encoding="utf-8")
    return d


def apply_defaults(args, given: set[str]) -> None:
    """Fill flags the user didn't type from defaults.json (the command line always wins)."""
    for k, v in load_defaults().items():
        if k not in given and hasattr(args, k) and getattr(args, k) == SETTABLE[k]:
            setattr(args, k, v)


# --- checks: each takes what it needs and returns a Problem or None ---------------------------------------------------

def vulkan_loader_present() -> bool:
    if os.name == "nt":
        return (Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "vulkan-1.dll").exists()
    if sys.platform == "darwin":
        return True                                     # llama.cpp uses Metal there
    try:
        out = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=10).stdout
        return "libvulkan.so.1" in out
    except (OSError, subprocess.TimeoutExpired):
        return any(Path(d, "libvulkan.so.1").exists() for d in ("/usr/lib", "/usr/lib64", "/usr/lib/x86_64-linux-gnu"))


def check_gpu(devs: list[dict], loader: bool) -> Problem | None:
    if devs:
        return None
    small = min(catalog.MODELS, key=lambda k: catalog.MODELS[k]["gb"])
    if not loader:
        return Problem("no-gpu", "No GPU was found and the Vulkan runtime is missing, so models would run on the CPU "
                       "only (a few tokens per second).",
                       "Install or update your graphics driver (AMD Adrenalin, NVIDIA Game Ready/Studio, Intel Arc); "
                       "it brings the Vulkan runtime. Then run `localllm doctor` again.")
    return Problem("no-gpu", "llama.cpp sees no GPU, although the Vulkan runtime is installed: the driver may be too "
                   "old, or this card has no Vulkan support. Models would run on the CPU only (a few tokens per second).",
                   "Update the graphics driver and run `localllm doctor` again. Until then, a small model keeps the "
                   f"CPU fallback usable: {small} ({catalog.MODELS[small]['gb']} GB).", apply={"model": small})


def check_small_bar(dev: dict | None, small_bar: bool | None, tuned: bool) -> Problem | None:
    if not dev or not small_bar or tuned or re.search(r"nvidia|geforce|rtx|gtx", dev["name"], re.I):
        return None
    return Problem("small-bar", f"Resizable BAR looks off on the {dev['name']}: llama.cpp's Vulkan buffers then sit in "
                   "system RAM, which makes some models 1.6-2.7x slower.",
                   "Turn on 'Resizable BAR' / 'Smart Access Memory' (and 'Above 4G decoding') in the BIOS. Without "
                   "that, `localllm tune` measures whether the Vulkan workaround helps your model and keeps it if it "
                   "does (a few minutes).", run=["tune"])


def check_vram(dev: dict | None, key: str | None, others: list[tuple[str, float]]) -> Problem | None:
    """others: (process name, GB of dedicated VRAM) of other apps, largest first."""
    if not dev or not key:
        return None
    m = catalog.MODELS[key]
    need = m["gb"] + m["kv_kb_per_token"] * catalog.MIN_CTX / 2**20 + m["fixed_cache_gb"] + 0.8
    free = dev.get("free_gb", dev["total_gb"])
    if free >= need:
        return None
    names = ", ".join(f"{n} ({gb:.1f} GB)" for n, gb in others[:4]) or "other apps"
    fit = catalog.pick(free, None, runtime.ram_available_gb(), candidates=[k for k in catalog.MODELS
                                                                           if catalog.MODELS[k]["gb"] < m["gb"]])
    fix = (f"Close what holds VRAM - {names} - and the model fits again. ")
    if fit:
        fix += f"Or use a smaller model that fits the {free:.1f} GB free now: {fit} ({catalog.MODELS[fit]['gb']} GB)."
    return Problem("vram-busy", f"{key} needs about {need:.1f} GB of VRAM but only {free:.1f} GB of "
                   f"{dev['total_gb']:.1f} GB is free, so part of it would spill into system RAM and answer slowly.",
                   fix, apply={"model": fit} if fit else {})


def check_ram(key: str | None, ram_free: float, need_gb: float) -> Problem | None:
    if not key or ram_free >= need_gb:
        return None
    return Problem("ram-low", f"Only {ram_free:.1f} GB of system RAM is free; {key} needs about {need_gb:.1f} GB next "
                   "to the GPU (caches, CPU-mapped weights), so Windows would page to disk.",
                   "Close memory-heavy apps (browsers with many tabs, games), or keep the lean profile, which uses "
                   "the least RAM.", apply={"profile": "lean"})


def port_owner(port: int) -> str | None:
    """Name of the process listening on 127.0.0.1:port, best effort."""
    try:
        if os.name == "nt":
            ps = (f"$c = Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue | "
                  "Select-Object -First 1; if ($c) { (Get-Process -Id $c.OwningProcess).ProcessName }")
            out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                                 timeout=20).stdout.strip()
            return out or None
        if shutil.which("lsof"):
            out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"], capture_output=True, text=True,
                                 timeout=10).stdout.splitlines()
            return out[1].split()[0] if len(out) > 1 else None
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def port_free(port: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def check_port(port: int, free: bool, owner: str | None, is_localllm: bool) -> Problem | None:
    if free or is_localllm:
        return None
    alt = next(p for p in range(port + 1, port + 200) if port_free(p))
    who = f"{owner} is" if owner else "another program is"
    return Problem("port-busy", f"Port {port} is taken: {who} already listening there, so localllm can't start its "
                   "server on it.", f"Use another port: localllm --port {alt} (saved as your default if you agree). "
                   "Apps you connected with `localllm connect` need the new address too.", apply={"port": alt})


def check_garbled(reply: str | None, key: str | None) -> Problem | None:
    if not reply:
        return None
    looped = re.search(r"(\b\w+\b)(?:\W+\1\b){7,}", reply)
    if not GARBLED.search(reply) and not looped:
        return None
    return Problem("garbled", f"The running model{' ' + key if key else ''} answered with "
                   f"{'raw chat-template tokens' if GARBLED.search(reply) else 'the same word over and over'}: "
                   "its chat template isn't being applied.",
                   "Restart localllm (it starts llama-server with the model's own template). If you run llama-server "
                   "yourself, add --jinja, and use the llama.cpp build `localllm doctor` names. If it persists, "
                   "`localllm report` and open an issue.")


def ask_running(port: int) -> str | None:
    """A short answer from a localllm already running on `port`, or None."""
    import urllib.request
    body = {"messages": [{"role": "user", "content": "Reply with the word OK."}], "max_tokens": 24, "temperature": 0}
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=60))
        return d["choices"][0]["message"].get("content") or ""
    except (OSError, ValueError, KeyError, IndexError):
        return None


def vram_users(exclude_pid: int | None = None) -> list[tuple[str, float]]:
    """Other processes' dedicated VRAM on Windows (GPU Process Memory counters), largest first; [] elsewhere."""
    if os.name != "nt":
        return []
    ps = ("(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples | "
          "Where-Object CookedValue -gt 200MB | ForEach-Object { $i = [int]($_.InstanceName -replace '^pid_(\\d+)_.*','$1'); "
          "$p = Get-Process -Id $i -ErrorAction SilentlyContinue; \"$i`t$($p.ProcessName)`t$($_.CookedValue)\" }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                             timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = {}
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[0].isdigit() and int(parts[0]) != exclude_pid:
            rows[parts[1] or parts[0]] = rows.get(parts[1] or parts[0], 0) + float(parts[2]) / 2**30
    return sorted(rows.items(), key=lambda kv: -kv[1])


def report(problems: list[Problem]) -> str:
    if not problems:
        return "No first-run problems found."
    out = []
    for i, p in enumerate(problems, 1):
        out.append(f"{i}. [{p.id}] {p.what}\n   fix: {p.fix}")
    return "\n".join(out)
