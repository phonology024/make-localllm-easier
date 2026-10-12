"""Fetch an upstream llama.cpp build, list GPUs, and launch llama-server with the settings that measured fastest.

Tuning (each measured on an RX 9070 XT, see README):
  GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1  1.7x decode when Resizable BAR is off (llama.cpp#27097); harmless when on
  -np 1 -kvu                             single user: one slot, unified KV
  --spec-type draft-mtp --spec-draft-n-max 2   models that ship an MTP head (Qwen3.8): +40% decode
  -fit off, --load-mode none             skip the fit dry-run (~0.6 s) and read weights straight into VRAM
"""
from __future__ import annotations

import io
import json
import os
import platform
import re
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path

HOME = Path(os.environ.get("LOCALLLM_HOME", Path.home() / ".localllm"))
EXE = "llama-server.exe" if os.name == "nt" else "llama-server"


def _asset() -> str:
    system = {"Windows": "win", "Linux": "ubuntu", "Darwin": "macos"}[platform.system()]
    arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64"
    if system == "macos":
        return f"bin-macos-{arch}"
    return f"bin-{system}-vulkan-{arch}"


def find_server() -> Path:
    if os.environ.get("LOCALLLM_LLAMA_SERVER"):
        return Path(os.environ["LOCALLLM_LLAMA_SERVER"])
    found = sorted((HOME / "llama.cpp").rglob(EXE)) if (HOME / "llama.cpp").exists() else []
    return found[-1] if found else fetch_server()


# The llama.cpp build every catalog score and speed was measured on. Downloads are checked against these SHA-256s
# (from the release's own asset digests); LOCALLLM_LLAMA_BUILD=bNNNNN or =latest picks another build, which is
# checked against the digest GitHub lists for it.
LLAMA_BUILD = "b11476"
LLAMA_SHA256 = {
    "llama-b11476-bin-win-vulkan-x64.zip": "5c71e7b749697da4a8d46e9ee55486845cbba27c9dfbecb4007f31ba6610d523",
    "llama-b11476-bin-win-vulkan-arm64.zip": "53659ca6e67dc624c62f6df48f204cd89d9d1ff12a642173ba67235be9ff4bf7",
    "llama-b11476-bin-ubuntu-vulkan-x64.tar.gz": "5bb4306d7917f33e81efda02e6f791ae6a82e86bee121227a3ab2b4e8e40427f",
    "llama-b11476-bin-ubuntu-vulkan-arm64.tar.gz": "8dae2f39afee01d3032a101d7c398a6734d6fd9697690d8f93bdce4e3a9efea2",
    "llama-b11476-bin-macos-arm64.tar.gz": "577634a1b8a59e8dabe02ba10de1e610be0574dfaf1cf3020e6dd42853ed877e",
    "llama-b11476-bin-macos-x64.tar.gz": "c2a0dfe7622a99fc3279454814045923e99f1cfdddb8f121c5969c5c675fc073",
}


def _releases(build: str) -> list[dict]:
    api = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
    if build == "latest":
        # "latest" can be a source-only tag (v0.6.0 had no binaries); the binaries ride on the per-build bNNNNN releases
        return json.load(urllib.request.urlopen(api + "?per_page=10"))
    return [json.load(urllib.request.urlopen(f"{api}/tags/{build}"))]


def expected_sha256(asset: dict) -> str | None:
    """Pinned digest for the measured build, else the one GitHub lists for the asset ("sha256:..."), else None."""
    if asset["name"] in LLAMA_SHA256:
        return LLAMA_SHA256[asset["name"]]
    digest = asset.get("digest") or ""
    return digest.split(":", 1)[1] if digest.startswith("sha256:") else None


def check_sha256(data: bytes, name: str, want: str | None) -> None:
    import hashlib
    if want is None:
        raise RuntimeError(f"{name}: no SHA-256 to check it against, refusing to run it. Use the pinned build "
                           f"({LLAMA_BUILD}) or set LOCALLLM_LLAMA_SERVER to a llama-server you built or trust.")
    got = hashlib.sha256(data).hexdigest()
    if got != want.lower():
        raise RuntimeError(f"{name}: SHA-256 is {got}, expected {want}. Nothing was unpacked; try again, and open "
                           "an issue if it keeps failing.")


def fetch_server() -> Path:
    build = os.environ.get("LOCALLLM_LLAMA_BUILD", LLAMA_BUILD)
    want = _asset()
    rel, asset = next((r, a) for r in _releases(build) for a in r["assets"]
                      if want in a["name"] and a["name"].startswith("llama-"))
    dest = HOME / "llama.cpp" / rel["tag_name"]
    print(f"downloading {asset['name']} ({asset['size'] / 2**20:.0f} MB) ...")
    data = urllib.request.urlopen(asset["browser_download_url"]).read()
    try:
        check_sha256(data, asset["name"], expected_sha256(asset))
    except RuntimeError as e:
        raise SystemExit(f"[localllm] {e}") from None
    if asset["name"].endswith(".zip"):
        zipfile.ZipFile(io.BytesIO(data)).extractall(dest)
    else:
        tarfile.open(fileobj=io.BytesIO(data)).extractall(dest, filter="data")
    exe = next(dest.rglob(EXE))
    if os.name != "nt":
        exe.chmod(0o755)
    return exe


def devices(server: Path) -> list[dict]:
    """[{'id': 'Vulkan0', 'name': ..., 'total_gb': .., 'free_gb': ..}] from `llama-server --list-devices`."""
    out = subprocess.run([str(server), "--list-devices"], capture_output=True, text=True, errors="replace")
    devs = []
    for m in re.finditer(r"^\s*(\w+\d+): (.+?) \((\d+) MiB, (\d+) MiB free\)", out.stdout + out.stderr, re.M):
        devs.append({"id": m[1], "name": m[2], "total_gb": int(m[3]) / 1024, "free_gb": int(m[4]) / 1024})
    return devs


def best_device(devs: list[dict]) -> dict | None:
    """Largest-memory device; integrated GPUs report shared RAM, so prefer names that aren't iGPUs."""
    igpu = re.compile(r"UHD|Iris|Radeon\(TM\) Graphics|Radeon Graphics|890M|780M|Apple", re.I)
    pool = [d for d in devs if not igpu.search(d["name"])] or devs
    return max(pool, key=lambda d: d["total_gb"]) if pool else None


def _windows_memory() -> tuple[float, float]:
    """(total_gb, avail_gb) from GlobalMemoryStatusEx."""
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                    ("avail", ctypes.c_ulonglong), ("a", ctypes.c_ulonglong), ("b", ctypes.c_ulonglong),
                    ("c", ctypes.c_ulonglong), ("d", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong)]
    s = MS(); s.len = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.total / 2**30, s.avail / 2**30


def ram_gb() -> float:
    if os.name == "nt":
        return _windows_memory()[0]
    if Path("/proc/meminfo").exists():
        return int(re.search(r"MemTotal:\s+(\d+)", Path("/proc/meminfo").read_text())[1]) / 2**20
    return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2**30


def ram_available_gb() -> float:
    """Best-effort currently-available system RAM in GB; falls back to total RAM when unknown."""
    try:
        if os.name == "nt":
            return _windows_memory()[1]
        if Path("/proc/meminfo").exists():
            m = re.search(r"MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text())
            if m:
                return int(m[1]) / 2**20
    except OSError:
        pass
    return ram_gb()


# Host-RAM caches llama-server keeps by default (8 GiB prompt cache + 32 checkpoints per slot) can grow to many GB
# (llama.cpp #21690: 0.7 -> 18 GB). One user doesn't need that much: size them from the installed RAM.
#   installed RAM (GB) -> (--cache-ram MiB, --ctx-checkpoints)
LOW_RAM_PROFILE = [(16, (512, 2)), (32, (1024, 4)), (64, (2048, 8))]


def ram_profile(ram_total_gb: float) -> tuple[int, int] | None:
    """(cache-ram MiB, ctx-checkpoints) for this PC, or None to keep llama.cpp's defaults on big-RAM machines."""
    for limit, prof in LOW_RAM_PROFILE:
        if ram_total_gb <= limit + 0.5:
            return prof
    return None


def server_args(model: Path, device: str | None, port: int, ctx: int, mtp: bool,
                ram_total_gb: float | None = None, cpu_moe: int = 0) -> list[str]:
    args = ["-m", str(model), "--host", "127.0.0.1", "--port", str(port), "-c", str(ctx), "-ngl", "999",
            "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-np", "1", "-kvu", "-fit", "off"]
    # read weights straight into place. Measured on gemma-4 with 13 layers' experts in RAM: same 36 tok/s as mmap but
    # 6.1 GB working set instead of 13.5 GB (mmap keeps the whole file resident)
    args += ["--load-mode", "none"] + (["--n-cpu-moe", str(cpu_moe)] if cpu_moe else [])
    prof = ram_profile(ram_total_gb if ram_total_gb is not None else ram_gb())
    if prof:
        args += ["--cache-ram", str(prof[0]), "--ctx-checkpoints", str(prof[1])]
    if device:
        args += ["-dev", device]
    if mtp:
        args += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(mtp if mtp is not True and mtp > 1 else 2)]
    return args


def gpu_spill_gb(pid: int) -> float | None:
    """Windows only: GB of this process's GPU allocations that live in *shared* (system) memory.

    When a model doesn't really fit, WDDM quietly backs part of it with system RAM and decode slows down several
    times; the "GPU Process Memory\\Shared Usage" counter shows it. None when the counter isn't available.
    """
    if os.name != "nt":
        return None
    ps = (f"(Get-Counter '\\GPU Process Memory(pid_{pid}_*)\\Shared Usage' -ErrorAction SilentlyContinue)"
          ".CounterSamples | Measure-Object CookedValue -Sum | ForEach-Object Sum")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=20)
        return float(out.stdout.strip() or 0) / 2**30
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def process_memory(pid: int) -> dict:
    """Best effort: {"ram_gb", "vram_gb", "spill_gb"} of one process, None for what this OS doesn't tell us.
    Windows: working set + the GPU Process Memory counters (dedicated = VRAM, shared = spilled into system RAM).
    Linux: VmRSS only."""
    out = {"ram_gb": None, "vram_gb": None, "spill_gb": None}
    if os.name == "nt":
        ps = (f"$p = Get-Process -Id {pid} -ErrorAction SilentlyContinue; "
              f"$c = (Get-Counter '\\GPU Process Memory(pid_{pid}_*)\\Dedicated Usage',"
              f"'\\GPU Process Memory(pid_{pid}_*)\\Shared Usage' -ErrorAction SilentlyContinue).CounterSamples; "
              "$d = ($c | Where-Object Path -like '*dedicated*' | Measure-Object CookedValue -Sum).Sum; "
              "$s = ($c | Where-Object Path -like '*shared*' | Measure-Object CookedValue -Sum).Sum; "
              "\"$($p.WorkingSet64) $d $s\"")
        try:
            r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                               timeout=20)
            vals = (r.stdout.split() + ["", "", ""])[:3]
            for k, v in zip(("ram_gb", "vram_gb", "spill_gb"), vals):
                out[k] = float(v) / 2**30 if v else None
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
        return out
    status = Path(f"/proc/{pid}/status")
    try:
        m = re.search(r"VmRSS:\s+(\d+)", status.read_text())
        out["ram_gb"] = int(m[1]) / 2**20 if m else None
    except OSError:
        pass
    return out


SPILL_WARN_GB = 1.5   # Vulkan keeps ~0.5-0.9 GB of host-visible buffers here even when the model fits (measured)


def server_env(tuned: dict | None = None, vk_fix: bool = True) -> dict:
    """Environment for llama-server. With a `localllm tune` result: exactly the settings that measured faster on this
    PC. Without one: the small-BAR fix when the catalog says it helps this model (measured on RX 9070 XT: Qwen3.8 1.64x
    faster, gemma-4 7-9% slower). llama.cpp treats any value of the variable as "on", so "off" means unset."""
    env = {**os.environ}
    if tuned is not None:
        env.pop("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM", None)
        return {**env, **tuned.get("env", {})}
    if "GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM" in os.environ:   # the user decided
        return env
    if vk_fix:
        env["GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM"] = "1"
    return env
