"""GPU board power while answering, for watts and joules per token (#42). Best effort: None where we can't read it.

  NVIDIA   nvidia-smi --query-gpu=power.draw (any OS with the driver)
  AMD      Windows: the driver's ADL library (atiadlxx.dll, PMLog sensors: total board power, else ASIC power)
           Linux: hwmon power1_average / power1_input of the amdgpu card (microwatts)
  Apple    powermetrics needs sudo, so not read; Intel: not yet

A `Sampler` reads the power every 0.2 s in a thread while a generation runs; joules per token = mean watts x seconds /
tokens. Only the GPU board is counted (not CPU, RAM or the PSU's losses), which is what the readers give on every vendor.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

INTERVAL_S = 0.2
# ADL PMLog sensor ids (adl_defines.h, ADLSensorType). Board total power is reported by RDNA3/4 drivers; older ones
# only give ASIC (chip) power.
ADL_BOARD_POWER, ADL_ASIC_POWER = 73, 23


def nvidia_reader(index: int = 0):
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None

    def read() -> float | None:
        try:
            out = subprocess.run([exe, "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-i", str(index)],
                                 capture_output=True, text=True, timeout=5).stdout
            return float(out.strip().splitlines()[0])
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            return None
    return read


def amd_sysfs_reader(root: Path = Path("/sys/class/drm")):
    for f in sorted(root.glob("card*/device/hwmon/hwmon*/power1_average")) + \
            sorted(root.glob("card*/device/hwmon/hwmon*/power1_input")):
        def read(f=f) -> float | None:
            try:
                return int(f.read_text().strip()) / 1e6
            except (OSError, ValueError):
                return None
        if read() is not None:
            return read
    return None


def amd_adl_reader():
    if sys.platform != "win32":
        return None
    import ctypes
    try:
        dll = ctypes.WinDLL("atiadlxx.dll")
    except OSError:
        return None
    keep = []

    @ctypes.WINFUNCTYPE(ctypes.c_void_p, ctypes.c_int)
    def alloc(n):
        b = ctypes.create_string_buffer(n)
        keep.append(b)
        return ctypes.addressof(b)

    class Sensor(ctypes.Structure):
        _fields_ = [("supported", ctypes.c_int), ("value", ctypes.c_int)]

    class PMLog(ctypes.Structure):
        _fields_ = [("size", ctypes.c_int), ("sensors", Sensor * 256)]

    ctx, n = ctypes.c_void_p(), ctypes.c_int()
    if dll.ADL2_Main_Control_Create(alloc, 1, ctypes.byref(ctx)) != 0:
        return None
    dll.ADL2_Adapter_NumberOfAdapters_Get(ctx, ctypes.byref(n))
    for i in range(n.value):
        log = PMLog(size=ctypes.sizeof(PMLog))
        if dll.ADL2_New_QueryPMLogData_Get(ctx, i, ctypes.byref(log)) != 0:
            continue
        sensor = next((s for s in (ADL_BOARD_POWER, ADL_ASIC_POWER) if log.sensors[s].supported), None)
        if sensor is None:
            continue

        def read(i=i, sensor=sensor) -> float | None:
            out = PMLog(size=ctypes.sizeof(PMLog))
            if dll.ADL2_New_QueryPMLogData_Get(ctx, i, ctypes.byref(out)) != 0:
                return None
            return float(out.sensors[sensor].value)
        read.keep = (dll, alloc, keep, ctx)          # the callback and context must outlive the reader
        return read
    return None


def reader(gpu_name: str):
    """A function returning the GPU's board power in watts, or None when this vendor/OS has no reader."""
    if re.search(r"nvidia|geforce|rtx|gtx|quadro|tesla", gpu_name, re.I):
        return nvidia_reader()
    if re.search(r"amd|radeon|\brx\b", gpu_name, re.I):
        return amd_adl_reader() if sys.platform == "win32" else amd_sysfs_reader()
    return None


class Sampler:
    """with Sampler(read) as s: ...generate...; then s.watts (mean) and s.seconds."""

    def __init__(self, read, interval: float = INTERVAL_S):
        self.read, self.interval, self.samples = read, interval, []
        self._stop = threading.Event()

    def _loop(self):
        while not self._stop.is_set():
            w = self.read()
            if w is not None and w > 0:
                self.samples.append(w)
            self._stop.wait(self.interval)

    def __enter__(self):
        self.t0 = time.time()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(5)
        self.seconds = time.time() - self.t0
        self.watts = sum(self.samples) / len(self.samples) if self.samples else None
        return False


def joules_per_token(watts: float | None, seconds: float, tokens: int) -> float | None:
    return round(watts * seconds / tokens, 3) if watts and tokens else None


def prefer_energy(base: dict, cand: dict, save: float = 0.15, max_slower: float = 0.05) -> bool:
    """`tune --prefer-energy`: take a setting that saves > 15% energy per token at < 5% speed loss."""
    if not (base.get("j_per_token") and cand.get("j_per_token") and base.get("tok_s") and cand.get("tok_s")):
        return False
    return cand["j_per_token"] < base["j_per_token"] * (1 - save) and cand["tok_s"] >= base["tok_s"] * (1 - max_slower)
