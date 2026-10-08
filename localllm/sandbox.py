"""Run model-written code safely: inside a throwaway Docker container, never on the host.

The container has no network, a read-only filesystem (only the in-memory /tmp and /dev/shm are writable), no Linux
capabilities, no privilege escalation, a memory/CPU/process cap, a non-root user, and a per-program timeout. Programs
are passed in through a read-only mount and results come back as JSON on stdout. Between two programs every process
the first one left behind is killed and /tmp and /dev/shm are wiped, so a fork bomb or a full /tmp can't spill into
the next one. If Docker (with Linux containers) isn't available the code is NOT run - there is no unsafe fallback.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

IMAGE = "localllm-sandbox:1"
DOCKERFILE = """FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==2.1.3 && useradd -m -u 10001 runner
USER runner
WORKDIR /tmp
"""
# PID 1 of the container. Per program: ok, rc (exit code, -9 = SIGKILL), err (stderr tail, or 'timeout'), oom / pids
# (it hit the --memory / --pids-limit cap: the container cgroup's own event counters went up) and t (seconds).
RUNNER = r'''
import ctypes, json, os, shutil, signal, subprocess, sys, threading, time
libc = ctypes.CDLL(None)
libc.prctl(4, 0, 0, 0, 0)                  # PR_SET_DUMPABLE 0: programs can't ptrace us or open /proc/1/fd/*

def events(key, *files):                   # cgroup v2 file, then v1
    for f in files:
        try:
            return int(dict(l.split() for l in open(f))[key])
        except (OSError, KeyError, ValueError):
            pass
    return 0

def limits():
    return (events("oom_kill", "/sys/fs/cgroup/memory.events", "/sys/fs/cgroup/memory/memory.oom_control"),
            events("max", "/sys/fs/cgroup/pids.events", "/sys/fs/cgroup/pids/pids.events"))

def child():                               # the OOM killer picks a program, never the runner
    libc.prctl(4, 1, 0, 0, 0)
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass

def reset():                               # SIGKILL whatever the program left (forks, daemons, bombs), reap, wipe
    try:
        os.kill(-1, signal.SIGKILL)        # every process but PID 1 (us); a fork can't slip past it
    except ProcessLookupError:
        pass
    try:
        while True:
            os.waitpid(-1, 0)
    except ChildProcessError:
        pass
    for d in ("/tmp", "/dev/shm"):
        for n in os.listdir(d):
            p = os.path.join(d, n)
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                try:
                    os.unlink(p)
                except OSError:
                    pass

def drain(fd, out):                        # keep the tail of stderr; EOF once every process holding the pipe is gone
    with open(fd, "rb", buffering=0) as f:
        while chunk := f.read(65536):
            out[0] = (out[0] + chunk)[-1200:]

def run(path, timeout):
    before, t, out = limits(), time.time(), [b""]
    r, w = os.pipe()
    p = subprocess.Popen([sys.executable, path], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=w,
                         cwd="/tmp", preexec_fn=child)
    os.close(w)
    th = threading.Thread(target=drain, args=(r, out), daemon=True)
    th.start()
    try:
        rc, late = p.wait(timeout), False
    except subprocess.TimeoutExpired:
        p.kill()
        rc, late = p.wait(), True
    dt = time.time() - t
    reset()
    th.join(10)
    after, err = limits(), "timeout" if late else out[0].decode("utf-8", "replace")[-300:]
    return {"ok": rc == 0 and not late, "rc": rc, "err": err, "oom": after[0] > before[0], "pids": after[1] > before[1],
            "t": round(dt, 2)}

res = []
for name in sorted(os.listdir("/work")):
    if name.startswith("p") and name.endswith(".py"):
        res.append({"name": name, **run("/work/" + name, float(sys.argv[1]))})
print(json.dumps(res))
'''


def docker() -> str | None:
    """The docker CLI if it reaches a running engine for Linux containers (the sandbox image is Linux), else None."""
    exe = shutil.which("docker")
    if not exe:
        return None
    try:
        p = subprocess.run([exe, "info", "--format", "{{.OSType}}"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return exe if p.returncode == 0 and p.stdout.strip() == "linux" else None


def ensure_image(exe: str) -> None:
    if subprocess.run([exe, "image", "inspect", IMAGE], capture_output=True).returncode == 0:
        return
    p = subprocess.run([exe, "build", "-t", IMAGE, "-"], input=DOCKERFILE.encode(), capture_output=True, timeout=900)
    if p.returncode:
        raise RuntimeError(f"building {IMAGE} failed: {p.stderr.decode('utf-8', 'replace')[-1500:]}")


def command(exe: str, work: str, timeout_s: float, name: str = "") -> list[str]:
    return [exe, "run", "--rm", *(["--name", name] if name else []), "--network", "none", "--read-only",
            "--tmpfs", "/tmp:rw,size=64m", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--memory", "1g", "--memory-swap", "1g", "--cpus", "2", "--pids-limit", "256", "--user", "10001",
            "-v", f"{work}:/work:ro", IMAGE, "python", "/work/_runner.py", str(timeout_s)]


def run(programs: list[str], timeout_s: float = 10.0) -> list[dict]:
    """Run each program in the sandbox; [{'ok': bool, 'err': str, 'rc', 'oom', 'pids', 't'}] in input order.
    Raises if Docker is missing: the code is never run any other way."""
    exe = docker()
    if not exe:
        raise RuntimeError("Docker (Linux containers) is not running: model-written code only runs in a container")
    ensure_image(exe)
    with tempfile.TemporaryDirectory(prefix="localllm-sbx-") as d:
        for i, src in enumerate(programs):
            Path(d, f"p{i:05d}.py").write_text(src, encoding="utf-8")
        Path(d, "_runner.py").write_text(RUNNER, encoding="utf-8")
        for f in Path(d).iterdir():            # mkdtemp makes the dir 0700: the container user (uid 10001) must read it
            f.chmod(0o644)
        os.chmod(d, 0o755)
        name = Path(d).name                    # so a run that overstays can be killed, not just its docker client
        try:
            out = subprocess.run(command(exe, d, timeout_s, name), capture_output=True, text=True,
                                 timeout=60 + (timeout_s + 1) * len(programs))
        except subprocess.TimeoutExpired:
            subprocess.run([exe, "kill", name], capture_output=True, timeout=60)
            raise
        lines = out.stdout.strip().splitlines()
        if out.returncode or not lines:
            raise RuntimeError(f"sandbox failed (exit {out.returncode}): {out.stderr[-1000:]}")
        rows = {r["name"]: r for r in json.loads(lines[-1])}
    missing = {"ok": False, "rc": None, "err": "missing", "oom": False, "pids": False, "t": 0.0}
    return [rows.get(f"p{i:05d}.py", {**missing}) for i in range(len(programs))]
