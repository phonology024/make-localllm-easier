"""Run model-written code safely: inside a throwaway Docker container, never on the host.

The container has no network, a read-only filesystem (only the in-memory /tmp and /dev/shm are writable), no Linux
capabilities, no privilege escalation, a memory/CPU/process cap, a non-root user, and a per-program timeout. Programs
are passed in through a read-only mount and results come back as JSON lines on stdout. A program passes only if it ran
to its last line and exited 0: it runs as a module, not as '__main__' (like EvalPlus's exec()), so an
`if __name__ == '__main__': unittest.main()` block doesn't run, and an early sys.exit(0) or os._exit(0) fails.
Between two programs every process the first one left behind is killed, and /tmp, /dev/shm and the POSIX/SysV IPC
objects are wiped. A program that leaves state the runner can't undo (it lowered the runner's limits, say) or that
kills the runner fails, and the rest go on in a fresh container. If Docker (with Linux containers) isn't available the
code is NOT run - there is no unsafe fallback.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

IMAGE = "localllm-sandbox:1"
# Python 3.11, like EvalPlus's own image: HumanEval+'s recorded answers predate 3.12's compensated float sum(), which
# moves HumanEval/32's reference (Newton's method) off them.
DOCKERFILE = """FROM python:3.11-slim
RUN pip install --no-cache-dir numpy==2.1.3 && useradd -m -u 10001 runner
USER runner
WORKDIR /tmp
"""
RECIPE = hashlib.sha256(DOCKERFILE.encode()).hexdigest()[:12]      # an image built from an older DOCKERFILE is rebuilt
# How each program is started: as a module named __sandbox__, then a byte on the runner's pipe says it reached its end.
# (Not runpy: its imports make every process of a fork bomb big enough to trip --memory before --pids-limit.)
BOOT = ("import os, sys\nfd, path = int(sys.argv.pop()), sys.argv.pop()\nsys.argv[0] = path\n"
        "m = sys.modules['__sandbox__'] = type(sys)('__sandbox__')\nm.__file__ = path\n"
        "exec(compile(open(path, 'rb').read(), path, 'exec'), m.__dict__)\nos.write(fd, b'1')\n")
# PID 1 of the container. Prints "ready", then per program: ok, rc (exit code, -9 = SIGKILL), err (stderr tail, or
# 'timeout'), oom / pids (it hit the --memory / --pids-limit cap: the container cgroup's own event counters went up)
# and t (seconds); then "end", or "dirty" when a program left state it can't undo (the host goes on in a new container).
RUNNER = r'''
import ctypes, json, os, resource, shutil, signal, subprocess, sys, threading, time
signal.signal(signal.SIGINT, signal.SIG_IGN)   # PID 1 only gets the signals it handles: no KeyboardInterrupt for us
libc = ctypes.CDLL(None)
libc.prctl(4, 0, 0, 0, 0)                  # PR_SET_DUMPABLE 0: programs can't ptrace us or open /proc/1/fd/*
BOOT = open("/work/_boot.py").read()
SCRATCH = [d for d in ("/tmp", "/dev/shm", "/dev/mqueue") if os.path.isdir(d)]
IPC = ("shm", "msg", "sem")

def say(x):
    print(json.dumps(x), flush=True)

def lines(f):
    try:
        return open(f).read().splitlines()[1:]
    except OSError:
        return []

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

def state():                               # what a program (same uid as us) could leave for the next one
    return {"files": sorted(os.path.join(d, n) for d in SCRATCH for n in os.listdir(d)),
            "sysv": [k + " " + l.split()[1] for k in IPC for l in lines("/proc/sysvipc/" + k)],
            "our limits": [resource.getrlimit(getattr(resource, n)) for n in dir(resource) if n.startswith("RLIMIT_")],
            "our priority": [os.getpriority(os.PRIO_PROCESS, 0), os.sched_getscheduler(0)]}

def child():                               # the program gets SIGINT back; the OOM killer picks it, never the runner
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    libc.prctl(4, 1, 0, 0, 0)
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass

def quietly(f, *a):
    try:
        f(*a)
    except OSError:
        pass

def wipe(top):                             # a program can chmod its dirs to 0: make them ours again, then delete
    for root, dirs, _ in os.walk(top):
        for n in dirs:
            if not os.path.islink(os.path.join(root, n)):
                quietly(os.chmod, os.path.join(root, n), 0o700)
    for n in os.listdir(top):
        p = os.path.join(top, n)
        shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) and not os.path.islink(p) else quietly(os.unlink, p)

def reset():                               # SIGKILL whatever the program left (forks, daemons, bombs), reap, wipe
    quietly(os.kill, -1, signal.SIGKILL)   # every process but PID 1 (us); a fork can't slip past it
    try:
        while True:
            os.waitpid(-1, 0)
    except ChildProcessError:
        pass
    for k in IPC:                          # SysV IPC outlives its maker: a big shm segment would eat the memory cap
        for l in lines("/proc/sysvipc/" + k):
            i = int(l.split()[1])
            libc.semctl(i, 0, 0) if k == "sem" else getattr(libc, k + "ctl")(i, 0, None)    # IPC_RMID
    for d in SCRATCH:
        try:
            wipe(d)
        except Exception:                  # state() sees what is left
            pass

def drain(fd, out):                        # keep the tail of stderr; EOF once every process holding the pipe is gone
    with open(fd, "rb", buffering=0) as f:
        while chunk := f.read(65536):
            out[0] = (out[0] + chunk)[-1200:]

def run(path, timeout):
    before, t, out = limits(), time.time(), [b""]
    r, w = os.pipe()
    mr, mw = os.pipe()                     # BOOT writes a byte here once the program has run to its end
    th = threading.Thread(target=drain, args=(r, out), daemon=True)
    th.start()                             # before the program starts: a fork bomb could leave us no thread to start
    p = subprocess.Popen([sys.executable, "-c", BOOT, path, str(mw)], stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=w, cwd="/tmp", preexec_fn=child, pass_fds=(mw,),
                         start_new_session=True)
    os.close(w)
    os.close(mw)
    try:
        rc, late = p.wait(timeout), False
    except subprocess.TimeoutExpired:
        p.kill()
        rc, late = p.wait(), True
    dt = time.time() - t
    reset()
    th.join(10)
    os.set_blocking(mr, False)
    try:
        done = os.read(mr, 1) == b"1"
    except BlockingIOError:
        done = False
    os.close(mr)
    after, err = limits(), "timeout" if late else out[0].decode("utf-8", "replace")[-300:]
    if rc == 0 and not done and not late:
        err = (err + "\n" if err.strip() else "") + "exited with 0 before its last line"
    return {"ok": rc == 0 and done and not late, "rc": rc, "err": err, "oom": after[0] > before[0],
            "pids": after[1] > before[1], "t": round(dt, 2)}

clean = state()
names = sorted(n for n in os.listdir("/work") if n.startswith("p") and n.endswith(".py"))
say("ready")
for name in names[int(sys.argv[2]):]:
    row, now = {"name": name, **run("/work/" + name, float(sys.argv[1]))}, state()
    left = [k for k in clean if now[k] != clean[k]]
    if left:
        row.update(ok=False, err=f"{row['err'][-200:]}\nit left the sandbox changed ({', '.join(left)}), so the rest "
                   "run in a new container")
    say(row)
    if left:
        say("dirty")
        sys.exit(0)
say("end")
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
    p = subprocess.run([exe, "image", "inspect", "--format", '{{index .Config.Labels "localllm.recipe"}}', IMAGE],
                       capture_output=True, text=True)
    if p.returncode == 0 and p.stdout.strip() == RECIPE:
        return
    p = subprocess.run([exe, "build", "--label", f"localllm.recipe={RECIPE}", "-t", IMAGE, "-"],
                       input=DOCKERFILE.encode(), capture_output=True, timeout=900)
    if p.returncode:
        raise RuntimeError(f"building {IMAGE} failed: {p.stderr.decode('utf-8', 'replace')[-1500:]}")


def command(exe: str, work: str, timeout_s: float, name: str = "", start: int = 0) -> list[str]:
    return [exe, "run", "--rm", *(["--name", name] if name else []), "--network", "none", "--read-only",
            "--tmpfs", "/tmp:rw,size=64m", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--memory", "1g", "--memory-swap", "1g", "--cpus", "2", "--pids-limit", "256", "--user", "10001",
            "-v", f"{work}:/work:ro", IMAGE, "python", "/work/_runner.py", str(timeout_s), str(start)]


def _container(exe: str, d: str, timeout_s: float, start: int, n: int) -> tuple[list, int, str]:
    """One container running programs start..n-1: its output lines (parsed), exit code and stderr."""
    name = f"{Path(d).name}-{start}"           # so a run that overstays can be killed, not just its docker client
    try:
        out = subprocess.run(command(exe, d, timeout_s, name, start), capture_output=True, text=True,
                             timeout=60 + (timeout_s + 1) * (n - start))
    except subprocess.TimeoutExpired:
        subprocess.run([exe, "kill", name], capture_output=True, timeout=60)
        raise
    got = []
    for line in out.stdout.splitlines():
        try:
            got.append(json.loads(line))
        except ValueError:
            pass
    return got, out.returncode, out.stderr


def run(programs: list[str], timeout_s: float = 10.0) -> list[dict]:
    """Run each program in the sandbox; [{'ok': bool, 'err': str, 'rc', 'oom', 'pids', 't'}] in input order.
    Raises if Docker is missing: the code is never run any other way."""
    exe = docker()
    if not exe:
        raise RuntimeError("Docker (Linux containers) is not running: model-written code only runs in a container")
    ensure_image(exe)
    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="localllm-sbx-") as d:
        for i, src in enumerate(programs):
            Path(d, f"p{i:05d}.py").write_text(src, encoding="utf-8")
        Path(d, "_runner.py").write_text(RUNNER, encoding="utf-8")
        Path(d, "_boot.py").write_text(BOOT, encoding="utf-8")
        for f in Path(d).iterdir():            # mkdtemp makes the dir 0700: the container user (uid 10001) must read it
            f.chmod(0o644)
        os.chmod(d, 0o755)
        while len(rows) < len(programs):       # one program that breaks the runner costs a new container, not the batch
            got, code, err = _container(exe, d, timeout_s, len(rows), len(programs))
            if not got or got[0] != "ready":
                raise RuntimeError(f"sandbox failed (exit {code}): {err[-1000:]}")
            new = [r for r in got if isinstance(r, dict)]
            if got[-1] not in ("end", "dirty") and len(rows) + len(new) < len(programs):   # died running the next one
                new.append({"name": f"p{len(rows) + len(new):05d}.py", "ok": False, "rc": None, "oom": False,
                            "pids": False, "t": 0.0, "err": f"the sandbox runner died (exit {code}): {err[-300:]}"})
            if not new:
                raise RuntimeError(f"sandbox ran nothing (exit {code}): {err[-1000:]}")
            rows += new
    return rows
