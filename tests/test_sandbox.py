"""The codegen sandbox (localllm/sandbox.py). The Docker tests need a running engine for Linux containers and are
skipped without one (Windows-container engines and macOS runners skip too), or when the sandbox image can't be built
here (offline, Docker Hub rate limit); .github/workflows/sandbox.yml sets LOCALLLM_SANDBOX_CI=1 so that there either
one fails instead. The EvalPlus canonical-solution run downloads HumanEval+/MBPP+ from Hugging Face, so it only runs
when LOCALLLM_CANONICAL=1."""
import ast
import os
import re
import subprocess
import time

import pytest

from localllm import bench, sandbox

CI = os.environ.get("LOCALLLM_SANDBOX_CI")
needs_docker = pytest.mark.skipif(not sandbox.docker() and not CI,
                                  reason="needs a running Docker engine for Linux containers")
ALONE = ("import os\nassert sorted(int(p) for p in os.listdir('/proc') if p.isdigit()) == [1, os.getpid()], "
         "os.listdir('/proc')\nassert os.listdir('/tmp') == [] and os.listdir('/dev/shm') == [], "
         "(os.listdir('/tmp'), os.listdir('/dev/shm'))\n"
         "assert all(len(open('/proc/sysvipc/' + k).readlines()) == 1 for k in ('shm', 'msg', 'sem'))\n")


@pytest.fixture
def image():
    try:
        sandbox.ensure_image(sandbox.docker() or "docker")
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
        if CI:
            raise
        pytest.skip(f"can't build {sandbox.IMAGE} here: {str(e)[-300:]}")


def test_sandbox_refuses_without_docker(monkeypatch):
    calls = []
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: None)
    monkeypatch.setattr(sandbox.subprocess, "run", lambda *a, **k: calls.append(a))
    assert sandbox.docker() is None
    with pytest.raises(RuntimeError, match="Docker"):
        sandbox.run(["print('never runs')"])
    assert calls == []                                       # nothing was started, on the host or anywhere else


def test_sandbox_needs_a_linux_container_engine(monkeypatch):
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/docker")
    for os_type, ok in (("windows\n", False), ("linux\n", True)):
        monkeypatch.setattr(sandbox.subprocess, "run",
                            lambda *a, o=os_type, **k: subprocess.CompletedProcess(a, 0, o, ""))
        assert (sandbox.docker() == "/usr/bin/docker") is ok
    monkeypatch.setattr(sandbox.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "no daemon"))
    assert sandbox.docker() is None
    compile(sandbox.RUNNER, "_runner.py", "exec")
    compile(sandbox.BOOT, "-c", "exec")


def test_a_program_that_breaks_the_runner_costs_a_new_container_not_the_batch(monkeypatch):
    def row(i):
        return {"name": f"p{i:05d}.py", "ok": True, "rc": 0, "err": "", "oom": False, "pids": False, "t": 0.1}
    script = {0: (["ready", row(0)], 137, "Killed"),            # the runner died while it ran p00001
              2: (["ready", row(2), "dirty"], 0, ""),             # p00002 left state behind: go on in a new container
              3: (["ready", row(3), "end"], 0, "")}
    starts = []
    monkeypatch.setattr(sandbox, "docker", lambda: "docker")
    monkeypatch.setattr(sandbox, "ensure_image", lambda exe: None)
    monkeypatch.setattr(sandbox, "_container", lambda exe, d, t, start, n: starts.append(start) or script[start])
    res = sandbox.run(["a", "b", "c", "d"])
    assert starts == [0, 2, 3] and [r["ok"] for r in res] == [True, False, True, True]
    assert res[1]["name"] == "p00001.py" and "runner died (exit 137): Killed" in res[1]["err"]
    monkeypatch.setattr(sandbox, "_container", lambda *a: ([], 125, "docker: invalid reference format"))
    with pytest.raises(RuntimeError, match="sandbox failed"):            # a broken sandbox is an error, not a score
        sandbox.run(["a"])


@needs_docker
def test_sandbox_blocks_attacks_for_the_right_reason(image):
    shm = ("import ctypes\nlibc = ctypes.CDLL(None)\n"
           "libc.shmget.argtypes = [ctypes.c_int, ctypes.c_size_t, ctypes.c_int]\n"
           "libc.shmat.restype = ctypes.c_void_p\ni = libc.shmget(0, 600 * 1024 ** 2, 0o1600)\nassert i >= 0\n"
           "ctypes.memset(libc.shmat(i, None, 0), 1, 600 * 1024 ** 2)\n")       # 600 MB that outlive the program
    progs = {
        "pass": "assert sum(range(10)) == 45\n",
        "posture": "import os\nassert os.getuid() == 10001\nst = open('/proc/self/status').read()\n"
                   "assert 'CapEff:\\t0000000000000000' in st and 'CapBnd:\\t0000000000000000' in st, st\n"
                   "assert 'NoNewPrivs:\\t1' in st, st\n"
                   "assert os.listdir('/sys/class/net') == ['lo']\n",
        "assert": "assert 1 + 1 == 3, 'wrong on purpose'\n",
        "loop": "while True:\n    pass\n",
        "fork_bomb": "import os\nwhile True:\n    os.fork()\n",
        "after_bomb": ALONE,
        "memory": "chunks = [bytearray(64 * 1024 ** 2) for _ in range(32)]\n",           # 2 GB, 64 MB at a time
        "network": "import socket\nsocket.create_connection(('1.1.1.1', 53), timeout=5)\n",
        "dns": "import socket\nsocket.getaddrinfo('example.com', 443)\n",
        "write_home": "open('/home/runner/x', 'w').write('x')\n",   # the sandbox user's dir: only --read-only stops it
        "write_work": "open('/work/p99999.py', 'w').write('x')\n",
        "write_usr": "import sysconfig\nopen(sysconfig.get_paths()['purelib'] + '/evil.pth', 'w').write('x')\n",
        "tmp_full": "open('/tmp/big', 'wb').write(bytes(100 * 1024 ** 2))\n",              # /tmp is a 64 MB tmpfs
        "daemon": "import os, time\nopen('/dev/shm/left', 'w').write('x')\n"
                  "if os.fork() == 0:\n    os.setsid()\n    time.sleep(600)\n",
        "after_daemon": ALONE,
        "tamper": "open('/proc/1/fd/1', 'w').write('[]\\n')\n",                          # forge the runner's output
        "kill_runner": "import os, signal\nos.kill(1, signal.SIGKILL)\n",
        "sigint_runner": "import os, signal, time\nos.kill(1, signal.SIGINT)\ntime.sleep(.3)\n",       # ignored
        "sigint_group": "import os, signal, time\nos.kill(0, signal.SIGINT)\ntime.sleep(5)\n",   # only itself
        "chmod_dirs": "import os\nos.makedirs('/tmp/d/e')\nopen('/tmp/d/e/f', 'wb').write(bytes(40 * 1024 ** 2))\n"
                      "os.makedirs('/dev/shm/q/r')\nfor p, m in (('/tmp/d/e', 0), ('/tmp/d', 0o500), "
                      "('/dev/shm/q/r', 0), ('/dev/shm/q', 0)):\n    os.chmod(p, m)\n",
        "after_chmod": ALONE + "open('/tmp/x', 'wb').write(bytes(60 * 1024 ** 2))\n",   # the 40 MB were freed too
        "sysv_shm": shm,
        "after_shm": "b = b'1' * (600 * 1024 ** 2)\n" + ALONE,                          # fits only if the 600 MB went
        "limit_runner": "import os, resource\nresource.prlimit(1, resource.RLIMIT_NOFILE, (64, 64))\n"
                        "os.setpriority(os.PRIO_PROCESS, 1, 19)\n",
        "after_limit": ALONE + "import resource\nassert resource.prlimit(1, resource.RLIMIT_NOFILE)[1] > 64\n"
                               "assert os.getpriority(os.PRIO_PROCESS, 1) == 0\n",       # a new runner
        "exit_early": "import sys\nsys.exit(0)\nassert False, 'the tests that follow never ran'\n",
        "os_exit": "import os\nos._exit(0)\n",
        "guarded": "if __name__ == '__main__':\n    raise SystemExit('the main block ran')\n",  # skipped (EvalPlus too)
        "last": "assert True\n",
    }
    t0 = time.time()
    res = dict(zip(progs, sandbox.run(list(progs.values()), timeout_s=5)))
    for name, r in res.items():
        print(f"{name:13} ok={r['ok']!s:5} rc={r['rc']!s:4} oom={r['oom']!s:5} pids={r['pids']!s:5} t={r['t']:5.2f}s  "
              f"{r['err'].strip().splitlines()[-1] if r['err'].strip() else ''}")
    print(f"{len(progs)} programs in {time.time() - t0:.1f}s")
    good = ("pass", "posture", "after_bomb", "daemon", "after_daemon", "kill_runner", "sigint_runner", "chmod_dirs",
            "after_chmod", "sysv_shm", "after_shm", "after_limit", "guarded", "last")
    for name, r in res.items():                               # the runner survived every attack and reported on all
        assert r["ok"] == (name in good), (name, r)
        assert (r["rc"] == 0) == (name in good or name in ("limit_runner", "exit_early", "os_exit")), (name, r)
    assert res["assert"]["rc"] == 1 and "AssertionError: wrong on purpose" in res["assert"]["err"]
    assert res["loop"]["err"] == "timeout" and res["loop"]["rc"] == -9 and res["loop"]["t"] >= 5
    bomb = res["fork_bomb"]                                   # fork() refused by --pids-limit, not by memory
    assert bomb["pids"] and not bomb["oom"] and "BlockingIOError: [Errno 11]" in bomb["err"], bomb
    mem = res["memory"]                                       # SIGKILLed by the cgroup OOM killer at --memory 1g
    assert mem["oom"] and mem["rc"] == -9 and not mem["pids"], mem
    assert "[Errno 101] Network is unreachable" in res["network"]["err"]                       # --network none
    assert "gaierror" in res["dns"]["err"]
    for name in ("write_home", "write_work", "write_usr"):                                     # --read-only
        assert "[Errno 30] Read-only file system" in res[name]["err"], (name, res[name])
    assert "[Errno 28] No space left on device" in res["tmp_full"]["err"]
    assert "PermissionError" in res["tamper"]["err"]
    assert res["sigint_group"]["rc"] == -2                    # the program got SIGINT back (the runner ignores it)
    assert "left the sandbox changed (our limits, our priority)" in res["limit_runner"]["err"]
    for name in ("exit_early", "os_exit"):                    # passing needs the program's last line, not just exit 0
        assert res[name]["err"].endswith("exited with 0 before its last line"), (name, res[name])
    assert not any(r["oom"] or r["pids"] for n, r in res.items() if n not in ("fork_bomb", "memory"))


def fence(code: str) -> str:
    return f"Here is the solution:\n```python\n{code.rstrip()}\n```\n"


def entry_only(row: dict) -> str:
    """Just the asked-for function, as a model that doesn't repeat the prompt's imports and helpers would send it."""
    src = row["prompt"] + row["canonical_solution"]
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == row["entry_point"])
    return ast.get_source_segment(src, fn)


@needs_docker
@pytest.mark.skipif(not os.environ.get("LOCALLLM_CANONICAL"),
                    reason="downloads HumanEval+/MBPP+ from Hugging Face; the sandbox CI job sets LOCALLLM_CANONICAL=1")
def test_evalplus_canonical_solutions_pass(image, monkeypatch, tmp_path):
    rows, real = {}, bench._rows
    monkeypatch.setattr(bench, "_rows", lambda ds, cfg, split="test": rows.setdefault(ds, real(ds, cfg, split)))
    monkeypatch.setattr(bench, "HOME", tmp_path)
    items = bench.load("codegen", "en")                      # exactly what `localllm eval --suites codegen` tests
    he, mb = rows["evalplus/humanevalplus"], rows["evalplus/mbppplus"]
    assert len(he) == 164 and len(mb) == 378 and len(items) == 542
    assert [it["id"] for it in items] == [r["task_id"] for r in he] + [f"Mbpp/{r['task_id']}" for r in mb]
    for it in items:                                         # every repair still finds what it repairs
        if it["id"] in bench.CODEGEN_FIXES:
            assert len(re.findall(bench.CODEGEN_FIXES[it["id"]][0], it["test"])) == 1, it["id"]
    cases = [("HumanEval+ full", it, fence(r["prompt"] + r["canonical_solution"])) for it, r in zip(items, he)]
    cases += [("HumanEval+ bare", it, fence(entry_only(r))) for it, r in zip(items, he)]
    cases += [("MBPP+", it, fence(r["code"])) for it, r in zip(items[len(he):], mb)]
    timeout = float(os.environ.get("LOCALLLM_CANONICAL_TIMEOUT", bench.CODEGEN_TIMEOUT))
    t0 = time.time()
    res = sandbox.run([bench.codegen_program(it, reply) for _, it, reply in cases], timeout)
    print(f"\n{len(cases)} programs in {time.time() - t0:.0f}s, per-program timeout {timeout:g}s")
    for group in ("HumanEval+ full", "HumanEval+ bare", "MBPP+"):
        mine = [r for (g, _, _), r in zip(cases, res) if g == group]
        print(f"  {group:16} {sum(r['ok'] for r in mine)}/{len(mine)} pass, slowest {max(r['t'] for r in mine):.2f}s")
    slow = sorted(((r["t"], f"{g} {it['id']}") for (g, it, _), r in zip(cases, res)), reverse=True)[:10]
    print("  slowest:", ", ".join(f"{n} {t:.2f}s" for t, n in slow))
    for (g, it, _), r in zip(cases, res):
        if it["id"] in bench.CODEGEN_FIXES or not r["ok"]:
            print(f"  {'ok  ' if r['ok'] else 'FAIL'} {g} {it['id']}: rc={r['rc']} oom={r['oom']} t={r['t']} "
                  f"{r['err'][-300:]!r}")
    bad = [f"{g} {it['id']}" for (g, it, _), r in zip(cases, res) if not r["ok"]]
    assert not bad, f"canonical solutions that failed: {bad}"
