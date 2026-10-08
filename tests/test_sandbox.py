"""The codegen sandbox (localllm/sandbox.py). The Docker tests need a running engine for Linux containers and are
skipped without one (Windows-container engines and macOS runners skip too); .github/workflows/sandbox.yml sets
LOCALLLM_SANDBOX_CI=1 so that there a missing engine fails instead. The EvalPlus canonical-solution run downloads
HumanEval+/MBPP+ from Hugging Face, so it only runs when LOCALLLM_CANONICAL=1."""
import ast
import os
import subprocess
import time

import pytest

from localllm import bench, sandbox

needs_docker = pytest.mark.skipif(not sandbox.docker() and not os.environ.get("LOCALLLM_SANDBOX_CI"),
                                  reason="needs a running Docker engine for Linux containers")
ALONE = ("import os\nassert sorted(int(p) for p in os.listdir('/proc') if p.isdigit()) == [1, os.getpid()], "
         "os.listdir('/proc')\nassert os.listdir('/tmp') == [] and os.listdir('/dev/shm') == []\n")


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


@needs_docker
def test_sandbox_blocks_attacks_for_the_right_reason():
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
        "write_home": "open('/home/runner/x', 'w').write('x')\n",   # the sandbox user owns it: only --read-only stops this
        "write_work": "open('/work/p99999.py', 'w').write('x')\n",
        "write_usr": "open('/usr/local/lib/python3.12/site-packages/evil.pth', 'w').write('x')\n",
        "tmp_full": "open('/tmp/big', 'wb').write(bytes(100 * 1024 ** 2))\n",              # /tmp is a 64 MB tmpfs
        "daemon": "import os, time\nopen('/dev/shm/left', 'w').write('x')\n"
                  "if os.fork() == 0:\n    os.setsid()\n    time.sleep(600)\n",
        "after_daemon": ALONE,
        "tamper": "open('/proc/1/fd/1', 'w').write('[]\\n')\n",                          # forge the runner's output
        "kill_runner": "import os, signal\nos.kill(1, signal.SIGKILL)\n",
        "last": "assert True\n",
    }
    t0 = time.time()
    res = dict(zip(progs, sandbox.run(list(progs.values()), timeout_s=5)))
    for name, r in res.items():
        print(f"{name:12} ok={r['ok']!s:5} rc={r['rc']!s:4} oom={r['oom']!s:5} pids={r['pids']!s:5} t={r['t']:5.2f}s  "
              f"{r['err'].strip().splitlines()[-1] if r['err'].strip() else ''}")
    print(f"{len(progs)} programs in {time.time() - t0:.1f}s")
    good = ("pass", "posture", "after_bomb", "daemon", "after_daemon", "kill_runner", "last")
    for name, r in res.items():                               # the runner survived every attack and reported on all
        assert r["ok"] == (name in good) == (r["rc"] == 0), (name, r)
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
def test_evalplus_canonical_solutions_pass(monkeypatch, tmp_path):
    rows, real = {}, bench._rows
    monkeypatch.setattr(bench, "_rows", lambda ds, cfg, split="test": rows.setdefault(ds, real(ds, cfg, split)))
    monkeypatch.setattr(bench, "HOME", tmp_path)
    items = bench.load("codegen", "en")                      # exactly what `localllm eval --suites codegen` tests
    he, mb = rows["evalplus/humanevalplus"], rows["evalplus/mbppplus"]
    kept = [r for r in mb if f"Mbpp/{r['task_id']}" not in bench.CODEGEN_SKIP]
    assert len(he) == 164 and len(mb) - len(kept) == len(bench.CODEGEN_SKIP) and len(items) == len(he) + len(kept)
    assert [it["id"] for it in items] == [r["task_id"] for r in he] + [f"Mbpp/{r['task_id']}" for r in kept]
    cases = [("HumanEval+ full", it, fence(r["prompt"] + r["canonical_solution"])) for it, r in zip(items, he)]
    cases += [("HumanEval+ bare", it, fence(entry_only(r))) for it, r in zip(items, he)]
    cases += [("MBPP+", it, fence(r["code"])) for it, r in zip(items[len(he):], kept)]
    cases += [("skipped", {"id": f"Mbpp/{r['task_id']}", "head": "", "test": r["test"]}, fence(r["code"]))
              for r in mb if r not in kept]                  # still run: they must fail for the stated reason
    timeout = float(os.environ.get("LOCALLLM_CANONICAL_TIMEOUT", bench.CODEGEN_TIMEOUT))
    t0 = time.time()
    res = sandbox.run([bench.codegen_program(it, reply) for _, it, reply in cases], timeout)
    print(f"\n{len(cases)} programs in {time.time() - t0:.0f}s, per-program timeout {timeout:g}s")
    for group in ("HumanEval+ full", "HumanEval+ bare", "MBPP+", "skipped"):
        mine = [r for (g, _, _), r in zip(cases, res) if g == group]
        print(f"  {group:16} {sum(r['ok'] for r in mine)}/{len(mine)} pass, slowest {max(r['t'] for r in mine):.2f}s")
    slow = sorted(((r["t"], f"{g} {it['id']}") for (g, it, _), r in zip(cases, res)), reverse=True)[:10]
    print("  slowest:", ", ".join(f"{n} {t:.2f}s" for t, n in slow))
    for (g, it, _), r in zip(cases, res):
        if not r["ok"]:
            print(f"  FAIL {g} {it['id']}: rc={r['rc']} oom={r['oom']} pids={r['pids']} t={r['t']} {r['err'][-300:]!r}")
    bad = [it["id"] for (g, it, _), r in zip(cases, res) if r["ok"] == (g == "skipped")]
    assert not bad, f"canonical solutions that failed: {bad}"
    assert all(r["oom"] for (g, _, _), r in zip(cases, res) if g == "skipped")    # too big for --memory 1g, as stated
