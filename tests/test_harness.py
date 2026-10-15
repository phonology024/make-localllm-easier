"""The #88 add-on (addons/harness): plan, tools, verify, retry - with a scripted fake model and a fake sandbox."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addons" / "harness"))
from localllm_harness import agent  # noqa: E402


def call(name, i=0, **args):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class Model:
    """Answers from a script; records what it was shown."""
    def __init__(self, script):
        self.script, self.seen = list(script), []

    def __call__(self, messages, tools=None):
        self.seen.append(([dict(m) for m in messages], tools))     # a copy: the loop keeps appending
        return self.script.pop(0)


def check_runner(project_ok):
    """A fake sandbox: the check passes when calc.py has the fix; `run` echoes the command."""
    def runner(cmd, project):
        ok = "return a + b" in (project / "calc.py").read_text(encoding="utf-8")
        return ok, f"$ {cmd}\n" + ("OK" if ok else "FAIL: add(2, 2) returned 0")
    return runner


def project(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return 0\n", encoding="utf-8")
    (tmp_path / "test_calc.py").write_text("import calc\nassert calc.add(2, 2) == 4\n", encoding="utf-8")
    return tmp_path


def test_plan_work_verify_in_one_round(tmp_path):
    p = project(tmp_path)
    m = Model([{"content": "1. read calc.py 2. fix add"},
               {"content": "", "tool_calls": [call("read_file", path="calc.py")]},
               {"content": "", "tool_calls": [call("write_file", 1, path="calc.py",
                                                   content="def add(a, b):\n    return a + b\n")]},
               {"content": "Fixed add."}])
    r = agent.run_task("make the test pass", p, "python test_calc.py", m, check_runner(True), log=lambda s: None)
    assert r["ok"] and r["rounds"] == 1 and r["written"] == ["calc.py"]
    assert "python test_calc.py" in m.seen[0][0][0]["content"]                      # done is fixed before work
    tool_reply = m.seen[2][0][-1]
    assert tool_reply["role"] == "tool" and "return 0" in tool_reply["content"]     # the model saw the file


def test_a_failed_check_starts_a_fresh_round_with_its_output(tmp_path):
    p = project(tmp_path)
    m = Model([{"content": "plan"},
               {"content": "", "tool_calls": [call("write_file", path="calc.py", content="def add(a, b):\n    return 4\n")]},
               {"content": "done"},
               {"content": "", "tool_calls": [call("write_file", path="calc.py",
                                                   content="def add(a, b):\n    return a + b\n")]},
               {"content": "done now"}])
    r = agent.run_task("make the test pass", p, "python test_calc.py", m, check_runner(True), log=lambda s: None)
    assert r["ok"] and r["rounds"] == 2
    second = m.seen[3][0]
    assert len(second) == 2 and "FAIL: add(2, 2) returned 0" in second[1]["content"]   # fresh context + the failure


def test_gives_up_after_the_rounds(tmp_path):
    p = project(tmp_path)
    m = Model([{"content": "plan"}] + [{"content": "I can't"}] * 3)
    r = agent.run_task("t", p, "check", m, check_runner(True), rounds=3, log=lambda s: None)
    assert not r["ok"] and r["rounds"] == 3 and "FAIL" in r["check_output"]


def test_writes_stay_in_the_project_unless_you_say_yes(tmp_path):
    p = project(tmp_path / "proj")
    outside = tmp_path / "outside.txt"
    t = agent.Tools(p, check_runner(True))
    assert "refused" in t.call("write_file", json.dumps({"path": "../outside.txt", "content": "x"}))
    assert not outside.exists()
    asked = []
    t = agent.Tools(p, check_runner(True), confirm=lambda q: asked.append(q) or True)
    assert t.call("write_file", json.dumps({"path": str(outside), "content": "x"})).startswith("wrote")
    assert outside.exists() and "outside the project" in asked[0]
    assert "is not a file in the project" in t.call("read_file", json.dumps({"path": str(outside)}))  # reads: never


def test_tool_errors_go_back_to_the_model_instead_of_crashing(tmp_path):
    t = agent.Tools(project(tmp_path), check_runner(True))
    assert "not valid JSON" in t.call("read_file", '{"path": "calc.py"')
    assert "no tool named 'rm'" in t.call("rm", "{}")
    assert "wrong arguments" in t.call("read_file", json.dumps({"file": "calc.py"}))
    assert "calc.py" in t.call("list_files", "{}") and "test_calc.py" in t.call("list_files", "{}")
    out = t.call("run", json.dumps({"cmd": "python test_calc.py"}))
    assert out.startswith("failed") and "$ python test_calc.py" in out


def test_the_base_install_never_imports_the_add_on():
    assert "localllm_harness" not in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    src = "".join(p.read_text(encoding="utf-8") for p in (ROOT / "localllm").glob("*.py"))
    assert "localllm_harness" not in src
