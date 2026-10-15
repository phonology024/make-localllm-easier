"""localllm-agent: a small plan / tool / verify / retry loop for local models (#88). An add-on: it is its own package
and the base `make-localllm-easier` install never imports it.

  localllm-agent "make the failing test pass" --check "python -m unittest" [--project .] [--url http://127.0.0.1:8080]

1. Plan. One call turns the task into a short plan; "done" is fixed up front: the --check command passes.
2. Work. The model gets four tools - list_files, read_file, write_file, run - and works until it stops calling them.
   Files change only through write_file, inside the project; a path outside it is refused unless you say yes.
   `run` executes in the Docker sandbox (no network, capped memory/CPU/processes, a non-root user) on a throwaway
   copy of the project, so a command never changes your files or reaches the network. No Docker, no commands.
3. Verify. The check runs the same way. Pass: done. Fail: the next round starts from a fresh context with the task, the
   plan and the check's output (a small context keeps the KV cache - and RAM - small), up to --rounds times.
Every component is an assumption about what a small model can't do alone; #88 measures which ones pay off before
anything ships to the base install.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

PLAN_PROMPT = ("You plan coding tasks for a small model with tools. Answer with at most 6 short numbered steps. "
               "The task is done when this command passes: {check}")
WORK_PROMPT = ("You change files in a project to finish a task. Use the tools: list_files and read_file to look, "
               "write_file to change a file (whole content), run to try a command. Paths are relative to the project. "
               "When you think the task is done, answer in one sentence without calling a tool.")
TOOLS = [
    {"type": "function", "function": {"name": "list_files", "description": "Files under a folder of the project",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "read_file", "description": "A file's text",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                                     "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Replace a file's whole content",
                                      "parameters": {"type": "object", "properties": {
                                          "path": {"type": "string"}, "content": {"type": "string"}},
                                          "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "run", "description": "Run a shell command in a sandboxed copy",
                                      "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}},
                                                     "required": ["cmd"]}}},
]
MAX_READ = 20_000            # characters of one file shown to the model
MAX_OUT = 4_000              # characters of command output shown to the model


class Tools:
    def __init__(self, project: Path, runner, confirm=lambda question: False):
        self.root, self.runner, self.confirm = project.resolve(), runner, confirm
        self.written: list[str] = []

    def _path(self, rel: str, writing: bool = False) -> Path | None:
        p = (self.root / rel).resolve()
        if p == self.root or self.root in p.parents:
            return p
        if writing and self.confirm(f"write {p}, outside the project {self.root}?"):
            return p
        return None

    def call(self, name: str, raw_args: str) -> str:
        try:
            args = json.loads(raw_args or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except ValueError as e:
            return f"error: the arguments are not valid JSON ({e}); call the tool again with a JSON object"
        try:
            return getattr(self, f"t_{name}")(**args)
        except AttributeError:
            return f"error: no tool named {name!r}; use list_files, read_file, write_file or run"
        except TypeError as e:
            return f"error: wrong arguments for {name}: {e}"
        except OSError as e:
            return f"error: {e}"

    def t_list_files(self, path: str = ".") -> str:
        p = self._path(path)
        if p is None or not p.is_dir():
            return f"error: {path} is not a folder in the project"
        names = sorted(str(f.relative_to(self.root)).replace("\\", "/") for f in p.rglob("*")
                       if f.is_file() and ".git" not in f.parts)
        return "\n".join(names[:500]) or "(empty)"

    def t_read_file(self, path: str) -> str:
        p = self._path(path)
        if p is None or not p.is_file():
            return f"error: {path} is not a file in the project"
        text = p.read_text(encoding="utf-8", errors="replace")
        return text if len(text) <= MAX_READ else text[:MAX_READ] + f"\n... ({len(text) - MAX_READ} more characters)"

    def t_write_file(self, path: str, content: str) -> str:
        p = self._path(path, writing=True)
        if p is None:
            return f"error: {path} is outside the project; refused"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        self.written.append(path)
        return f"wrote {path} ({len(content)} characters)"

    def t_run(self, cmd: str) -> str:
        ok, out = self.runner(cmd, self.root)
        return ("passed" if ok else "failed") + f"\n{out[-MAX_OUT:]}"


def docker_runner(cmd: str, project: Path, timeout_s: float = 300) -> tuple[bool, str]:
    """Run `cmd` in localllm's sandbox image on a throwaway copy of the project (no network, capped, non-root)."""
    from localllm import sandbox
    exe = sandbox.docker()
    if not exe:
        return False, "Docker (Linux containers) is not running: commands only run in the sandbox"
    sandbox.ensure_image(exe)
    with tempfile.TemporaryDirectory(prefix="localllm-agent-") as d:
        work = Path(d) / "work"
        shutil.copytree(project, work, ignore=shutil.ignore_patterns(".git", "__pycache__", ".venv", "node_modules"))
        for f in [work, *work.rglob("*")]:
            f.chmod(0o777 if f.is_dir() else 0o666)          # the container user (uid 10001) works on the copy
        argv = [exe, "run", "--rm", "--network", "none", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--memory", "1g", "--memory-swap", "1g", "--cpus", "2", "--pids-limit", "256", "--user", "10001",
                "-v", f"{work}:/work", "-w", "/work", sandbox.IMAGE, "sh", "-c", cmd]
        try:
            p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return False, f"timed out after {timeout_s:g} s"
        return p.returncode == 0, (p.stdout + p.stderr)


def http_chat(url: str, model: str = "local"):
    def chat(messages: list[dict], tools: list[dict] | None = None) -> dict:
        body = {"model": model, "messages": messages, "stream": False, **({"tools": tools} if tools else {})}
        req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            return json.load(r)["choices"][0]["message"]
    return chat


def run_task(task: str, project: Path, check: str, chat, runner, rounds: int = 3, max_steps: int = 25,
             confirm=lambda q: False, log=print) -> dict:
    """Plan, then up to `rounds` work+verify rounds. Returns {"ok", "rounds", "steps", "written", "check_output"}."""
    tools = Tools(project, runner, confirm)
    plan = chat([{"role": "system", "content": PLAN_PROMPT.format(check=check)},
                 {"role": "user", "content": task}]).get("content") or ""
    log(f"plan:\n{plan}")
    failed, steps = "", 0
    for rnd in range(1, rounds + 1):
        brief = f"Task: {task}\n\nPlan:\n{plan}\n\nDone means this command passes: {check}"
        if failed:
            brief += f"\n\nThe last attempt did not pass. Output of the check:\n{failed[-MAX_OUT:]}"
        msgs = [{"role": "system", "content": WORK_PROMPT}, {"role": "user", "content": brief}]   # a fresh context
        for _ in range(max_steps):
            msg = chat(msgs, TOOLS)
            steps += 1
            calls = msg.get("tool_calls") or []
            msgs.append({"role": "assistant", "content": msg.get("content") or "", **({"tool_calls": calls} if calls else {})})
            if not calls:
                break
            for c in calls:
                fn = c.get("function") or {}
                result = tools.call(fn.get("name", ""), fn.get("arguments", "{}"))
                log(f"  {fn.get('name')}: {result.splitlines()[0] if result else ''}")
                msgs.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": result})
        ok, out = runner(check, tools.root)
        log(f"round {rnd}: check {'passed' if ok else 'failed'}")
        if ok:
            return {"ok": True, "rounds": rnd, "steps": steps, "written": tools.written, "check_output": out}
        failed = out
    return {"ok": False, "rounds": rounds, "steps": steps, "written": tools.written, "check_output": failed}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("task")
    ap.add_argument("--check", required=True, help="the command that must pass, e.g. 'python -m unittest'")
    ap.add_argument("--project", type=Path, default=Path("."))
    ap.add_argument("--url", default="http://127.0.0.1:8080", help="the localllm endpoint")
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()

    def ask(question: str) -> bool:
        return sys.stdin.isatty() and input(f"[localllm-agent] {question} [y/N] ").strip().lower() in ("y", "yes")

    r = run_task(a.task, a.project, a.check, http_chat(a.url), docker_runner, a.rounds, confirm=ask)
    print(json.dumps({k: v for k, v in r.items() if k != "check_output"}))
    sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
