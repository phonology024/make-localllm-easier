"""`localllm launch <tool>` (#117): server, connect, tool, undo - in that order, and cleaned up however the tool ends."""
import pytest

from localllm import cli, connect, launch


class Proc:
    def __init__(self, log, name, code=0, raises=None):
        self.log, self.name, self.code, self.raises = log, name, code, raises

    def wait(self):
        self.log.append(f"{self.name} exited")
        if self.raises:
            exc, self.raises = self.raises, None
            raise exc
        return self.code

    def terminate(self):
        self.log.append(f"{self.name} stopped")


def _run(tool="opencode", alive=False, keep=False, connected_before=False, code=0, raises=None, which=True,
         extra=()):
    log = []

    def start_server():
        log.append("server started")
        return Proc(log, "server"), "http://127.0.0.1:8080"

    def popen(argv):
        log.append("run " + " ".join(argv))
        return Proc(log, "tool", code, raises)

    rc = launch.run(tool, 8080, keep, list(extra), start_server, alive=lambda url: alive, say=lambda m: None,
                    popen=popen, which=(lambda c: f"/bin/{c}") if which else (lambda c: None),
                    connect_fn=lambda t, url, model: log.append(f"connect {t} {url} {model}") or "ok",
                    undo_fn=lambda t: log.append(f"undo {t}") or "restored",
                    model_id=lambda url: "gemma4-26b-a4b-qat", was_connected=lambda t: connected_before)
    return rc, log


def test_starts_server_connects_runs_and_cleans_up_in_order():
    rc, log = _run(code=3, extra=["--continue"])
    assert rc == 3
    assert log == ["server started", "connect opencode http://127.0.0.1:8080 gemma4-26b-a4b-qat",
                   "run /bin/opencode --continue", "tool exited", "undo opencode", "server stopped"]


def test_a_running_server_is_reused_and_left_running():
    rc, log = _run(alive=True)
    assert "server started" not in log and "server stopped" not in log and log[-1] == "undo opencode"


def test_keep_and_an_earlier_connect_leave_the_config():
    assert "undo opencode" not in _run(keep=True)[1]
    assert "undo opencode" not in _run(connected_before=True)[1]     # `connect` was the user's choice: keep it


def test_codex_gets_the_profile_connect_writes():
    assert "run /bin/codex --profile localllm" in _run("codex")[1]


def test_ctrl_c_goes_to_the_tool_and_cleanup_still_runs():
    rc, log = _run(raises=KeyboardInterrupt(), code=130)
    assert rc == 130 and log.count("tool exited") == 2 and log[-2:] == ["undo opencode", "server stopped"]


def test_a_tool_that_fails_to_start_still_undoes_and_stops():
    log = []

    def popen(argv):
        raise FileNotFoundError(argv[0])

    with pytest.raises(FileNotFoundError):
        launch.run("aider", 8080, False, [], lambda: (Proc(log, "server"), "http://127.0.0.1:8080"),
                   alive=lambda u: False, say=lambda m: None, popen=popen, which=lambda c: c,
                   connect_fn=lambda *a: "ok", undo_fn=lambda t: log.append("undo") or "", model_id=lambda u: "m",
                   was_connected=lambda t: False)
    assert log == ["undo", "server stopped"]


def test_missing_tool_stops_before_anything_starts():
    with pytest.raises(SystemExit, match="not installed"):
        _run(which=False)


def test_is_connected_reads_what_connect_remembers(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALLLM_CONNECT_HOME", str(tmp_path))
    monkeypatch.setattr(connect, "_state_file", lambda: tmp_path / "connect.json")
    assert not connect.is_connected("aider")
    connect.connect("aider", "http://127.0.0.1:8080", "m")
    assert connect.is_connected("aider") and not connect.is_connected("cline")
    connect.undo("aider")
    assert not connect.is_connected("aider")


def test_cli_parses_launch_and_passes_tool_args():
    a = cli.build_parser().parse_args(["launch", "aider", "--keep", "--", "--model", "x"])
    assert a.fn is cli.cmd_launch and a.tool == "aider" and a.keep and a.args[-2:] == ["--model", "x"]
