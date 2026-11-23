"""`localllm mcp` (#183): tool list, a call against a stubbed server, no-model errors, localhost-only, progress."""
import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("mcp.server.mcpserver")        # the SDK is an optional extra and needs Python 3.10+

from localllm import cli, mcp_server  # noqa: E402


class Stub(BaseHTTPRequestHandler):
    """A localllm gateway in miniature: /health, /localllm/status and a streaming /v1/chat/completions."""
    protocol_version = "HTTP/1.0"
    seen: list = []
    slow = 0.0          # seconds between chunks
    finish_reason = "stop"
    reply = ["Hello ", "from ", "the local model"]
    empties = 0         # the next N answers are empty: a few characters of "thinking", then the end token
    empty_finish = "stop"

    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {"status": "ok"})
        if self.path == "/localllm/status":
            return self._json(200, {"model": "stub-model", "loaded": True, "running": "qwen3.5-4b-q4",
                                    "vram_gb": 3.456, "ram_gb": 1.2, "spill_gb": 0.0})
        self._json(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Stub.seen.append((self.path, body))
        if body["messages"][-1]["content"] == "too long":
            return self._json(400, {"error": {"message": "prompt exceeds the context size"}})
        self.send_response(200); self.send_header("Content-Type", "text/event-stream")
        self.send_header("X-Localllm-Model", "qwen3.5-4b-q4"); self.end_headers()
        if Stub.empties > 0:
            Stub.empties -= 1
            for c in ({"choices": [{"delta": {"reasoning_content": "Thinking\n\n"}}]},
                      {"choices": [{"delta": {}, "finish_reason": Stub.empty_finish}], "timings": {"predicted_n": 3}}):
                self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
            return
        for piece in Stub.reply:
            time.sleep(Stub.slow)
            self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': piece}}]}, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()
        last = {"choices": [{"delta": {}, "finish_reason": Stub.finish_reason}],
                "timings": {"predicted_n": 3, "predicted_per_second": 42.04}}
        self.wfile.write(f"data: {json.dumps(last)}\n\ndata: [DONE]\n\n".encode())


@pytest.fixture
def stub():
    Stub.seen, Stub.slow, Stub.finish_reason, Stub.reply = [], 0.0, "stop", ["Hello ", "from ", "the local model"]
    Stub.empties, Stub.empty_finish = 0, "stop"
    s = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_port}"
    s.shutdown()


def _free_url():
    """A loopback URL nothing listens on."""
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{s.getsockname()[1]}"


def call(backend, tool, args=None, **kw):
    """Run one tool through the real MCP protocol with an in-process client."""
    from mcp.client import Client

    async def go():
        async with Client(mcp_server.build(backend)) as c:
            return await c.call_tool(tool, args or {}, **kw)
    return asyncio.run(go())


def data(r):
    """The JSON the tool returned (clients read it from the text content)."""
    assert not r.is_error, r.content[0].text
    return json.loads(r.content[0].text)


def test_tool_list_and_descriptions_say_nothing_leaves_the_computer():
    from mcp.client import Client

    async def go():
        async with Client(mcp_server.build(mcp_server.Backend("http://127.0.0.1:1"))) as c:
            return (await c.list_tools()).tools
    tools = {t.name: t for t in asyncio.run(go())}
    assert set(tools) == {"ask_local_model", "translate", "summarize", "status"}
    for name in ("ask_local_model", "translate", "summarize", "status"):
        assert "nothing leaves it" in tools[name].description, name
        assert tools[name].annotations.read_only_hint is True and tools[name].annotations.open_world_hint is False
    assert set(tools["ask_local_model"].input_schema["properties"]) == {"prompt", "system", "max_tokens"}
    assert set(tools["translate"].input_schema["properties"]) == {"text", "target_lang", "max_tokens"}
    assert set(tools["summarize"].input_schema["properties"]) == {"text", "length", "max_tokens"}
    assert tools["ask_local_model"].input_schema["required"] == ["prompt"]


def test_ask_local_model_streams_from_the_stub_and_reports_the_answer(stub):
    r = call(mcp_server.Backend(stub), "ask_local_model", {"prompt": "hi", "system": "be brief", "max_tokens": 50})
    assert not r.is_error
    out = data(r)
    assert out["text"] == "Hello from the local model" and out["model"] == "qwen3.5-4b-q4"
    assert out["tokens"] == 3 and out["tok_s"] == 42.0 and out["truncated"] is False
    path, body = Stub.seen[-1]
    assert path == "/v1/chat/completions" and body["stream"] is True and body["max_tokens"] == 50
    assert body["messages"] == [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]


def test_translate_and_summarize_ask_in_words_the_gateway_router_recognises(stub):
    from localllm import router
    be = mcp_server.Backend(stub)
    call(be, "translate", {"text": "Good morning", "target_lang": "Thai"})
    sent = Stub.seen[-1][1]["messages"][-1]["content"]
    assert router.detect_task(sent) == "translate" and "into Thai" in sent and sent.endswith("Good morning")
    call(be, "summarize", {"text": "A long article.", "length": "short"})
    sent = Stub.seen[-1][1]["messages"][-1]["content"]
    assert router.SUMMARIZE.search(sent[:200]) and "one or two sentences" in sent and sent.endswith("A long article.")
    call(be, "summarize", {"text": "x", "length": "3 bullet points"})
    assert "3 bullet points" in Stub.seen[-1][1]["messages"][-1]["content"]


def test_thai_and_other_non_ascii_text_survives_the_stream(stub):
    Stub.reply = ["อรุณ", "สวัสดิ์ ", "おはよう"]
    out = data(call(mcp_server.Backend(stub), "translate", {"text": "Good morning", "target_lang": "Thai"}))
    assert out["text"] == "อรุณสวัสดิ์ おはよう"


def test_a_truncated_answer_says_so(stub):
    Stub.finish_reason = "length"
    out = data(call(mcp_server.Backend(stub), "ask_local_model", {"prompt": "hi", "max_tokens": 3}))
    assert out["truncated"] is True and "larger max_tokens" in out["note"]


def test_the_servers_refusal_comes_back_as_a_readable_error(stub):
    r = call(mcp_server.Backend(stub), "ask_local_model", {"prompt": "too long"})
    assert r.is_error and "HTTP 400" in r.content[0].text and "context size" in r.content[0].text


def test_no_model_running_says_run_localllm_first(monkeypatch):
    monkeypatch.setattr(cli, "_have_model", lambda key: None)
    r = call(mcp_server.Backend(_free_url()), "ask_local_model", {"prompt": "hi"})
    assert r.is_error
    text = r.content[0].text
    assert "Run `localllm` once" in text and "Traceback" not in text and "TaskGroup" not in text


def test_model_flag_that_is_not_downloaded_is_not_started_or_downloaded(monkeypatch):
    started = []
    monkeypatch.setattr(cli, "_have_model", lambda key: None)
    be = mcp_server.Backend(_free_url(), model="qwen3.5-4b-q4", spawn=lambda *a, **k: started.append(a))
    r = call(be, "translate", {"text": "hi", "target_lang": "th"})
    assert r.is_error and "is not downloaded" in r.content[0].text and "localllm serve qwen3.5-4b-q4" in r.content[0].text
    assert started == []


def test_model_flag_without_llama_server_installed_does_not_fetch_it(monkeypatch):
    from localllm import runtime
    monkeypatch.setattr(cli, "_have_model", lambda key: "/models/x.gguf")
    monkeypatch.setattr(runtime, "installed_server", lambda: None)
    monkeypatch.setattr(runtime, "fetch_server", lambda: pytest.fail("must not download llama-server"))
    be = mcp_server.Backend(_free_url(), model="qwen3.5-4b-q4", spawn=lambda *a, **k: pytest.fail("no spawn"))
    r = call(be, "ask_local_model", {"prompt": "hi"})
    assert r.is_error and "llama-server is not installed" in r.content[0].text


def test_model_flag_on_disk_starts_serve_with_no_stdin_and_stops_it_with_the_server(monkeypatch, stub):
    """The child is `localllm serve KEY --port N`; its stdin is closed (stdout is the MCP channel, stdin is not ours to
    share) and it is stopped by close()."""
    from localllm import runtime
    monkeypatch.setattr(cli, "_have_model", lambda key: "/models/x.gguf")
    monkeypatch.setattr(runtime, "installed_server", lambda: "/bin/llama-server")
    calls = []

    class Child:
        returncode = None
        pid = 4242
        def poll(self): return None
        def wait(self, t=None): return 0

    child = Child()
    killed = []
    monkeypatch.setattr(mcp_server, "_kill_tree", lambda proc: killed.append(proc.pid))
    monkeypatch.setattr(runtime, "kill_with_parent", lambda proc: None)

    def spawn(cmd, **kw):
        calls.append((cmd, kw))
        return child
    # the stub answers /health only after the "spawn" (stand-in for the model loading)
    real_alive, state = mcp_server.alive, {"n": 0}

    def alive(url):
        state["n"] += 1
        return state["n"] > 2 and real_alive(stub)
    monkeypatch.setattr(mcp_server, "alive", alive)
    be = mcp_server.Backend(stub, model="qwen3.5-4b-q4", port=8123, spawn=spawn)
    be.ensure()
    cmd, kw = calls[0]
    assert cmd[1:] == ["-m", "localllm", "serve", "qwen3.5-4b-q4", "--no-ui", "--port", "8123"]
    assert kw["stdin"] == subprocess.DEVNULL and kw["stdout"] == subprocess.DEVNULL
    be.close()
    assert killed == [4242]            # the whole tree, not just `localllm serve` (see the orphan tests below)


def test_a_server_that_is_up_is_used_as_is_and_nothing_is_started(stub):
    be = mcp_server.Backend(stub, model="qwen3.5-4b-q4", spawn=lambda *a, **k: pytest.fail("must not start"))
    be.ensure()
    assert be.proc is None


def test_only_localhost_unless_the_person_who_configures_the_server_allows_it(monkeypatch):
    monkeypatch.delenv("LOCALLLM_MCP_ALLOW_REMOTE", raising=False)
    for ok in ("http://127.0.0.1:8080", "http://localhost:8080/", "http://[::1]:8080", "http://[::ffff:127.0.0.1]:1"):
        assert mcp_server.check_url(ok)
    for bad in ("http://192.168.1.20:8080", "https://example.com", "http://0.0.0.0:8080"):
        with pytest.raises(ValueError, match="LOCALLLM_MCP_ALLOW_REMOTE=1"):
            mcp_server.check_url(bad)
    monkeypatch.setenv("LOCALLLM_MCP_ALLOW_REMOTE", "1")
    assert mcp_server.check_url("http://192.168.1.20:8080") == "http://192.168.1.20:8080"
    monkeypatch.setenv("LOCALLLM_API_KEY", "k123")
    assert mcp_server._headers()["Authorization"] == "Bearer k123"


def test_no_tool_takes_a_url_so_an_assistant_cannot_redirect_the_text():
    from mcp.client import Client

    async def go():
        async with Client(mcp_server.build(mcp_server.Backend("http://127.0.0.1:1"))) as c:
            return (await c.list_tools()).tools
    for t in asyncio.run(go()):
        assert not {"url", "host", "endpoint", "base_url", "port"} & set(t.input_schema["properties"]), t.name


def test_status_reads_the_gateway_and_starts_nothing(stub):
    be = mcp_server.Backend(stub, model="qwen3.5-4b-q4", spawn=lambda *a, **k: pytest.fail("status must not start"))
    s = data(call(be, "status"))
    assert s["running"] is True and s["model"] == "qwen3.5-4b-q4" and s["vram_gb"] == 3.5 and s["ram_gb"] == 1.2
    assert s["last_answer_tok_s"] is None            # nothing asked yet: no number made up
    call(be, "ask_local_model", {"prompt": "hi"})
    assert data(call(be, "status"))["last_answer_tok_s"] == 42.0


def test_status_with_no_server_lists_what_is_downloaded(monkeypatch):
    monkeypatch.setattr(cli, "_have_model", lambda key: "x" if key == "qwen3.5-4b-q4" else None)
    s = data(call(mcp_server.Backend(_free_url()), "status"))
    assert s["running"] is False and s["installed_models"] == ["qwen3.5-4b-q4"] and "localllm" in s["hint"]


def test_a_slow_answer_sends_progress_notifications(stub, monkeypatch):
    monkeypatch.setattr(mcp_server, "HEARTBEAT", 0.1)
    Stub.slow = 0.25
    seen = []

    async def on_progress(progress, total, message):
        seen.append(message)
    r = call(mcp_server.Backend(stub), "ask_local_model", {"prompt": "hi"}, progress_callback=on_progress)
    assert not r.is_error and len(seen) >= 2 and seen[-1] == "done"
    assert any("answering" in m for m in seen)


def test_stdout_is_kept_for_the_protocol(tmp_path, monkeypatch):
    real = open(tmp_path / "stdout.txt", "wb")
    monkeypatch.setattr(sys, "stdout", real)
    proto = mcp_server._guard_stdout()
    print("stray library output")                 # goes to stderr now, never into the JSON-RPC stream
    assert sys.stdout is sys.stderr
    proto.write("{}\n")
    proto.close()
    real.close()
    assert (tmp_path / "stdout.txt").read_bytes() == b"{}\n"


def test_cli_registers_the_mcp_command():
    a = cli.parse_args(["mcp", "--model", "qwen3.5-4b-q4", "--port", "9000"])
    assert a.fn is cli.cmd_mcp and a.model == "qwen3.5-4b-q4" and a.port == 9000 and a.url is None


def test_without_the_sdk_the_command_prints_how_to_install_it(monkeypatch):
    import builtins
    real = builtins.__import__

    def no_mcp(name, *a, **k):
        if name == "mcp" or name.startswith("mcp."):
            raise ImportError(name)
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_mcp)
    with pytest.raises(SystemExit) as e:
        mcp_server.main()
    assert "pip install" in str(e.value) and "make-localllm-easier[mcp]" in str(e.value)


def test_real_stdio_process_end_to_end(stub):
    """`python -m localllm mcp` as an AI app starts it: JSON-RPC over stdio, the answer comes from the stub."""
    from mcp.client import Client
    from mcp.client.stdio import StdioServerParameters

    async def go():
        params = StdioServerParameters(command=sys.executable, args=["-m", "localllm", "mcp", "--url", stub])
        async with Client(params) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            r = await c.call_tool("translate", {"text": "Good morning", "target_lang": "Thai"})
            return names, r
    names, r = asyncio.run(go())
    assert names == {"ask_local_model", "translate", "summarize", "status"}
    assert json.loads(r.content[0].text)["text"] == "Hello from the local model"


# --- distribution: the registry entry, the .mcpb and the README must agree with each other and with the server ----------
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_registry_entry_and_extension_name_the_same_tools_and_version():
    reg = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
    ext = json.loads((ROOT / "mcpb" / "manifest.json").read_text(encoding="utf-8"))
    pkg = reg["packages"][0]
    assert reg["version"] == pkg["version"] == ext["version"]
    assert (ROOT / "mcpb" / "pyproject.toml").read_text(encoding="utf-8").count(f'version = "{reg["version"]}"') == 1
    assert pkg["identifier"] == "make-localllm-easier" and pkg["packageArguments"] == [{"type": "positional", "value": "mcp"}]
    assert pkg["runtimeArguments"] == [{"type": "named", "name": "--from", "value": "make-localllm-easier[mcp]"}]
    assert len(reg["description"]) <= 100                      # the registry's limit
    # the registry checks ownership of a PyPI package by this line in the README PyPI shows
    assert f"mcp-name: {reg['name']}" in (ROOT / "README.md").read_text(encoding="utf-8")
    from mcp.client import Client

    async def go():
        async with Client(mcp_server.build(mcp_server.Backend("http://127.0.0.1:1"))) as c:
            return {t.name for t in (await c.list_tools()).tools}
    assert {t["name"] for t in ext["tools"]} == asyncio.run(go())
    assert (ROOT / "mcpb" / ext["icon"]).exists() and (ROOT / "mcpb" / ext["server"]["entry_point"]).exists()


def test_the_mcp_extra_is_declared_and_the_package_still_has_no_hard_dependency():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "dependencies = []" in text and "\nmcp = [" in text


def test_the_docs_cover_every_client_the_issue_names():
    doc = (ROOT / "docs" / "mcp.md").read_text(encoding="utf-8")
    for client in ("Claude Desktop", "Claude Code", "Cursor", "VS Code", "Gemini CLI", "Codex", "tool_timeout_sec"):
        assert client in doc, client


# --- an empty answer is retried, with a different draw (#183 release test: Thai translate on a 2-bit model) -------------

def test_an_empty_answer_is_asked_again_with_a_new_seed_and_no_prompt_cache(stub):
    Stub.empties = 1
    out = data(call(mcp_server.Backend(stub), "translate", {"text": "Good morning", "target_lang": "Thai"}))
    assert out["text"] == "Hello from the local model" and out["retries"] == 1
    first, second = Stub.seen[-2][1], Stub.seen[-1][1]
    assert "seed" not in first and "cache_prompt" not in first
    assert isinstance(second["seed"], int) and second["cache_prompt"] is False
    assert second["messages"] == first["messages"]


def test_the_model_that_keeps_answering_nothing_gets_a_clear_error_after_three_tries(stub):
    Stub.empties = 99
    r = call(mcp_server.Backend(stub), "ask_local_model", {"prompt": "hi"})
    assert r.is_error and "no text in 3 tries" in r.content[0].text
    assert len(Stub.seen) == 1 + mcp_server.EMPTY_RETRIES


def test_an_empty_answer_that_ran_out_of_tokens_is_not_asked_again(stub):
    Stub.empties, Stub.empty_finish = 99, "length"
    r = call(mcp_server.Backend(stub), "ask_local_model", {"prompt": "hi", "max_tokens": 5})
    assert r.is_error and "larger max_tokens" in r.content[0].text
    assert len(Stub.seen) == 1


def test_a_good_first_answer_is_asked_once_and_reports_no_retries(stub):
    out = data(call(mcp_server.Backend(stub), "ask_local_model", {"prompt": "hi"}))
    assert "retries" not in out and len(Stub.seen) == 1


def test_the_wait_note_does_not_repeat_the_time_the_heartbeat_adds(monkeypatch):
    from localllm import runtime
    monkeypatch.setattr(cli, "_have_model", lambda key: "/models/x.gguf")
    monkeypatch.setattr(runtime, "installed_server", lambda: "/bin/llama-server")
    monkeypatch.setattr(mcp_server.time, "sleep", lambda s: None)
    monkeypatch.setattr(mcp_server, "START_WAIT", 0.05)
    monkeypatch.setattr(runtime, "kill_with_parent", lambda proc: None)

    class Child:
        pid, returncode = 1, None
        def poll(self): return None
    notes = []
    be = mcp_server.Backend(_free_url(), model="qwen3.5-4b-q4", spawn=lambda *a, **k: Child())
    with pytest.raises(mcp_server.NoModel):
        be.ensure(on_wait=notes.append)
    assert notes and all(n == "starting qwen3.5-4b-q4" for n in notes)


# --- no llama-server left behind (#183 release test: orphans kept VRAM after the MCP client closed) --------------------

GRANDCHILD = ("import subprocess, sys, time\n"
              "time.sleep(1)                # the job object is put on this process before it starts its own child\n"
              "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
              "open(sys.argv[1], 'w').write(str(p.pid))\n"
              "time.sleep(600)\n")


def _pid_alive(pid):
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        return os.waitpid(pid, os.WNOHANG) == (0, 0)       # a zombie is not running
    except ChildProcessError:
        return True


def _wait_for(cond, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.1)
    return False


def _force_kill(pid):
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True)
    elif _pid_alive(pid):
        os.kill(pid, 9)


def test_close_stops_the_server_and_the_llama_server_under_it(tmp_path):
    """A stand-in for `localllm serve` that has a child of its own, started the way Backend.ensure starts it."""
    marker = tmp_path / "grandchild.pid"
    kw = {"creationflags": 0x08000000} if os.name == "nt" else {"start_new_session": True}
    proc = subprocess.Popen([sys.executable, "-c", GRANDCHILD, str(marker)], **kw)
    grandchild = None
    try:
        assert _wait_for(lambda: marker.exists() and marker.read_text()), "the stand-in never started its child"
        grandchild = int(marker.read_text())
        assert _pid_alive(grandchild)
        be = mcp_server.Backend("http://127.0.0.1:1")
        be.proc = proc
        be.close()
        assert proc.poll() is not None
        assert _wait_for(lambda: not _pid_alive(grandchild)), "llama-server was left running"
    finally:
        _force_kill(proc.pid)
        if grandchild:
            _force_kill(grandchild)


@pytest.mark.skipif(os.name != "nt", reason="job objects are a Windows mechanism; elsewhere start_new_session + killpg")
def test_a_client_that_kills_the_mcp_server_outright_leaves_no_llama_server_either(tmp_path):
    """The MCP process dies without running any cleanup (a client that kills it): Windows kills the job's processes."""
    marker, pidfile = tmp_path / "grandchild.pid", tmp_path / "child.pid"
    middle = ("import os, subprocess, sys\n"
              f"sys.path.insert(0, {str(ROOT)!r})\n"
              "from localllm import runtime\n"
              f"p = subprocess.Popen([sys.executable, '-c', {GRANDCHILD!r}, {str(marker)!r}], creationflags=0x08000000)\n"
              "job = runtime.kill_with_parent(p)\n"
              "assert job, 'no job object'\n"
              f"open({str(pidfile)!r}, 'w').write(str(p.pid))\n"
              f"while not os.path.exists({str(marker)!r}) or not open({str(marker)!r}).read(): pass\n"
              "os._exit(0)\n")                       # no finally, no atexit: nothing of ours runs
    subprocess.run([sys.executable, "-c", middle], check=True, timeout=60)
    child, grandchild = int(pidfile.read_text()), int(marker.read_text())
    try:
        assert _wait_for(lambda: not _pid_alive(grandchild)), "the child of `localllm serve` was left running"
        assert not _pid_alive(child)
    finally:
        _force_kill(child)
        _force_kill(grandchild)


def test_llama_server_is_started_inside_the_kill_with_parent_job(monkeypatch):
    """`localllm serve` killed outright (Stop-Process, a client closing) must not leave llama-server holding VRAM."""
    from localllm import runtime
    jobbed = []

    class P:
        pid = 1

        def poll(self):
            return None
    monkeypatch.setattr(cli, "_model_path", lambda k: cli.Path("m.gguf"))
    monkeypatch.setattr(runtime, "ram_gb", lambda: 32.0)
    monkeypatch.setattr(cli.subprocess, "Popen", lambda args, **k: P())
    monkeypatch.setattr(runtime, "kill_with_parent", lambda proc: jobbed.append(type(proc)))
    monkeypatch.setattr(cli, "_free_port", lambda: 1)
    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(SystemExit("stop")))
    with pytest.raises(SystemExit):
        cli._launch("qwen3.5-4b-q4", cli.Path("llama-server"), {"id": "Vulkan0", "total_gb": 16.0, "name": "x"}, 8192, 20.0)
    assert jobbed == [P]


def test_release_files_pin_one_version_and_the_tools_that_publish_are_pinned():
    import re
    import localllm
    reg = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1) == localllm.__version__ == reg["version"]
    assert 'requires-python = ">=3.10"' in pyproject            # 3.9 cannot even import localllm.catalog
    flows = {n: (ROOT / ".github" / "workflows" / f"{n}.yml").read_text(encoding="utf-8") for n in ("publish", "release")}
    assert re.search(r"@anthropic-ai/mcpb@\d+\.\d+\.\d+ pack", flows["release"])
    assert re.search(r"releases/download/v\d+\.\d+\.\d+/mcp-publisher", flows["publish"])
    assert "releases/latest/download" not in flows["publish"]
