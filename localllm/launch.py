"""`localllm launch <tool>`: one command for a coding agent on the local model (#117).

  localllm launch opencode|aider|hermes|claude-code|codex [--keep] [--port 8080] [-- ARGS FOR THE TOOL]

1. uses the localllm already answering on --port, or starts one with `--profile max` (room for long agent prompts)
2. `localllm connect <tool>`: the tool's config points at it (the old file is backed up)
3. starts the tool in this terminal and waits for it
4. on exit, however it ends: puts the old config back (unless --keep, or the tool was already connected before) and
   stops the server it started (a server that was already running is left alone)
"""
from __future__ import annotations

import shutil
import subprocess

from . import connect

# tool -> the command that starts it (codex reads the profile `connect` writes)
COMMANDS = {"opencode": ["opencode"], "aider": ["aider"], "hermes": ["hermes"], "claude-code": ["claude"],
            "codex": ["codex", "--profile", "localllm"]}


def run(tool: str, port: int, keep: bool, extra: list[str], start_server, alive, say=print,
        popen=subprocess.Popen, which=shutil.which, connect_fn=connect.connect, undo_fn=connect.undo,
        model_id=connect.model_id, was_connected=connect.is_connected) -> int:
    """Returns the tool's exit code. start_server() -> (proc, url) starts localllm; alive(url) -> bool."""
    cmd = COMMANDS[tool]
    exe = which(cmd[0])
    if not exe:
        raise SystemExit(f"[localllm] `{cmd[0]}` is not installed (or not on PATH). Install {tool} first; "
                         f"`localllm connect {tool}` shows the config it would get.")
    url, server = f"http://127.0.0.1:{port}", None
    if alive(url):
        say(f"using the localllm already running at {url}")
    else:
        say("starting localllm with --profile max (long agent prompts fit) ...")
        server, url = start_server()
    restore = not keep and not was_connected(tool)
    connected = False
    try:
        say(connect_fn(tool, url, model_id(url)))
        connected = True
        say(f"starting {tool}; quit it to stop" + ("" if server is None else " localllm too"))
        child = popen([exe, *cmd[1:], *extra])
        while True:
            try:
                return child.wait()
            except KeyboardInterrupt:          # Ctrl+C reaches the tool too; let it decide whether to quit
                continue
    finally:
        if connected and restore:
            say(undo_fn(tool))
        if server is not None:
            server.terminate()
