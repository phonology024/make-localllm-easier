"""localllm as an MCP server: Claude, Cursor, Codex, ... hand private or bulk work to the model on this computer.

  localllm mcp [--model KEY] [--port 8080]       stdio server; AI apps launch it themselves (README "Use it from an AI app")

Nothing leaves this computer: the tools talk to the localllm gateway on 127.0.0.1 (OpenAI /v1/chat/completions and
GET /localllm/status). No telemetry, no other network calls. A gateway on another machine is used only when the person
who configures the server sets LOCALLLM_MCP_ALLOW_REMOTE=1 in its environment; a tool argument can never do that.

If nothing answers on the port, the server starts `localllm serve KEY` itself, but only for a model named with --model
that is already on disk (and a llama-server already installed): it never downloads anything. Otherwise the tools say
to run `localllm` once first.

Needs Python 3.10+ (the mcp SDK): pip install "make-localllm-easier[mcp]"."""

import io
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

HEARTBEAT = 5          # seconds between progress notifications while the model is answering
EMPTY_RETRIES = 2      # extra tries when the model ends its turn with no text at all
START_WAIT = 300       # seconds to wait for an auto-started server (model load) before giving up
DEFAULT_URL = "http://127.0.0.1:{port}"
INSTALL_HINT = 'localllm mcp needs Python 3.10+ and the MCP SDK: pip install "make-localllm-easier[mcp]"'
LENGTHS = {"short": "in one or two sentences", "medium": "in one short paragraph", "long": "in several paragraphs"}


def _guard_stdout():
    """stdout is the JSON-RPC channel. Keep a private handle to it for the protocol and point everything else
    (print, logging, stray library output) at stderr so it can never corrupt a message."""
    proto = io.TextIOWrapper(os.fdopen(os.dup(sys.stdout.fileno()), "wb"), encoding="utf-8", newline="\n",
                             write_through=True)
    sys.stdout = sys.stderr
    return proto


async def _report_progress(ctx, progress, total, message) -> None:
    """Send an MCP progress notification when a client request is in flight. Calling a tool directly (tests,
    scripts) has no request context: ctx.report_progress raises ValueError then, which must not abort the answer.
    Annotations such as `float | None` stay out of module-level signatures: only `build` needs the SDK."""
    if ctx is None:
        return
    try:
        await ctx.report_progress(progress, total, message)
    except ValueError:
        pass


def _tool_error(e: BaseException):
    """ToolError, because the SDK only forwards the text of a ToolError; any other exception reaches the client as a
    bare "Error executing tool" and the assistant could not tell the user what to do."""
    from mcp.server.mcpserver.exceptions import ToolError
    text = str(e) or "localllm stopped without a message"
    return ToolError(text if isinstance(e, (SystemExit, OSError, ValueError, RuntimeError)) else
                     f"{type(e).__name__}: {text}")


class NoModel(RuntimeError):
    """No local model is answering and none can be started without the person's say-so."""


def remote_allowed() -> bool:
    return os.environ.get("LOCALLLM_MCP_ALLOW_REMOTE", "").strip().lower() in ("1", "true", "yes")


def check_url(url: str) -> str:
    """The gateway to use. Loopback unless LOCALLLM_MCP_ALLOW_REMOTE=1 is in the environment of this process."""
    from .gateway import is_loopback
    host = urllib.parse.urlsplit(url).hostname or ""
    if not is_loopback(host) and not remote_allowed():
        raise ValueError(f"{url} is not on this computer. localllm mcp only talks to 127.0.0.1; to use a server on "
                         "your network anyway, set LOCALLLM_MCP_ALLOW_REMOTE=1 in this MCP server's environment")
    return url.rstrip("/")


def _headers() -> dict:
    key = os.environ.get("LOCALLLM_API_KEY")
    return {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}


def _get_json(url: str, timeout: float = 5):
    try:
        req = urllib.request.Request(url, headers=_headers())
        return json.load(urllib.request.urlopen(req, timeout=timeout))
    except (OSError, ValueError):
        return None


def alive(url: str) -> bool:
    try:
        return b'"ok"' in urllib.request.urlopen(url + "/health", timeout=2).read()
    except OSError:
        return False


def _kill_tree(proc) -> None:
    """Stop `proc` and everything it started. `localllm serve` runs llama-server as a child; killing only the parent
    leaves the child alive and holding VRAM. Windows: taskkill /T. Elsewhere: the child leads its own session
    (start_new_session), so the whole group gets SIGTERM."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=30)
        return
    import signal
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        proc.terminate()


class Backend:
    """The gateway the tools talk to, and (only if asked with --model) the `localllm serve` this process started."""

    def __init__(self, url: str, model: "str | None" = None, port: int = 8080, spawn=subprocess.Popen):
        self.url, self.model, self.port, self._spawn = check_url(url), model, port, spawn
        self.proc = None
        self._job = None
        self.lock = threading.Lock()
        self.last = {}          # timings of the last answer, for status()

    # -- start --------------------------------------------------------------------------------------------------
    def _cannot_start(self) -> str:
        if self.model is None:
            return ("no local model is running. Run `localllm` once in a terminal (it picks the best model for this "
                    "PC and downloads it after asking), then try again. To let this server start it by itself, add "
                    "`--model KEY` (see `localllm list`) to its arguments once that model is downloaded.")
        from . import catalog
        if self.model not in catalog.MODELS:
            return f"unknown model {self.model!r}; `localllm list` shows the keys"
        from . import cli, runtime
        if not cli._have_model(self.model):
            return (f"{self.model} is not downloaded. Run `localllm serve {self.model}` once in a terminal "
                    "(it asks before downloading), then try again.")
        if runtime.installed_server() is None:
            return "llama-server is not installed yet. Run `localllm` once in a terminal, then try again."
        return ""

    def ensure(self, on_wait=None) -> None:
        """Make sure a server answers; start one only for --model KEY that is on disk. Never downloads."""
        with self.lock:            # one check inside the lock: a refused connection takes ~2 s on Windows
            if alive(self.url):
                return
            from .gateway import is_loopback
            if not is_loopback(urllib.parse.urlsplit(self.url).hostname or ""):
                raise NoModel(f"the localllm server at {self.url} is not answering")
            why = self._cannot_start()
            if why:
                raise NoModel(why)
            from . import runtime
            if self.proc is None or self.proc.poll() is not None:
                kw = {"creationflags": 0x08000000} if os.name == "nt" else {"start_new_session": True}   # NO_WINDOW
                self.proc = self._spawn([sys.executable, "-m", "localllm", "serve", self.model, "--no-ui", "--port",
                                         str(self.port)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL, **kw)
                self._job = runtime.kill_with_parent(self.proc)
            t0 = time.time()
            while time.time() - t0 < START_WAIT:
                if alive(self.url):
                    return
                if self.proc.poll() is not None:
                    raise NoModel(f"`localllm serve {self.model}` exited with code {self.proc.returncode}; run it in "
                                  "a terminal to see why (`localllm doctor --fix` checks the usual causes)")
                if on_wait:
                    on_wait(f"starting {self.model}")        # the heartbeat adds the elapsed time
                time.sleep(1)
            raise NoModel(f"{self.model} did not answer within {START_WAIT} s; run `localllm serve {self.model}` "
                          "in a terminal to see what it is doing")

    def close(self) -> None:
        """Stop the server this process started, llama-server included: terminate() alone would leave it running
        (and holding VRAM) once its parent is gone, on Windows always."""
        if self.proc is not None and self.proc.poll() is None:
            _kill_tree(self.proc)
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self._job = None

    # -- one completion -----------------------------------------------------------------------------------------
    def complete(self, messages: list, max_tokens: int, tick=None) -> dict:
        """One chat completion, streamed. tick(n_chunks) is called as text arrives (progress). Returns
        {text, model, tokens, tok_s, truncated}.

        A model sometimes stops at once with an empty answer (a few characters of "thinking", then the end token);
        that is not a result. It is asked again, up to EMPTY_RETRIES times, with a new random seed and without the
        prompt cache so the retry is not the same draw. An empty answer that ran out of max_tokens is not retried:
        asking again would spend the same tokens thinking."""
        for attempt in range(EMPTY_RETRIES + 1):
            res = self._stream(messages, max_tokens, tick, retry=attempt)
            if res["text"] or res["truncated"]:
                break
        if not res["text"]:
            raise RuntimeError("the model returned no text" + (
                f" within {max_tokens} tokens (it may have spent them thinking): ask again with a larger max_tokens"
                if res["truncated"] else f" in {attempt + 1} tries: ask again, or try a different wording"))
        if res["truncated"]:
            res["note"] = f"stopped at max_tokens={max_tokens}; ask again with a larger max_tokens for the rest"
        if attempt:
            res["retries"] = attempt
        return res

    def _stream(self, messages: list, max_tokens: int, tick, retry: int = 0) -> dict:
        body = {"messages": messages, "max_tokens": max_tokens, "stream": True}
        if retry:
            body.update(seed=random.randrange(2**31), cache_prompt=False)
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers=_headers())
        text, timings, finish, n, model = [], {}, None, 0, None
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                model = r.headers.get("X-Localllm-Model")
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    d = json.loads(line[5:])
                    timings = d.get("timings", timings)
                    for c in d.get("choices") or []:
                        delta = c.get("delta", {})
                        piece = delta.get("content") or ""
                        if piece or delta.get("reasoning_content"):
                            n += 1
                            if tick:
                                tick(n)
                        text.append(piece)
                        finish = c.get("finish_reason") or finish
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:400]
            raise RuntimeError(f"the local server refused the request (HTTP {e.code}): {detail}") from None
        except OSError as e:
            raise NoModel(f"lost the connection to the local server at {self.url}: {e}") from None
        tok_s = timings.get("predicted_per_second")
        if not tok_s and timings.get("predicted_ms"):
            tok_s = timings.get("predicted_n", 0) * 1000 / timings["predicted_ms"]
        self.last = {"model": model, "tok_s": round(tok_s, 1) if tok_s else None}
        return {"text": "".join(text).strip(), "model": model or self.model, "tokens": timings.get("predicted_n"),
                "tok_s": self.last["tok_s"], "truncated": finish == "length"}

    # -- status -------------------------------------------------------------------------------------------------
    def status(self) -> dict:
        """What is running, memory in use, and the speed of the last answer. Read-only: starts nothing."""
        out = {"url": self.url, "running": alive(self.url)}
        st = _get_json(self.url + "/localllm/status") if out["running"] else None
        if not out["running"]:
            from . import catalog, cli
            out["installed_models"] = [k for k in catalog.MODELS if cli._have_model(k)]
            out["hint"] = ("no local model is running; run `localllm` once, or add `--model KEY` to this server's "
                           "arguments to let it start one that is already downloaded")
            return out
        if st is None or "loaded" not in st:
            out["note"] = "this server is running but is not localllm's gateway: no memory figures"
        else:
            out["model"] = st.get("running") or st.get("model")
            out["loaded"] = st.get("loaded")
            for k in ("vram_gb", "ram_gb", "spill_gb"):
                if isinstance(st.get(k), (int, float)):
                    out[k] = round(st[k], 1)
        out["last_answer_tok_s"] = self.last.get("tok_s")        # None until this session has asked something
        return out


def _prompt_translate(text: str, target_lang: str) -> list:
    # "Translate" at the start is what the gateway's --models auto routing looks for (router.TRANSLATE)
    return [{"role": "user", "content": f"Translate the following text into {target_lang}. Reply with the translation "
                                        f"only, no notes.\n\n{text}"}]


def _prompt_summarize(text: str, length: str) -> list:
    how = LENGTHS.get(length) or length          # a free-form length ("3 bullet points") is passed through
    return [{"role": "user", "content": f"Summarize the following text {how}, in the language of the text. Reply with "
                                        f"the summary only.\n\n{text}"}]


def build(backend: "Backend | None" = None):
    import anyio
    from mcp.server.mcpserver import Context, MCPServer
    from mcp.types import ToolAnnotations

    from . import __version__

    be = backend or Backend(os.environ.get("LOCALLLM_URL") or DEFAULT_URL.format(port=8080))
    srv = MCPServer("localllm", version=__version__, instructions=(
        "A language model running on the user's own computer (localllm). Everything you send to these tools stays "
        "on this machine, so use them for private text (contracts, medical or financial notes, source code the user "
        "does not want in the cloud), for bulk work that would use up your own budget (translating or summarizing "
        "many documents, one call per document), and when the user asks for the local model. It is smaller than you: "
        "check its answers on anything that matters. translate and summarize pick the model that is best at the "
        "language when the server runs with --models auto. If a tool says no model is running, tell the user to "
        "run `localllm` once in a terminal; do not try to work around it."))

    async def _run(ctx, messages, max_tokens, what):
        """Stream the answer in a worker thread; every 5 s tell the client how far it is, so it does not time out."""
        count, note = [0], ["starting ..."]
        done = anyio.Event()
        t0 = time.time()

        def work():
            try:
                be.ensure(on_wait=lambda m: note.__setitem__(0, m))
                note[0] = f"{what} ..."
                return be.complete(messages, max_tokens, tick=lambda n: count.__setitem__(0, n))
            except (Exception, SystemExit) as e:        # returned, not raised: see _tool_error in babelscribe #6
                return e

        async def heartbeat():
            while not done.is_set():
                with anyio.move_on_after(HEARTBEAT):
                    await done.wait()
                if not done.is_set():
                    more = f" - {count[0]} chunks so far" if count[0] > 0 else ""
                    await _report_progress(ctx, time.time() - t0, None, f"{note[0]}{more} ({time.time() - t0:.0f} s)")

        async with anyio.create_task_group() as tg:
            tg.start_soon(heartbeat)
            try:
                result = await anyio.to_thread.run_sync(work)
            finally:
                done.set()
        if isinstance(result, BaseException):
            raise _tool_error(result) from None
        await _report_progress(ctx, 1, 1, "done")
        return result

    ro = dict(read_only_hint=True, destructive_hint=False, idempotent_hint=False, open_world_hint=False)

    @srv.tool(title="Ask the local model", annotations=ToolAnnotations(**ro))
    async def ask_local_model(prompt: str, system: str = "", max_tokens: int = 1024, ctx: Context | None = None) -> dict:
        """Runs on this computer on the local model; nothing leaves it. Ask the language model running on the user's own computer (localllm) one question and get its
        answer. Use it for private or bulk work you do not want to send to a cloud service.

        prompt: the question or task, with any text it needs inside it.
        system: optional instructions for how to answer, e.g. "answer in Thai".
        max_tokens: the longest answer in tokens (1024 is about 700 words); a reasoning model also spends these on thinking.
        Returns text, the model that answered, tokens, tok_s, and truncated=true if it ran out of max_tokens.
        If no model is running it says so: tell the user to run `localllm` once."""
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return await _run(ctx, msgs, max_tokens, "answering")

    @srv.tool(title="Translate with the local model", annotations=ToolAnnotations(**ro))
    async def translate(text: str, target_lang: str, max_tokens: int = 4096, ctx: Context | None = None) -> dict:
        """Runs on this computer on the local model; nothing leaves it. Translate text into another language with the model on the user's own computer; with
        `localllm --models auto` it uses the model measured best for that language.

        text: what to translate (one document per call for bulk work).
        target_lang: the language, by name or code: "Thai", "th", "Japanese", "es", ...
        max_tokens: the longest translation in tokens.
        Returns text (the translation only), model, tokens, tok_s, truncated."""
        return await _run(ctx, _prompt_translate(text, target_lang), max_tokens, "translating")

    @srv.tool(title="Summarize with the local model", annotations=ToolAnnotations(**ro))
    async def summarize(text: str, length: str = "medium", max_tokens: int = 1024, ctx: Context | None = None) -> dict:
        """Runs on this computer on the local model; nothing leaves it. Summarize text with the model on the user's own computer, in the language of the text.

        text: what to summarize (one document per call for bulk work).
        length: short (1-2 sentences) | medium (a paragraph) | long (several paragraphs), or your own wording such as
            "3 bullet points".
        max_tokens: the longest summary in tokens.
        Returns text (the summary only), model, tokens, tok_s, truncated."""
        return await _run(ctx, _prompt_summarize(text, length), max_tokens, "summarizing")

    @srv.tool(title="Local model status", annotations=ToolAnnotations(**ro))
    def status() -> dict:
        """Runs on this computer on the local model; nothing leaves it. Is the local model running, which one, and how much video and system memory it uses, plus the
        speed (tokens per second) of the last answer from this session. Starts nothing. If no model is running it
        lists the models already downloaded and says how to start one."""
        return be.status()

    srv._localllm_backend = be
    return srv


def main(model: "str | None" = None, port: int = 8080, url: "str | None" = None) -> None:
    """Entry point of `localllm mcp`. Anything wrong with the setup is printed to stderr as a plain sentence."""
    try:
        import anyio
        from mcp.server.stdio import stdio_server
        import mcp.server.mcpserver  # noqa: F401  (2.x layout: this is what `build` needs)
    except ImportError:
        raise SystemExit(INSTALL_HINT) from None
    proto = _guard_stdout()
    try:
        backend = Backend(url or os.environ.get("LOCALLLM_URL") or DEFAULT_URL.format(port=port), model, port)
    except ValueError as e:
        raise SystemExit(f"[localllm] {e}") from None
    srv = build(backend)

    async def run():
        async with stdio_server(stdout=anyio.wrap_file(proto)) as (r, w):
            low = srv._lowlevel_server
            await low.run(r, w, low.create_initialization_options())

    try:
        anyio.run(run)
    finally:
        backend.close()          # a server this process started stops with it


if __name__ == "__main__":
    main()
