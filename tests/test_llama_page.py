"""#188: without --ui, "/" is llama-server's own web page. That page loads split files (/_app/immutable/bundle.<hash>.js,
icons, manifest), and the gateway answered 404 for them, so the browser showed a blank page."""
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from localllm import gateway, workspace

INDEX = (b'<!doctype html><link rel="icon" href="favicon.ico"><link rel="manifest" href="./manifest.webmanifest">'
         b'<script type="module">import("./_app/immutable/bundle.x.js")</script>')
FILES = {
    "/": (INDEX, "text/html"),
    "/_app/immutable/bundle.x.js": (b"console.log('app')", "text/javascript"),
    "/_app/immutable/assets/bundle.y.css": (b"body{}", "text/css"),
    "/favicon.ico": (b"\x00\x00icon", "image/x-icon"),
    "/favicon.svg": (b"<svg/>", "image/svg+xml"),
    "/manifest.webmanifest": (b"{}", "application/manifest+json"),
    "/apple-touch-icon-180x180.png": (b"png", "image/png"),
    "/apple-splash-portrait-dark-640x1136.png": (b"png", "image/png"),
    "/pwa-192x192.png": (b"png", "image/png"),
    "/maskable-icon-512x512.png": (b"png", "image/png"),
    "/sw.js": (b"//sw", "text/javascript"),
    "/recommended-mcp/exa.ico": (b"ico", "image/x-icon"),
    # llama-server answers these too, but they are not page files: the gateway must not open them
    "/cors-proxy": (b"proxy", "text/plain"),
    "/tools": (b"tools", "application/json"),
    "/metrics": (b"metrics", "text/plain"),
    "/models": (b"[]", "application/json"),
}


class Llama(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        hit = FILES.get(self.path.split("?", 1)[0])
        body, ctype = hit if hit else (b'{"error":"nope"}', "application/json")
        self.send_response(200 if hit else 404)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def upstream():
    s = HTTPServer(("127.0.0.1", 0), Llama)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_port}"
    s.shutdown()


def _get(port, path, key=None, host=None):
    h = {**({"Authorization": f"Bearer {key}"} if key else {}), **({"Host": host} if host else {})}
    return urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers=h), timeout=30)


def _status(port, path, **kw):
    try:
        return _get(port, path, **kw).status
    except urllib.error.HTTPError as e:
        return e.code


PAGE_FILES = [p for p in FILES if p not in ("/cors-proxy", "/tools", "/metrics", "/models")]


def test_the_page_and_every_file_it_loads_come_through_without_ui(upstream):
    g = gateway.serve(upstream, port=0, model_name="m")
    port = g.server_address[1]
    try:
        for path in PAGE_FILES:
            r = _get(port, path)
            assert r.status == 200 and r.read() == FILES[path][0], path
            assert r.headers["Content-Type"] == FILES[path][1], path
        assert _get(port, "/_app/immutable/bundle.x.js?v=3").status == 200      # a cache-busting query
    finally:
        g.shutdown()


@pytest.mark.parametrize("path", ["/cors-proxy", "/tools", "/metrics", "/models", "/nothing-here", "/_app", "/apple",
                                  "/pwa.json", "/localllm/nothing"])
def test_other_llama_server_routes_stay_closed(upstream, path):
    g = gateway.serve(upstream, port=0, model_name="m")
    try:
        assert _status(g.server_address[1], path) == 404
    finally:
        g.shutdown()


def test_with_ui_the_page_is_ours_and_llama_servers_files_are_not_proxied(upstream, tmp_path):
    ui = {"sessions": workspace.Sessions(tmp_path / "s"), "catalog": workspace.catalog_info([], "en", False)}
    g = gateway.serve(upstream, port=0, model_name="m", ui=ui)
    port = g.server_address[1]
    try:
        page = _get(port, "/").read()
        assert page != INDEX and b"localllm" in page.lower()
        assert _get(port, "/manifest.webmanifest").read() != b"{}"               # ours (pwa.py), not llama-server's
        assert _status(port, "/_app/immutable/bundle.x.js") == 404
        assert _status(port, "/favicon.ico") == 404
    finally:
        g.shutdown()


def test_the_key_and_host_checks_apply_to_the_page_files(upstream):
    g = gateway.serve(upstream, port=0, model_name="m", api_key="k", trust_loopback=False)
    port = g.server_address[1]
    try:
        assert _status(port, "/_app/immutable/bundle.x.js") == 401                # no key: refused, as for any route
        assert _status(port, "/favicon.ico", host="evil.example") == 401
        assert _status(port, "/_app/immutable/bundle.x.js", key="wrong") == 401
        assert _status(port, "/_app/immutable/bundle.x.js", key="k") == 200
        assert _status(port, "/pwa-192x192.png", key="k", host="mypc.lan") == 200   # a valid key works under any host name
    finally:
        g.shutdown()


def test_the_host_guard_covers_page_files_without_a_key(upstream):
    g = gateway.serve(upstream, port=0, model_name="m")
    port = g.server_address[1]
    try:
        assert _status(port, "/_app/immutable/bundle.x.js", host="evil.example") == 403   # DNS rebinding (#137)
        assert _status(port, "/_app/immutable/bundle.x.js", host=f"localhost:{port}") == 200
    finally:
        g.shutdown()


# --- #188: our own page is the default; --no-ui gives llama-server's -------------------------------------------------

from localllm import cli                                                      # noqa: E402


class _Proc:
    def wait(self):
        pass


def _ui_seen(monkeypatch, argv):
    """Parse argv like the command line does, run cmd_run with a stubbed _start, return the ui= it was given."""
    seen = []
    monkeypatch.setattr(cli, "first_run", lambda: False)
    monkeypatch.setattr(cli, "_starter_plan", lambda a: ("small", None))
    monkeypatch.setattr(cli, "_start", lambda key, *a, **k: (seen.append(k.get("ui")), (_Proc(), "http://x"))[1])
    args = cli.build_parser().parse_args(argv)
    args.no_browser = True
    args.fn(args)
    return seen


@pytest.mark.parametrize("argv,want", [([], True), (["--ui"], True), (["--no-ui"], False),
                                       (["serve"], True), (["serve", "--ui"], True), (["serve", "--no-ui"], False)])
def test_our_page_is_on_by_default_and_no_ui_turns_it_off(monkeypatch, argv, want):
    assert _ui_seen(monkeypatch, argv) == [want]


def test_the_other_commands_that_start_a_server_do_not_get_the_page():
    import inspect
    assert inspect.signature(cli._start).parameters["ui"].default is False     # chat, route, launch, eval: no workspace


def test_a_default_gateway_serves_our_page_and_no_ui_serves_llama_servers(upstream, tmp_path):
    ui = {"sessions": workspace.Sessions(tmp_path / "s"), "catalog": workspace.catalog_info([], "en", False)}
    ours, theirs = gateway.serve(upstream, port=0, model_name="m", ui=ui), gateway.serve(upstream, port=0, model_name="m")
    try:
        assert _get(ours.server_address[1], "/").read() != INDEX
        assert _get(theirs.server_address[1], "/").read() == INDEX
        assert _get(theirs.server_address[1], "/_app/immutable/bundle.x.js").status == 200
    finally:
        ours.shutdown(); theirs.shutdown()
