"""The chat page as an app (#122): manifest, icon and service worker are served with it, nothing personal is cached."""
import json
import urllib.error
import urllib.request

import pytest

from localllm import gateway, pwa, workspace
from test_gateway import _up


@pytest.fixture
def lan(tmp_path):
    s, up = _up()
    ui = {"sessions": workspace.Sessions(tmp_path / "s"), "catalog": workspace.catalog_info([], "en", False)}
    g = gateway.serve(up, port=0, model_name="m", api_key="k", trust_loopback=False, ui=ui)
    yield f"http://127.0.0.1:{g.server_address[1]}"
    g.shutdown(); s.shutdown()


def test_app_files_are_served_without_the_key_like_the_page(lan):
    for path, ctype in (("/manifest.webmanifest", "application/manifest+json"), ("/sw.js", "text/javascript"),
                        ("/icon.svg", "image/svg+xml")):
        r = urllib.request.urlopen(lan + path, timeout=10)
        assert r.status == 200 and r.headers["Content-Type"].startswith(ctype), path
    m = json.load(urllib.request.urlopen(lan + "/manifest.webmanifest", timeout=10))
    assert m["start_url"] == "/" and m["display"] == "standalone" and m["icons"][0]["src"] == "/icon.svg"
    with pytest.raises(urllib.error.HTTPError) as e:                       # everything else still needs the key
        urllib.request.urlopen(lan + "/localllm/sessions", timeout=10)
    assert e.value.code == 401


def test_the_font_is_served_without_the_key_and_only_by_its_names(lan):
    for w in (400, 500, 600):
        r = urllib.request.urlopen(f"{lan}/fonts/poppins-{w}.woff2", timeout=10)
        body = r.read()
        assert r.status == 200 and r.headers["Content-Type"] == "font/woff2" and body[:4] == b"wOF2"
        assert int(r.headers["Content-Length"]) == len(body) and "max-age=31536000" in r.headers["Cache-Control"]
    for path in ("/fonts/poppins-700.woff2", "/fonts/OFL.txt", "/fonts/SOURCE.json", "/fonts/../ui.html",
                 "/fonts/..%2fui.html", "/fonts/%2e%2e/gateway.py", "/fonts/poppins-400.woff2/../../ui.html",
                 "/fonts/", "/fonts/poppins-400.woff2?x=1"):
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(lan + path, timeout=10)
        assert e.value.code in (401, 404), path                            # not a font: the key, or not found
    s, up = _up()                                                         # with the key: still only these names
    g = gateway.serve(up, port=0, model_name="m", ui={"sessions": None, "catalog": {}})
    try:
        for path in ("/fonts/poppins-700.woff2", "/fonts/..%2fui.html", "/fonts/%2e%2e/gateway.py"):
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(f"http://127.0.0.1:{g.server_address[1]}{path}", timeout=10)
            assert e.value.code == 404, path
    finally:
        g.shutdown(); s.shutdown()


def test_the_shipped_font_files_match_their_recorded_source():
    import hashlib
    from pathlib import Path
    fonts = Path(gateway.__file__).with_name("fonts")
    src = json.loads((fonts / "SOURCE.json").read_text(encoding="utf-8"))
    assert set(src["files"]) == set(gateway.FONTS.values()) | {"OFL.txt"}
    for name, f in src["files"].items():
        assert hashlib.sha256((fonts / name).read_bytes()).hexdigest() == f["sha256"], name
    assert "SIL OPEN FONT LICENSE Version 1.1" in (fonts / "OFL.txt").read_text(encoding="utf-8")
    assert "google/fonts/tree/" in src["upstream"]


def test_the_wheel_contains_the_font(tmp_path):
    """Built the way pip builds it from a checkout (setuptools, no network): the fonts and their licence ship."""
    import subprocess
    import sys
    import zipfile
    from pathlib import Path
    import importlib.util
    setuptools = pytest.importorskip("setuptools")
    if tuple(int(x) for x in setuptools.__version__.split(".")[:2]) < (70, 1) and not importlib.util.find_spec("wheel"):
        pytest.skip("this setuptools needs the wheel package to build a wheel (CI installs setuptools>=70.1)")
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-c", "import sys; from setuptools import build_meta as b; "
                    "print(b.build_wheel(sys.argv[1]))", str(tmp_path)], cwd=root, check=True, capture_output=True,
                   timeout=300)
    with zipfile.ZipFile(next(tmp_path.glob("*.whl"))) as z:
        names = set(z.namelist())
    assert {f"localllm/fonts/{n}" for n in (*gateway.FONTS.values(), "OFL.txt", "SOURCE.json")} <= names
    assert "localllm/ui.html" in names


def test_service_worker_keeps_only_the_page_shell():
    assert pwa.SHELL == ["/", "/manifest.webmanifest", "/icon.svg"]
    assert "FILES.includes(url.pathname)" in pwa.SW and 'e.request.method !== "GET"' in pwa.SW
    assert "/localllm" not in pwa.SW and "/v1" not in pwa.SW                # API, chats, status: never cached


def test_page_links_the_app_and_csp_allows_only_our_worker(lan):
    r = urllib.request.urlopen(lan + "/", timeout=10)
    page = r.read().decode()
    csp = r.headers["Content-Security-Policy"]
    assert "worker-src 'self'" in csp and "manifest-src 'self'" in csp and "default-src 'self'" in csp
    assert "font-src 'self'" in csp and "font-src 'self' " not in csp                   # our /fonts/, nothing else
    assert '<link rel="manifest" href="/manifest.webmanifest">' in page and 'register("/sw.js")' in page
    assert "window.isSecureContext" in page and "history.replaceState" in page   # a ?key= link is not kept in the URL
