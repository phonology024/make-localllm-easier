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


def test_service_worker_keeps_only_the_page_shell():
    assert pwa.SHELL == ["/", "/manifest.webmanifest", "/icon.svg"]
    assert "FILES.includes(url.pathname)" in pwa.SW and 'e.request.method !== "GET"' in pwa.SW
    assert "/localllm" not in pwa.SW and "/v1" not in pwa.SW                # API, chats, status: never cached


def test_page_links_the_app_and_csp_allows_only_our_worker(lan):
    r = urllib.request.urlopen(lan + "/", timeout=10)
    page = r.read().decode()
    csp = r.headers["Content-Security-Policy"]
    assert "worker-src 'self'" in csp and "manifest-src 'self'" in csp and "default-src 'self'" in csp
    assert '<link rel="manifest" href="/manifest.webmanifest">' in page and 'register("/sw.js")' in page
    assert "window.isSecureContext" in page and "history.replaceState" in page   # a ?key= link is not kept in the URL
