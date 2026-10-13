"""#76: `--ui` serves the local chat workspace; chats are plain files; on the LAN everything but the page needs the key."""
import json
import urllib.error
import urllib.request

import pytest

from localllm import gateway, pool, router, workspace
from test_gateway import _up
from test_pool import FakeProc, G, Q, ZH


def _req(url, method="GET", body=None, key=None):
    h = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    data = json.dumps(body).encode() if body is not None else None
    return urllib.request.urlopen(urllib.request.Request(url, data=data, method=method, headers=h), timeout=30)


@pytest.fixture
def ui(tmp_path):
    return {"sessions": workspace.Sessions(tmp_path / "sessions"),
            "catalog": workspace.catalog_info([G, Q], "th", True)}


def test_page_is_ours_and_self_contained(ui):
    s, up = _up()
    g = gateway.serve(up, port=0, model_name="m", ui=ui)
    r = _req(f"http://127.0.0.1:{g.server_address[1]}/")
    page = r.read().decode()
    assert "<title>localllm</title>" in page and "default-src 'self'" in r.headers["Content-Security-Policy"]
    assert "http://" not in page.replace("http://127.0.0.1", "") and "https://" not in page   # no outside assets
    g.shutdown(); s.shutdown()


def test_sessions_are_plain_files_with_crud(ui):
    s, up = _up()
    g = gateway.serve(up, port=0, model_name="m", ui=ui)
    base = f"http://127.0.0.1:{g.server_address[1]}/localllm/sessions"
    chat = {"title": "สวัสดี", "messages": [{"role": "user", "content": "สวัสดี"}]}
    assert _req(base + "/abc-1", "PUT", chat).status == 200
    assert json.loads((ui["sessions"].dir / "abc-1.json").read_text(encoding="utf-8"))["title"] == "สวัสดี"
    assert json.load(_req(base))[0]["id"] == "abc-1"
    assert json.load(_req(base + "/abc-1"))["messages"][0]["content"] == "สวัสดี"
    with pytest.raises(urllib.error.HTTPError) as e:
        _req(base + "/..%2Fevil", "PUT", chat)
    assert e.value.code == 400
    assert _req(base + "/abc-1", "DELETE").status == 200 and json.load(_req(base)) == []
    g.shutdown(); s.shutdown()


def test_lan_needs_the_key_for_everything_but_the_page(ui):
    s, up = _up()
    g = gateway.serve(up, port=0, model_name="m", api_key="k", trust_loopback=False, ui=ui)
    base = f"http://127.0.0.1:{g.server_address[1]}"
    assert _req(base + "/").status == 200
    for path in ("/localllm/sessions", "/localllm/catalog"):
        with pytest.raises(urllib.error.HTTPError) as e:
            _req(base + path)
        assert e.value.code == 401
        assert _req(base + path, key="k").status == 200
    g.shutdown(); s.shutdown()


def test_catalog_info_scores_in_the_users_language():
    info = workspace.catalog_info([G, Q], "th", True)
    assert [m["key"] for m in info["models"]] == [G, Q] and info["choosable"] and info["single"] is None
    assert 60 < info["models"][0]["score"] < 100          # mean of th/regional and th/math, translation excluded


def test_pool_uses_a_model_named_in_the_request(monkeypatch):
    monkeypatch.setattr(router, "load_config", lambda: {"enabled": False})

    def launch(key):
        s, url = _up()
        return FakeProc(s), url

    p = pool.Pool([Q, G], launch, 15.9, first=G)
    with p.use({"model": G, "messages": [{"role": "user", "content": ZH}]}) as (_url, label):
        assert label.startswith(f"{G} (asked for by name)")       # Chinese would have picked Qwen
    with p.use({"model": "gpt-4o", "messages": [{"role": "user", "content": ZH}]}) as (_url, label):
        assert label.startswith(Q)                                  # unknown names: routing as before
    p.close()
