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


# ---- Models tab (#107) -------------------------------------------------------------------------------------------------
IQ2 = "qwen3.8-27b-iq2"


def test_fit_says_what_runs_on_this_pc():
    assert workspace.fit(Q, 15.9, 20.0) == ("fits", 0)
    assert workspace.fit(Q, 12.0, 20.0) == ("no", 0)                 # dense 12.2 GB: no way to keep part in RAM
    state, layers = workspace.fit(G, 12.0, 20.0)
    assert state == "spill" and layers > 0                             # MoE: some experts stay in system RAM
    assert workspace.fit(G, 8.0, 4.0) == ("no", 0)                     # ... unless RAM is short too


def test_browser_lists_runnable_models_first_with_scores_in_the_users_language():
    info = workspace.browser_info(12.0, 20.0, "th", active=[IQ2], have=lambda k: k == IQ2)
    assert [(m["key"], m["fit"]) for m in info["models"]] == [(IQ2, "fits"), (G, "spill"), (Q, "no")]
    iq2, g, q = info["models"]
    assert iq2["active"] and iq2["downloaded"] and not g["downloaded"] and q["tok_s"] is None
    assert set(q["badges"]) == {"regional", "math", "translate"}       # every th/ score, translation included
    assert q["score"] == round((q["badges"]["regional"] + q["badges"]["math"]) / 2, 1)   # chrF++ left out of the %
    assert g["tok_s"] < 85 and info["best"] in (IQ2, G) and info["lang"] == "th"
    sw = workspace.browser_info(15.9, 20.0, "sw", active=[], have=lambda k: False)
    assert all(m["score"] is None and m["badges"] == {} for m in sw["models"])   # not measured: said, not invented


def test_downloads_run_in_the_background_and_report_progress(tmp_path):
    import threading
    go, calls = threading.Event(), []

    def fetch(key):
        calls.append(key)
        go.wait(10)
        if key == Q:
            raise SystemExit("[localllm] SHA-256 mismatch")

    d = workspace.Downloads(fetch, lambda k: tmp_path / (k + ".part"))
    with pytest.raises(ValueError):
        d.start("not-a-model")
    assert d.start(G)["state"] == "downloading" and d.start(G)["state"] == "downloading"
    d.start(Q)
    with open(tmp_path / (G + ".part"), "wb") as f:
        f.truncate(2**30)                                              # 1 of 13.3 GB, without writing it
    assert 7 < d.state(G)["pct"] < 8
    go.set()
    for _ in range(100):
        if d.state(G)["state"] != "downloading" and d.state(Q)["state"] != "downloading":
            break
        threading.Event().wait(0.05)
    assert d.state(G) == {"state": "done", "error": None, "pct": 100.0} and calls.count(G) == 1   # started once
    assert d.state(Q)["state"] == "error" and "SHA-256" in d.state(Q)["error"]
    assert d.state(IQ2) is None


def test_models_endpoints_and_lan_key(ui, tmp_path):
    started = []
    downloads = workspace.Downloads(lambda k: started.append(k), lambda k: tmp_path / "none.part")
    ui.update(downloads=downloads,
              models=lambda: workspace.browser_info(15.9, 20.0, "th", [G], lambda k: k == G, downloads))
    s, up = _up()
    g = gateway.serve(up, port=0, model_name="m", api_key="k", trust_loopback=False, ui=ui)
    base = f"http://127.0.0.1:{g.server_address[1]}/localllm/models"
    for method in ("GET", "POST"):
        with pytest.raises(urllib.error.HTTPError) as e:
            _req(base + ("" if method == "GET" else f"/{Q}/download"), method, {} if method == "POST" else None)
        assert e.value.code == 401
    info = json.load(_req(base, key="k"))
    assert {m["key"] for m in info["models"]} == {G, Q, IQ2} and info["active"] == [G]
    r = _req(f"{base}/{Q}/download", "POST", {}, key="k")
    assert r.status == 202 and json.load(r)["state"] in ("downloading", "done")
    with pytest.raises(urllib.error.HTTPError) as e:
        _req(f"{base}/nope/download", "POST", {}, key="k")
    assert e.value.code == 404
    g.shutdown(); s.shutdown()


def test_models_tab_is_keyboard_reachable():
    page = workspace.PAGE.read_text(encoding="utf-8")
    assert 'role="tablist"' in page and 'id="tab-models" role="tab"' in page and "ArrowRight" in page
    assert ":focus-visible" in page and 'data-dl="' in page and "<button" in page.split("function modelCard", 1)[1]
