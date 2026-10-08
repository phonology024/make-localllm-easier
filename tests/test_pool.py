import json
import urllib.error
import urllib.request

from localllm import pool, router
from test_gateway import _gw, _post, _up

Q, G = "qwen3.8-27b-q3", "gemma4-26b-a4b-qat"
ZH = "请用三句话解释为什么天空是蓝色的。"
TH = "ช่วยอธิบายว่าทำไมท้องฟ้าถึงเป็นสีฟ้า"


def test_pick_local_switches_only_for_a_clear_gain():
    assert router.pick_local(ZH, [Q, G], G, 15.9)[0] == Q                 # zh: Qwen +5 pts -> worth a swap
    assert router.pick_local(TH, [Q, G], Q, 15.9)[0] == Q                 # th: gemma within 2 pts -> keep loaded
    assert router.pick_local("Hello there, how are you?", [Q, G], None, 15.9)[0] == G   # cold start: tie -> faster
    assert router.pick_local(ZH, [Q], None, 15.9)[0] == Q                 # single model: nothing to pick


class FakeProc:
    def __init__(self, server):
        self.server = server

    def terminate(self):
        self.server.shutdown()

    def wait(self, _t=None):
        return 0


def test_pool_lazy_loads_and_gateway_reports_the_model(monkeypatch):
    monkeypatch.setattr(router, "load_config", lambda: {"enabled": False})
    launched = []

    def launch(key):
        launched.append(key)
        s, url = _up()
        return FakeProc(s), url

    p = pool.Pool([Q, G], launch, 15.9, first=G)
    g, gurl = _gw(p)
    r = _post(gurl + "/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi, what is 2+2 in words?"}]})
    assert r.headers["X-Localllm-Model"].startswith(G) and launched == [G]          # English: stays on gemma
    r = _post(gurl + "/v1/chat/completions", {"messages": [{"role": "user", "content": ZH}]})
    assert r.headers["X-Localllm-Model"].startswith(Q) and "swapped" in r.headers["X-Localllm-Model"]
    assert json.load(r)["choices"][0]["message"]["content"] == "echo: " + ZH
    assert launched == [G, Q] and len(p.swaps) == 2
    r = _post(gurl + "/api/chat", {"messages": [{"role": "user", "content": ZH}], "stream": False})
    assert r.headers["X-Localllm-Model"].startswith(Q) and launched == [G, Q]      # Ollama path goes through the pool too
    p.close(); g.shutdown()


def test_detect_task():
    assert router.detect_task("Janet has 16 eggs, eats 3 and bakes with 4. How many are left?") == "math"
    assert router.detect_task("เป็ดวางไข่วันละ 16 ฟอง กินไป 3 ฟอง เหลือกี่ฟอง") == "math"
    assert router.detect_task("What is 17 * 23?") == "math"
    assert router.detect_task("Write a poem about the sea.") == "general"
    assert router.detect_task("I was born in 1990 and moved in 2010.") == "general"   # numbers alone aren't math


def test_task_changes_the_pick_for_chinese():
    zh_math = "小明有 15 个苹果，给了朋友 7 个，又买了 12 个。他现在有多少个苹果？"
    assert router.pick_local(ZH, [Q, G], G, 15.9)[0] == Q          # zh knowledge: Qwen +5.5
    assert router.pick_local(zh_math, [Q, G], Q, 15.9)[0] == G     # zh math: gemma +4.4 (MGSM)


def test_resident_pool_routes_without_swaps(monkeypatch):
    monkeypatch.setattr(router, "load_config", lambda: {"enabled": False})
    launched = []

    def launch(key):
        launched.append(key)
        s, url = _up()
        return FakeProc(s), url

    p = pool.Pool([Q, G], launch, 32.0, first=G, resident=True)
    g, gurl = _gw(p)
    assert sorted(launched) == sorted([Q, G]) and p.swaps == []
    th = _post(gurl + "/v1/chat/completions", {"messages": [{"role": "user", "content": TH}]})
    zh = _post(gurl + "/v1/chat/completions", {"messages": [{"role": "user", "content": ZH}]})
    assert th.headers["X-Localllm-Model"].startswith(G) and zh.headers["X-Localllm-Model"].startswith(Q)
    assert "swapped" not in zh.headers["X-Localllm-Model"] and len(launched) == 2
    p.close(); g.shutdown()


def test_detect_translate():
    assert router.detect_task("Translate into Thai: The meeting is at 3 pm.") == "translate"
    assert router.detect_task("ช่วยแปลประโยคนี้เป็นภาษาอังกฤษ: วันนี้อากาศดี") == "translate"
    assert router.detect_task("请把这句话翻译成英文：今天天气很好。") == "translate"
    assert router.detect_task("How many people speak Thai?") == "general"


def test_translation_request_does_not_swap_for_chinese():
    zh_tr = "请把这句话翻译成英文：今天天气很好，我们去公园散步吧。"
    assert router.pick_local(zh_tr, [Q, G], G, 15.9)[0] == G      # translate: Qwen only +1.4 -> stay on gemma
    assert router.pick_local(ZH, [Q, G], G, 15.9)[0] == Q         # same language, knowledge question: +5.5 -> swap


def test_prefetch_reads_other_models_and_respects_ram(tmp_path):
    import threading
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x" * (3 * pool.PREFETCH_CHUNK + 5))
    assert pool.prefetch(f, threading.Event(), ram_free_gb=64.0)
    assert not pool.prefetch(f, threading.Event(), ram_free_gb=1.0)          # not enough spare RAM: skip
    stop = threading.Event(); stop.set()
    assert not pool.prefetch(f, stop, ram_free_gb=64.0)                       # a swap started: give up


def test_pool_prefetches_the_model_that_is_not_loaded(tmp_path, monkeypatch):
    import time as _t
    monkeypatch.setattr(pool, "prefetch", lambda p, stop, ram_free_gb=None: True)
    files = {Q: tmp_path / "q.gguf", G: tmp_path / "g.gguf"}

    def launch(key):
        s, url = _up()
        return FakeProc(s), url

    p = pool.Pool([Q, G], launch, 15.9, first=G, files=files)
    for _ in range(50):
        if p.prefetched:
            break
        _t.sleep(0.02)
    assert p.prefetched == ["q.gguf"]                                         # gemma loaded -> warm Qwen's file
    p.close()


def _wait(cond, timeout=5.0):
    import time
    end = time.time() + timeout
    while time.time() < end and not cond():
        time.sleep(0.02)
    return cond()


def test_idle_unload_frees_the_gpu_and_the_next_message_loads_again(monkeypatch):
    monkeypatch.setattr(router, "load_config", lambda: {"enabled": False})
    launched, procs = [], []

    def launch(key):
        launched.append(key)
        s, url = _up()
        procs.append(FakeProc(s))
        return procs[-1], url

    p = pool.Pool([Q, G], launch, 15.9, first=G, idle_unload_s=0.2)
    g, gurl = _gw(p)
    try:
        assert _wait(lambda: p.unloads == 1) and p.url is None and p.proc is None      # idle: nothing loaded
        r = _post(gurl + "/v1/chat/completions", {"messages": [{"role": "user", "content": ZH}]})
        hdr = r.headers["X-Localllm-Model"]
        assert hdr.startswith(Q) and "loaded after idle" in hdr and "swapped" not in hdr   # best model, no swap cost
        assert launched == [G, Q] and len(p.swaps) == 1
        assert _wait(lambda: p.unloads == 2)
        assert json.load(urllib.request.urlopen(gurl + "/health", timeout=10)) == {"status": "ok", "model": "unloaded"}
        ids = [m["id"] for m in json.load(urllib.request.urlopen(gurl + "/v1/models", timeout=10))["data"]]
        page = urllib.request.urlopen(gurl + "/", timeout=10).read().decode()
        try:
            urllib.request.urlopen(gurl + "/props", timeout=10)
            raise AssertionError("/props should not wake the model")
        except urllib.error.HTTPError as e:
            assert e.code == 503
        assert ids == [Q, G] and "unloaded while idle" in page and launched == [G, Q]    # no GET loaded anything
        _post(gurl + "/v1/chat/completions", {"messages": [{"role": "user", "content": ZH}]})
        assert launched == [G, Q, Q]                                                       # a message does
    finally:
        p.close(); g.shutdown()


def test_polling_gets_do_not_keep_the_model_loaded(monkeypatch):
    import time
    monkeypatch.setattr(router, "load_config", lambda: {"enabled": False})
    launched = []

    def launch(key):
        launched.append(key)
        s, url = _up()
        return FakeProc(s), url

    p = pool.Pool([Q, G], launch, 15.9, first=G, idle_unload_s=0.3)
    g, gurl = _gw(p)
    try:
        end = time.time() + 1.5
        while time.time() < end and not p.unloads:     # a dashboard polling every 50 ms while the model is loaded
            urllib.request.urlopen(gurl + "/health", timeout=10).read()
            urllib.request.urlopen(gurl + "/v1/models", timeout=10).read()
            time.sleep(0.05)
        assert p.unloads == 1 and launched == [G]
    finally:
        p.close(); g.shutdown()


def test_resident_pool_unloads_and_reloads_every_model(monkeypatch):
    monkeypatch.setattr(router, "load_config", lambda: {"enabled": False})
    launched = []

    def launch(key):
        launched.append(key)
        s, url = _up()
        return FakeProc(s), url

    p = pool.Pool([Q, G], launch, 32.0, first=G, resident=True, idle_unload_s=0.2)
    g, gurl = _gw(p)
    try:
        assert _wait(lambda: p.unloads == 1) and not p.loaded
        r = _post(gurl + "/v1/chat/completions", {"messages": [{"role": "user", "content": ZH}]})
        assert r.headers["X-Localllm-Model"].startswith(Q) and "loaded after idle" in r.headers["X-Localllm-Model"]
        assert sorted(launched) == sorted([Q, G, Q, G]) and set(p.loaded) == {Q, G}
    finally:
        p.close(); g.shutdown()


def test_stay_keeps_the_loaded_model(monkeypatch):
    monkeypatch.setattr(router, "load_config", lambda: {"enabled": False})
    launched = []

    def launch(key):
        launched.append(key)
        s, url = _up()
        return FakeProc(s), url

    p = pool.Pool([Q, G], launch, 15.9, first=G, idle_unload_s=0)
    g, gurl = _gw(p)
    try:
        req = urllib.request.Request(gurl + "/v1/chat/completions", headers={"Content-Type": "application/json",
                                                                             "X-Localllm-Stay": "1"},
                                     data=json.dumps({"messages": [{"role": "user", "content": ZH}]}).encode())
        hdr = urllib.request.urlopen(req, timeout=30).headers["X-Localllm-Model"]
        assert hdr.startswith(G) and "stayed as asked" in hdr and Q in hdr and launched == [G]   # no swap
        r = _post(gurl + "/v1/chat/completions", {"messages": [{"role": "user", "content": ZH}]})
        assert r.headers["X-Localllm-Model"].startswith(Q) and launched == [G, Q]          # without it: switches
    finally:
        p.close(); g.shutdown()
