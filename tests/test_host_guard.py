"""#137: without a key, only requests that name this server by IP, localhost or its --host get in (DNS rebinding)."""
import json
import urllib.error
import urllib.request

import pytest

from localllm import gateway, workspace
from test_gateway import _up


def _call(port, path="/v1/models", host=None, method="GET", body=None, key=None):
    h = {"Content-Type": "application/json", **({"Host": host} if host else {}),
         **({"Authorization": f"Bearer {key}"} if key else {})}
    data = json.dumps(body).encode() if body is not None else None
    return urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method,
                                                         headers=h), timeout=30)


@pytest.fixture
def gw(tmp_path):
    s, up = _up()
    ui = {"sessions": workspace.Sessions(tmp_path / "s"), "catalog": workspace.catalog_info([], "en", False)}
    g = gateway.serve(up, port=0, model_name="m", ui=ui)
    yield g.server_address[1]
    g.shutdown(); s.shutdown()


@pytest.mark.parametrize("host", ["127.0.0.1:{p}", "localhost:{p}", "LOCALHOST:{p}", "[::1]:{p}", "127.0.0.1", None])
def test_ip_and_localhost_hosts_are_served(gw, host):
    assert _call(gw, host=host.format(p=gw) if host else None).status == 200
    chat = {"messages": [{"role": "user", "content": "hi"}]}
    assert _call(gw, "/v1/chat/completions", host.format(p=gw) if host else None, "POST", chat).status == 200


@pytest.mark.parametrize("host", ["evil.example:{p}", "evil.example", "127.0.0.1.nip.io:{p}", "localhost.evil.example"])
@pytest.mark.parametrize("path,method,body", [("/v1/models", "GET", None), ("/", "GET", None),
                                              ("/localllm/i18n?lang=th", "GET", None),
                                              ("/v1/chat/completions", "POST", {"messages": []}),
                                              ("/localllm/sessions", "GET", None)])
def test_rebinding_hosts_are_refused_on_every_route(gw, host, path, method, body):
    with pytest.raises(urllib.error.HTTPError) as e:
        _call(gw, path, host.format(p=gw), method, body)
    assert e.value.code == 403 and b"DNS-rebinding" in e.value.read()


def test_a_valid_key_works_under_any_host_name():
    s, up = _up()
    g = gateway.serve(up, port=0, model_name="m", api_key="k", trust_loopback=False)
    port = g.server_address[1]
    assert _call(port, host=f"mypc.lan:{port}", key="k").status == 200          # LAN mode: the key is the check
    with pytest.raises(urllib.error.HTTPError) as e:
        _call(port, host=f"mypc.lan:{port}")
    assert e.value.code == 401
    g.shutdown(); s.shutdown()


def test_the_host_it_was_started_with_is_accepted():
    handler = gateway.make_handler("http://127.0.0.1:1", "m", bind_host="studio-pc")
    h = handler.__new__(handler)
    for host, ok in (("studio-pc:8080", True), ("studio-pc", True), ("other-pc:8080", False), ("10.0.0.5:8080", True)):
        h.headers = {"Host": host}
        assert h._host_ok() is ok, host
