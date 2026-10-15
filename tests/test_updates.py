"""Update notices (#120): once a day, opt-out, quiet offline, never downloads; the website publishes catalog.json."""
import copy
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from localllm import __version__, catalog, cli, runtime, updates

ROOT = Path(__file__).resolve().parents[1]
G = "gemma4-26b-a4b-qat"


class Opener:
    def __init__(self, payload=None, error=None):
        self.payload, self.error, self.calls = payload, error, 0

    def __call__(self, url, timeout):
        self.calls += 1
        assert url == updates.URL and timeout <= 10
        if self.error:
            raise self.error
        return io.BytesIO(self.payload if isinstance(self.payload, bytes) else json.dumps(self.payload).encode())


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch, _no_update_check):     # after conftest turned checks off: on again here
    monkeypatch.setattr(runtime, "HOME", tmp_path)
    monkeypatch.delenv("LOCALLLM_NO_UPDATE_CHECK", raising=False)
    return tmp_path


def _remote(version="99.0.0", models=None):
    return {"version": version, "models": models if models is not None else copy.deepcopy(catalog.MODELS)}


def test_at_most_once_a_day_and_check_forces_it():
    o = Opener(_remote())
    assert updates.fetch(o, now=1000.0) and o.calls == 1
    assert not updates.fetch(o, now=1000.0 + 3600) and o.calls == 1          # same day: no request
    assert updates.fetch(o, now=1000.0 + 25 * 3600) and o.calls == 2
    assert updates.fetch(o, now=1000.0 + 25 * 3600, force=True) and o.calls == 3


def test_offline_and_junk_are_quiet_and_retried_tomorrow():
    assert updates.fetch(Opener(error=OSError("no network")), now=10.0)
    assert updates.notice() is None
    assert updates.fetch(Opener(b"<html>not json"), now=10.0 + 25 * 3600) and updates.notice() is None
    assert updates.fetch(Opener({"version": "99.0.0", "models": "nope"}), now=10.0 + 50 * 3600)
    assert updates.notice() is None


def test_opt_out_by_command_or_environment(monkeypatch):
    o = Opener(_remote())
    updates.set_enabled(False)
    assert not updates.fetch(o, now=1.0) and o.calls == 0 and updates.notice() is None
    updates.set_enabled(True)
    monkeypatch.setenv("LOCALLLM_NO_UPDATE_CHECK", "1")
    assert not updates.fetch(o, now=1.0) and o.calls == 0
    monkeypatch.delenv("LOCALLLM_NO_UPDATE_CHECK")
    assert updates.fetch(o, now=1.0) and o.calls == 1


def test_newer_version_notice_and_nothing_when_current():
    updates.fetch(Opener(_remote("99.0.0")), now=1.0)
    n = updates.notice(15.9, "th", 20.0)
    assert "99.0.0" in n and __version__ in n and "pip install -U make-localllm-easier" in n
    updates.fetch(Opener(_remote(__version__)), now=1.0, force=True)
    assert updates.notice(15.9, "th", 20.0) is None


def test_better_model_for_your_language_that_fits():
    models = copy.deepcopy(catalog.MODELS)
    models["gemma5-27b-qat"] = {**copy.deepcopy(catalog.MODELS[G]),
                                "scores": {k: min(100.0, v + 10) for k, v in catalog.MODELS[G]["scores"].items()}}
    models["huge-400b"] = {**copy.deepcopy(catalog.MODELS["qwen3.8-27b-q3"]), "gb": 220.0,
                           "scores": {"th/regional": 99.0}}
    updates.fetch(Opener(_remote("99.0.0", models)), now=1.0)
    n = updates.notice(15.9, "th", 20.0)
    assert "gemma5-27b-qat" in n and "for th" in n and "huge-400b" not in n
    assert catalog.MODELS.keys() == set(k for k in models if k not in ("gemma5-27b-qat", "huge-400b"))   # restored
    assert "gemma5-27b-qat" not in updates.notice(0.0, "th", 20.0)           # no GPU: only the version notice


def test_scoring_new_models_never_touches_the_global_catalog(monkeypatch):
    """The server's routing threads read catalog.MODELS while the notice is worked out at startup: no swap, ever."""
    models = copy.deepcopy(catalog.MODELS)
    models["gemma5-27b-qat"] = {**copy.deepcopy(catalog.MODELS[G]),
                                "scores": {k: min(100.0, v + 10) for k, v in catalog.MODELS[G]["scores"].items()}}
    orig, seen, real = catalog.MODELS, [], updates._private_catalog

    def spy(merged):
        mod = real(merged)
        score = mod.score

        def checked(*a, **k):
            seen.append(catalog.MODELS is orig and "gemma5-27b-qat" not in catalog.MODELS)
            return score(*a, **k)
        mod.score = checked
        return mod
    monkeypatch.setattr(updates, "_private_catalog", spy)
    updates.fetch(Opener(_remote("99.0.0", models)), now=1.0)
    assert "gemma5-27b-qat" in updates.notice(15.9, "th", 20.0)
    assert seen and all(seen) and catalog.MODELS is orig


def test_status_carries_the_notice(monkeypatch):
    monkeypatch.setattr(cli, "NOTICE", ["localllm 99.0.0 is out"])
    assert cli._status(None, None)["notice"] == "localllm 99.0.0 is out"
    monkeypatch.setattr(cli, "NOTICE", [None])
    assert "notice" not in cli._status(None, None)


def test_website_publishes_the_catalog_the_notice_reads():
    subprocess.run([sys.executable, str(ROOT / "site" / "build.py")], check=True, capture_output=True,
                   env={**os.environ, "SITE_URL": "https://example.org"})
    published = json.loads((ROOT / "site" / "public" / "catalog.json").read_text(encoding="utf-8"))
    assert published["version"] == __version__
    assert published["models"] == json.loads(json.dumps(catalog.MODELS))
    assert updates.URL.endswith("/catalog.json") and updates.URL.startswith("https://")
