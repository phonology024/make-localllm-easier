"""Update notices (#120): once a day, opt-out, quiet offline, never downloads; they read the website's latest.json and
the published catalog/v1 (#164), the same file other tools read."""
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
sys.path.insert(0, str(ROOT / "tools"))
import build_catalog  # noqa: E402

G = "gemma4-26b-a4b-qat"


class Opener:
    """Serves latest.json and catalog/v1 from a _remote() spec; bytes are served as-is for both (junk)."""
    def __init__(self, payload=None, error=None):
        self.payload, self.error, self.calls = payload, error, 0

    def __call__(self, url, timeout):
        self.calls += 1
        assert url in (updates.LATEST_URL, updates.CATALOG_URL) and timeout <= 10
        if self.error:
            raise self.error
        if isinstance(self.payload, bytes):
            return io.BytesIO(self.payload)
        if url == updates.LATEST_URL:
            body = {"version": self.payload["version"]}
        else:
            models = self.payload["models"]
            body = {"catalog_version": 1, "models": [build_catalog.model_entry(k, m) for k, m in models.items()]
                    if isinstance(models, dict) else models}
        return io.BytesIO(json.dumps(body).encode())


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch, _no_update_check):     # after conftest turned checks off: on again here
    monkeypatch.setattr(runtime, "HOME", tmp_path)
    monkeypatch.delenv("LOCALLLM_NO_UPDATE_CHECK", raising=False)
    return tmp_path


def _remote(version="99.0.0", models=None):
    return {"version": version, "models": models if models is not None else copy.deepcopy(catalog.MODELS)}


def test_at_most_once_a_day_and_check_forces_it():
    o = Opener(_remote())
    assert updates.fetch(o, now=1000.0) and o.calls == 2                     # latest.json + catalog/v1
    assert not updates.fetch(o, now=1000.0 + 3600) and o.calls == 2          # same day: no request
    assert updates.fetch(o, now=1000.0 + 25 * 3600) and o.calls == 4
    assert updates.fetch(o, now=1000.0 + 25 * 3600, force=True) and o.calls == 6


def test_offline_and_junk_are_quiet_and_retried_tomorrow():
    assert updates.fetch(Opener(error=OSError("no network")), now=10.0)
    assert updates.notice() is None
    assert updates.fetch(Opener(b"<html>not json"), now=10.0 + 25 * 3600) and updates.notice() is None
    assert updates.fetch(Opener({"version": __version__, "models": "nope"}), now=10.0 + 50 * 3600)
    assert updates.notice() is None and "models" not in updates._load()["remote"]
    broken = {G: {k: v for k, v in catalog.MODELS[G].items()}}
    o = Opener({"version": "99.0.0", "models": broken})
    o.payload["models"] = [{"id": "x", "memory": {}}]                       # an entry this version can't read
    assert updates.fetch(o, now=10.0 + 75 * 3600) and "99.0.0" in updates.notice(15.9, "th", 20.0)


def test_opt_out_by_command_or_environment(monkeypatch):
    o = Opener(_remote())
    updates.set_enabled(False)
    assert not updates.fetch(o, now=1.0) and o.calls == 0 and updates.notice() is None
    updates.set_enabled(True)
    monkeypatch.setenv("LOCALLLM_NO_UPDATE_CHECK", "1")
    assert not updates.fetch(o, now=1.0) and o.calls == 0
    monkeypatch.delenv("LOCALLLM_NO_UPDATE_CHECK")
    assert updates.fetch(o, now=1.0) and o.calls == 2


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


def test_website_publishes_what_the_notice_reads():
    subprocess.run([sys.executable, str(ROOT / "site" / "build.py")], check=True, capture_output=True,
                   env={**os.environ, "SITE_URL": "https://example.org"})
    out = ROOT / "site" / "public"
    assert json.loads((out / "latest.json").read_text(encoding="utf-8")) == {"version": __version__}
    assert json.loads((out / "catalog" / "v1" / "catalog.json").read_text(encoding="utf-8")) == build_catalog.build()
    assert not (out / "catalog.json").exists()                  # the old dump of the internal dict is gone
    assert updates.LATEST_URL.endswith("/latest.json") and updates.CATALOG_URL.endswith("/catalog/v1/catalog.json")


def test_v1_entries_read_back_into_the_same_fit_and_score_answers():
    """from_v1 is the inverse of build_catalog.model_entry for everything the catalog rules read."""
    back = dict(updates.from_v1(build_catalog.model_entry(k, m)) for k, m in catalog.MODELS.items())
    assert back.keys() == catalog.MODELS.keys()
    cat = updates._private_catalog(back)
    langs = sorted({s.split("/")[0] for m in catalog.MODELS.values() for s in m["scores"]}) + [None]
    for k in catalog.MODELS:
        assert cat.below_floor(k) == catalog.below_floor(k)
        for lang in langs:
            assert cat.score(k, lang) == catalog.score(k, lang)
        for vram in (4, 6, 8, 10, 12, 16, 20, 24, 32, 48):
            for ram in (4.0, 12.0, 20.0, 48.0):
                assert cat.cpu_moe_layers(k, vram, ram) == catalog.cpu_moe_layers(k, vram, ram), (k, vram, ram)
                assert cat.speed(k, vram, ram) == catalog.speed(k, vram, ram), (k, vram, ram)
                assert cat.fits(k, vram) == catalog.fits(k, vram)
    for vram in (6, 8, 12, 16, 24):
        for lang in langs:
            assert cat.pick(vram, lang, 20.0) == catalog.pick(vram, lang, 20.0)
