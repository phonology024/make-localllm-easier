"""#101: reports from rented cloud machines carry host=cloud:<provider> through to the leaderboard."""
import sys
from pathlib import Path

from localllm import report

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import build_leaderboard  # noqa: E402

DEV = [{"id": "Vulkan0", "name": "NVIDIA GeForce RTX 4090", "total_gb": 24.0, "free_gb": 23.0}]


def _rep(monkeypatch, host):
    if host is None:
        monkeypatch.delenv("LOCALLLM_REPORT_HOST", raising=False)
    else:
        monkeypatch.setenv("LOCALLLM_REPORT_HOST", host)
    monkeypatch.setattr(report, "driver", lambda: "560.1")
    return report.collect(server=Path("b11476/llama-server"), devs=DEV, ram_gb=64, lang="en")


def test_host_only_when_set_and_well_formed(monkeypatch):
    assert "host" not in _rep(monkeypatch, None)
    assert _rep(monkeypatch, "cloud:runpod")["host"] == "cloud:runpod"
    assert "host" not in _rep(monkeypatch, "<script>")


def test_leaderboard_keeps_the_cloud_mark(monkeypatch):
    rep = _rep(monkeypatch, "cloud:vast")
    rep["tune"] = {"gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf": {"b11476": {"gpu": "x", "tok_s": 150.0}}}
    rows = [e for e in build_leaderboard.entries([rep]) if e["source"] == "community"]
    assert rows and rows[0]["host"] == "cloud:vast"
    assert all("host" not in e for e in build_leaderboard.entries([]))
