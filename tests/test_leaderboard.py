"""Leaderboard data and platform matrix from shared reports (#73, #99)."""
import json
import sys
from pathlib import Path

from localllm import report

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import build_leaderboard as lb  # noqa: E402

REP = {"schema": 1, "localllm": "0.6.1", "date": "2026-10-12", "os": "Linux 6.8.0", "arch": "x86_64",
       "ram_gb": 64.0, "language": "de", "llama_cpp": "b11476", "gpu": "NVIDIA GeForce RTX 4090", "vram_gb": 24.0,
       "tune": {"gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf": {"b11476": {"env": {}, "mtp": 0, "tok_s": 140.0}}},
       "results": {"gemma4-26b-a4b-qat": {"de/global": {"acc": 78.0}, "_meta": {}}}}


def test_vendor_and_os_family():
    assert [report.vendor(g) for g in ("AMD Radeon RX 9070 XT", "NVIDIA GeForce RTX 4060", "Intel(R) Arc(TM) A770",
                                       "Apple M3 Max", "Radeon RX 7600", None)] == \
        ["AMD", "NVIDIA", "Intel", "Apple", "AMD", None]
    assert [report.os_family(o) for o in ("Windows 11", "Linux 6.8", "Darwin 24.1", "FreeBSD")] == \
        ["Windows", "Linux", "macOS", None]


def test_validate_rejects_broken_reports():
    assert lb.validate(REP) == []
    assert "unknown report schema 9" in lb.validate({**REP, "schema": 9})
    assert any("vram_gb" in w for w in lb.validate({**REP, "vram_gb": -1}))
    bad = {**REP, "results": {"x": {"en/global": {"acc": 140}}}}
    assert any("not 0-100" in w for w in lb.validate(bad))


def test_entries_keep_maintainer_rows_and_add_community_ones():
    es = lb.entries([REP])
    ours = [e for e in es if e["source"] == "maintainers"]
    theirs = [e for e in es if e["source"] == "community"]
    assert len(ours) == len(lb.catalog.MODELS) and all(e["os_family"] == "Windows" for e in ours)
    assert theirs == [{"source": "community", "gpu": "NVIDIA GeForce RTX 4090", "vram_gb": 24.0, "ram_gb": 64.0,
                       "os": "Linux 6.8.0", "llama_cpp": "b11476", "vendor": "NVIDIA", "os_family": "Linux",
                       "model": "gemma4-26b-a4b-qat", "tok_s": 140.0, "scores": {"de/global": 78.0}, "flags": []}]
    fast = json.loads(json.dumps(REP))
    fast["tune"]["gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"]["b11476"]["tok_s"] = 900.0
    assert lb.entries([fast])[-1]["flags"]


def test_platform_matrix_counts_reports():
    m = lb.platforms([REP, {**REP, "date": "2026-10-13"}])
    assert m["Windows/AMD"] == {"maintainers": True, "community_reports": 0}
    assert m["Linux/NVIDIA"] == {"maintainers": False, "community_reports": 2}
    assert "Windows/Apple" not in m and "macOS/AMD" not in m
    t = lb.table(m)
    assert "| Linux | untested | community reports (2) |" in t


def test_load_reports_skips_with_reason(tmp_path):
    (tmp_path / "ok.json").write_text(json.dumps(REP))
    (tmp_path / "issue.md").write_text(f"text\n```json\n{json.dumps(REP)}\n```\n")
    (tmp_path / "bad.json").write_text(json.dumps({**REP, "schema": 0}))
    (tmp_path / "junk.json").write_text("{nope")
    (tmp_path / "README.md").write_text("how to add a report")
    reps, skipped = lb.load_reports(tmp_path)
    assert len(reps) == 2 and len(skipped) == 2 and skipped[0].startswith("bad.json: unknown report schema")


def test_doctor_label_reads_the_generated_matrix(tmp_path, monkeypatch):
    f = tmp_path / "platforms.json"
    f.write_text(json.dumps({"Windows/AMD": {"maintainers": True, "community_reports": 3},
                             "Linux/NVIDIA": {"maintainers": False, "community_reports": 0}}))
    monkeypatch.setattr(report, "PLATFORMS", f)
    assert report.platform_label("AMD Radeon RX 9070 XT", "Windows") == \
        "Windows/AMD: tested by maintainers, 3 community report(s)"
    assert report.platform_label("NVIDIA GeForce RTX 3060", "Linux").startswith("Linux/NVIDIA: untested")


def test_generated_files_are_up_to_date():
    reps, _ = lb.load_reports(lb.ROOT / "reports")
    for rel, text in lb.build(reps).items():
        assert (lb.ROOT / rel).read_text(encoding="utf-8") == text, f"{rel}: run python tools/build_leaderboard.py"
