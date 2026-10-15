"""Shared results: the issue form, the privacy check and turning an accepted report into reports/*.json (#101)."""
import copy
import json
import re
import subprocess
import sys
from pathlib import Path

from localllm import report

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import add_report  # noqa: E402
import build_leaderboard  # noqa: E402

REP = {"schema": 1, "localllm": "0.6.1", "date": "2026-10-12", "os": "Linux 6.8.0", "arch": "x86_64",
       "ram_gb": 64.0, "language": "de", "llama_cpp": "b11476", "gpu": "NVIDIA GeForce RTX 4090", "vram_gb": 24.0,
       "driver": "32.0.15.6094", "devices": [{"name": "NVIDIA GeForce RTX 4090", "vram_gb": 24.0}],
       "tune": {"gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf": {"b11476": {"env": {}, "mtp": 0, "tok_s": 140.0}}},
       "results": {"gemma4-26b-a4b-qat": {"de/global": {"acc": 78.0}, "_meta": {}}}}


def form_body(rep: dict) -> str:
    """What GitHub writes for an issue opened from .github/ISSUE_TEMPLATE/results.yml."""
    return (f"### Summary\n\n{report.summary(rep)}\n\n### Report\n\n```json\n{json.dumps(rep, indent=1)}\n```\n\n"
            "### Notes\n\n_No response_\n\n### Publishing\n\n- [X] I read the report\n")


def test_form_fields_match_what_localllm_report_prefills():
    form = (ROOT / ".github" / "ISSUE_TEMPLATE" / "results.yml").read_text(encoding="utf-8")
    ids = re.findall(r"^\s+id: (\w+)", form, re.M)
    assert {"summary", "report"} <= set(ids) and "render: json" in form
    assert re.search(r"^labels: \[results\]", form, re.M)


def test_a_clean_report_from_the_form_is_accepted_and_written(tmp_path):
    rep, why = add_report.check(form_body(REP))
    assert why == [] and rep == REP
    out = add_report.file_name(rep, tmp_path)
    assert out.name == "nvidia-geforce-rtx-4090-2026-10-12.json"
    out.write_text("{}")
    assert add_report.file_name(rep, tmp_path).name == "nvidia-geforce-rtx-4090-2026-10-12-2.json"   # never overwrite


def test_what_collect_produces_passes_the_privacy_check(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "_user_bits", lambda: [str(tmp_path), "alice"])
    raw = copy.deepcopy(REP)
    raw["results"]["alice-run"] = {"de/global": {"acc": 70.0, "_file": str(tmp_path / "models" / "x.gguf")}}
    assert report.privacy_problems(report.scrub(raw)) == []


def test_privacy_leaks_are_refused_with_the_reason():
    leaks = {"api_key": ("extra", {"api_key": "sk-123"}),
             "abs path": ("pick", "C:\\Users\\alice\\models\\x.gguf"),
             "home in text": ("notes", "ran it from /home/alice/llm"),
             "email": ("language", "de, write me at alice@example.com")}
    for name, (field, value) in leaks.items():
        rep = {**copy.deepcopy(REP), field: value}
        why = build_leaderboard.validate(rep)
        assert any(w.startswith("privacy:") for w in why), (name, why)
        assert add_report.check(form_body(rep))[1] == why


def test_score_names_that_would_become_page_paths_are_refused():
    for bad in ("../../x/global", 'x"onclick=alert(1)/global', "TH/global", "th/../.."):
        rep = copy.deepcopy(REP)
        rep["results"]["gemma4-26b-a4b-qat"][bad] = {"acc": 50.0}
        assert any("not a language/suite name" in w for w in build_leaderboard.validate(rep)), bad
    ok = copy.deepcopy(REP)
    ok["results"]["gemma4-26b-a4b-qat"]["pt-br/longctx-8k"] = {"acc": 50.0}
    assert build_leaderboard.validate(ok) == []


def test_unreadable_or_broken_reports_say_why():
    assert add_report.check("### Report\n\nI forgot to paste it")[1][0].startswith("no readable report JSON")
    bad = {**REP, "vram_gb": -1}
    assert any("vram_gb" in w for w in add_report.check(form_body(bad))[1])


def test_check_mode_exit_code_and_message(tmp_path):
    good, bad = tmp_path / "good.md", tmp_path / "bad.md"
    good.write_text(form_body(REP), encoding="utf-8")
    bad.write_text(form_body({**REP, "pick": "/home/alice/x.gguf"}), encoding="utf-8")
    run = lambda f: subprocess.run([sys.executable, str(ROOT / "tools" / "add_report.py"), str(f), "--check"],  # noqa: E731
                                   capture_output=True, text=True)
    ok, refused = run(good), run(bad)
    assert ok.returncode == 0 and "Looks good" in ok.stdout
    assert refused.returncode == 1 and "privacy" in refused.stdout
    assert not list((ROOT / "reports").glob("nvidia-geforce-rtx-4090-2026-10-12*.json"))      # --check writes nothing
