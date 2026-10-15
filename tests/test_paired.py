"""Per-item outcomes and the paired McNemar comparison (#65, #168)."""
import json
import sys
from pathlib import Path

from localllm import bench, report

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import paired  # noqa: E402

ITEMS = [{"q": f"question {i}", "ans": "A"} for i in range(6)]


def test_bench_records_per_item_outcomes(monkeypatch):
    saved = []
    monkeypatch.setattr(bench, "load", lambda suite, lang, limit: ITEMS)
    monkeypatch.setattr(bench, "ask", lambda url, it: "A" if it["q"][-1] in "024" else "B")
    monkeypatch.setattr(bench, "_save", lambda name, res: saved.append(res))
    res = bench.run("http://x", "m", ["en"], 0, ("global",))
    r = res["en/global"]
    assert r["outcomes"] == "101010" and r["correct"] == 3 and r["n"] == 6
    assert r["items_sha"] == bench.paired(ITEMS, [True] * 6)["items_sha"]          # depends on the items, not the answers
    assert r["items_sha"] != bench.paired(ITEMS[:5], [True] * 5)["items_sha"]


def test_shared_reports_leave_the_per_item_strings_out():
    res = {"m": {"en/global": {"acc": 50.0, "n": 6, "outcomes": "101010", "items_sha": "ab"}, "_meta": {"x": 1}}}
    assert report.lean_results(res) == {"m": {"en/global": {"acc": 50.0, "n": 6}, "_meta": {"x": 1}}}


def test_mcnemar_counts_only_the_items_the_two_disagree_on():
    r = paired.mcnemar("1111100000", "1111000001")
    assert (r["a"], r["b"], r["a_only"], r["b_only"]) == (5, 5, 1, 1) and r["p"] == 1.0
    r = paired.mcnemar("1" * 20 + "0" * 80, "0" * 100)                                # 20 vs 0 discordant: clearly real
    assert r["a_only"] == 20 and r["b_only"] == 0 and r["p"] < 1e-5
    assert paired.mcnemar("10", "10")["p"] == 1.0                                     # no disagreement at all


def test_paired_compares_jobs_on_the_same_items_only(tmp_path):
    def line(job, tests):
        return json.dumps({"job": {"id": job, "status": "ok"}, "results": {job: tests}}) + "\n"
    same = {"outcomes": "110", "items_sha": "s1"}
    res = tmp_path / "r.jsonl"
    res.write_text(line("a", {"en/code": same, "en/math": {"outcomes": "1", "items_sha": "x"}, "en/old": {"acc": 1}})
                   + line("b", {"en/code": {"outcomes": "100", "items_sha": "s1"},
                                "en/math": {"outcomes": "1", "items_sha": "y"}}), encoding="utf-8")
    rows = paired.compare(paired.latest(res, "a"), paired.latest(res, "b"))
    assert rows["en/math"] == {"skipped": "different items"} and "en/old" not in rows
    assert rows["en/code"]["a_only"] == 1 and rows["en/code"]["b_only"] == 0
