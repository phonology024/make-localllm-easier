import json
import sys
import urllib.parse
from pathlib import Path

from localllm import report, runtime

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import merge_reports  # noqa: E402

DEV = {"id": "Vulkan0", "name": "AMD Radeon RX 9070 XT", "total_gb": 15.9, "free_gb": 15.0}


def _home(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "HOME", tmp_path)
    monkeypatch.setattr(report, "driver", lambda: "32.0.21001")
    monkeypatch.setattr(runtime, "ram_available_gb", lambda: 28.0)
    (tmp_path / "tune.json").write_text(json.dumps({
        "AMD Radeon RX 9070 XT|Qwen3.8-27B-UD-Q3_K_XL.gguf|b11457": {
            "env": {"GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM": "1"}, "mtp": 2, "tok_s": 55.0}}))
    (tmp_path / "results.json").write_text(json.dumps({"qwen3.8-27b-q3": {
        "en/global": {"acc": 81.5, "correct": 326, "n": 400}, "th/regional": {"acc": 67.1, "n": 900},
        "_meta": {"model": "qwen3.8-27b-q3", "seconds": 600}}}))


def test_scrub_removes_user_paths_and_secrets():
    bits = ["/home/somchai", "C:\\Users\\somchai", "somchai"]
    out = report.scrub({"model": "/home/somchai/models/x.gguf", "note": "run by somchai",
                        "win": "C:\\Users\\somchai\\.localllm\\llama.cpp\\b11457", "api_key": "sk-123",
                        "nested": [{"Authorization": "Bearer x", "ok": 1}]}, bits)
    assert out == {"model": "x.gguf", "note": "run by <user>", "win": "b11457", "nested": [{"ok": 1}]}
    common = report.scrub({"a": "the user asked", "b": "username users userland", "c": "in /home/user/x, user's"},
                          ["/home/user", "user"])
    assert common == {"a": "the <user> asked", "b": "username users userland", "c": "in ~/x, <user>'s"}


def test_collect_reports_machine_tune_and_scores_without_personal_data(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    server = tmp_path / "llama.cpp" / "b11457" / "llama-server"
    rep = report.collect(server=server, devs=[DEV], ram_gb=32.0, lang="th")
    assert rep["gpu"] == DEV["name"] and rep["vram_gb"] == 15.9 and rep["ram_gb"] == 32.0
    assert rep["llama_cpp"] == "b11457" and rep["pick"] and rep["driver"] == "32.0.21001"
    assert rep["tune"]["Qwen3.8-27B-UD-Q3_K_XL.gguf"]["b11457"]["tok_s"] == 55.0
    assert rep["results"]["qwen3.8-27b-q3"]["th/regional"]["acc"] == 67.1
    assert str(tmp_path) not in json.dumps(rep)


def test_issue_url_carries_the_report_or_asks_for_the_file(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    rep = report.collect(server=tmp_path / "b1" / "llama-server", devs=[DEV], ram_gb=32.0, lang="th")
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(report.issue_url(rep)).query)
    assert "RX 9070 XT" in q["title"][0] and '"tok_s": 55.0' in q["body"][0]
    rep["results"] = {f"model-{i}": {f"xx/s{j}": {"acc": 50.0} for j in range(40)} for i in range(20)}
    url = report.issue_url(rep, tmp_path / "report-2026-10-08.json")
    assert len(url) <= report.MAX_URL and "report-2026-10-08.json" in urllib.parse.unquote_plus(url)


def test_merge_reports_builds_one_row_per_gpu_model_build(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    a = report.collect(server=tmp_path / "b11457" / "llama-server", devs=[DEV], ram_gb=32.0, lang="th")
    b = {**a, "date": "2026-11-01", "gpu": "NVIDIA GeForce RTX 4060", "vram_gb": 8.0,
         "tune": {"gemma-4-E4B.gguf": {"b11500": {"env": {"GGML_CUDA_GRAPH_OPT": "1"}, "mtp": 0, "tok_s": 70.0}}},
         "results": {}}
    issue = tmp_path / "issue.md"
    issue.write_text(f"**GPU:** ...\n\n```json\n{json.dumps(b)}\n```\n")
    rows = merge_reports.rows([a, merge_reports.load(issue)])
    assert [(r["gpu"], r["tok_s"]) for r in rows] == \
        [("AMD Radeon RX 9070 XT", 55.0), ("NVIDIA GeForce RTX 4060", 70.0)]
    assert rows[0]["scores"] == {"en/global": 81.5, "th/regional": 67.1}                # eval matched to tuned model
    assert rows[0]["settings"] == "GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1, MTP 2"
    readme = f"x\n{merge_reports.START}\nold\n{merge_reports.END}\ny\n"
    new = merge_reports.update(readme, merge_reports.table(rows))
    assert new.startswith("x\n") and new.endswith("y\n") and "RTX 4060" in new and "old" not in new
    assert merge_reports.update(new, merge_reports.table(rows)) == new                       # stable
