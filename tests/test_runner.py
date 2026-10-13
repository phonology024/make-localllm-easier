"""The measurement runner runs jobs, writes report-shaped lines, and resumes (#91)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import runner  # noqa: E402
import build_leaderboard  # noqa: E402

STATE = {"time": "t", "gpu": "AMD Radeon RX 9070 XT", "vram_gb": 15.9, "vram_free_gb": 15.0, "dev_id": "Vulkan0",
         "ram_gb": 32.0, "ram_free_gb": 20.0, "driver": "x"}


class Proc:
    def __init__(self, *a, **k):
        self.cmd = a[0]

    def poll(self):
        return None

    def terminate(self):
        pass

    def wait(self, t=None):
        return 0


def test_validate_manifest():
    assert runner.validate({"llama_cpp": "b1", "jobs": [{"id": "a", "model": "qwen3.8-27b-q3"}]}) == []
    why = runner.validate({"jobs": [{"id": "a", "model": "nope"}, {"id": "a", "file": "x.gguf"}]})
    assert len(why) == 3


def test_run_job_returns_a_report_the_leaderboard_accepts(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "state", lambda server: STATE)
    monkeypatch.setattr(runner, "wait_up", lambda proc, url: None)
    monkeypatch.setattr(runner, "decode_speed", lambda url: 88.5)
    monkeypatch.setattr(runner.bench, "run", lambda url, name, langs, limit, suites: {"th/regional": {"acc": 65.0}})
    job = {"id": "j1", "model": "gemma4-26b-a4b-qat", "args": ["-ctk", "q4_0"], "suites": "regional", "langs": "th"}
    rec = runner.run_job(job, "b11476", Path("llama-server"), lambda k: tmp_path / "g.gguf", popen=Proc)
    assert rec["tune"]["g.gguf"]["b11476"]["tok_s"] == 88.5 and rec["results"]["j1"]["th/regional"]["acc"] == 65.0
    assert rec["job"]["args_full"][-2:] == ["-ctk", "q4_0"] and rec["job"]["state_before"]["ram_free_gb"] == 20.0
    assert build_leaderboard.validate(rec) == []


def test_done_ids_skip_finished_and_retry_failed(tmp_path):
    f = tmp_path / "r.jsonl"
    f.write_text(json.dumps({"job": {"id": "a", "status": "ok"}}) + "\n" +
                 json.dumps({"job": {"id": "b", "status": "failed"}}) + "\n")
    assert runner.done_ids(f) == {"a"} and runner.done_ids(tmp_path / "none.jsonl") == set()


def test_example_manifest_is_valid():
    m = json.loads((Path(runner.__file__).with_name("runner_example.json")).read_text(encoding="utf-8"))
    assert runner.validate(m) == []


def test_vision_and_memory_options(tmp_path, monkeypatch):
    import localllm.cli
    monkeypatch.setattr(runner, "state", lambda server: STATE)
    monkeypatch.setattr(runner, "wait_up", lambda proc, url: None)
    monkeypatch.setattr(runner, "decode_speed", lambda url: 50.0)
    monkeypatch.setattr(localllm.cli, "_mmproj_path", lambda key: tmp_path / "mmproj.gguf")
    monkeypatch.setattr(runner.runtime, "process_memory", lambda pid: {"ram_gb": 2.0, "vram_gb": 14.0, "spill_gb": 0.5})
    Proc.pid = 123
    try:
        job = {"id": "v", "model": "gemma4-26b-a4b-qat", "vision": True, "memory": True}
        rec = runner.run_job(job, "b11476", Path("llama-server"), lambda k: tmp_path / "g.gguf", popen=Proc)
    finally:
        del Proc.pid
    assert rec["job"]["args_full"][-2] == "--mmproj"
    assert rec["job"]["memory"]["loaded"]["vram_gb"] == 14.0 and rec["job"]["memory"]["after"]["spill_gb"] == 0.5


def test_speed_is_dropped_when_the_model_spilled(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "state", lambda server: STATE)
    monkeypatch.setattr(runner, "wait_up", lambda proc, url: None)
    monkeypatch.setattr(runner, "decode_speed", lambda url: 29.2)
    monkeypatch.setattr(runner.runtime, "process_memory", lambda pid: {"ram_gb": 5.3, "vram_gb": 6.8, "spill_gb": 7.4})
    Proc.pid = 7
    try:
        rec = runner.run_job({"id": "s", "model": "gemma4-26b-a4b-qat"}, "b11476", Path("llama-server"),
                             lambda k: tmp_path / "g.gguf", popen=Proc)
    finally:
        del Proc.pid
    assert rec["tune"] == {} and "spilled 7.4 GB" in rec["job"]["notes"][0]
