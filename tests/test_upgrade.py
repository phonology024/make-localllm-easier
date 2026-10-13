"""#95: the llama.cpp bump process - smoke-test verdicts, build-aware server lookup, stale-score notes."""
import sys
from pathlib import Path

from localllm import catalog, runtime

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import smoke_build  # noqa: E402

OK = {"loaded": True, "answer": "Paris.", "tok_s": 90.0, "spill_gb": 0.6}


def test_verdict_passes_a_build_within_tolerance():
    assert smoke_build.verdict({**OK, "tok_s": 87.0}, OK, 0.05) == []


def test_verdict_names_every_problem():
    assert smoke_build.verdict({"loaded": False, "error": "boom"}, OK, 0.05) == ["does not load: boom"]
    why = smoke_build.verdict({**OK, "answer": "Lyon", "tok_s": 80.0, "spill_gb": 3.0}, OK, 0.05)
    assert len(why) == 3 and "-11%" in why[2]


def test_find_server_uses_the_wanted_build(tmp_path, monkeypatch):
    for b in ("b11476", "b11999"):
        (tmp_path / "llama.cpp" / b).mkdir(parents=True)
        (tmp_path / "llama.cpp" / b / runtime.EXE).write_text("")
    monkeypatch.setattr(runtime, "HOME", tmp_path)
    monkeypatch.delenv("LOCALLLM_LLAMA_SERVER", raising=False)
    monkeypatch.delenv("LOCALLLM_LLAMA_BUILD", raising=False)
    assert runtime.find_server().parent.name == "b11476"          # the pin, though a newer one is unpacked
    monkeypatch.setenv("LOCALLLM_LLAMA_BUILD", "b11999")
    assert runtime.find_server().parent.name == "b11999"
    assert runtime.build_of(runtime.find_server()) == "b11999"


def test_build_note_only_when_the_builds_differ():
    key = "gemma4-26b-a4b-qat"
    assert catalog.build_note(key, catalog.MODELS[key]["build"]) is None
    assert catalog.build_note(key, None) is None
    assert "newer" in catalog.build_note(key, "b99999")
    assert all(m.get("build") for m in catalog.MODELS.values())
