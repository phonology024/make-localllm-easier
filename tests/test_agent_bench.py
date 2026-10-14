"""The #96 task set stays valid: unique ids, every Thai task has a check, code ids exist in the EvalPlus list."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_task_set_is_well_formed():
    d = json.loads((ROOT / "tools" / "agent_bench" / "tasks.json").read_text(encoding="utf-8"))
    repo = {t["id"]: t for t in d["repo_tasks"]}
    ids = list(repo) + [t["id"] for t in d["thai_tasks"]]
    assert len(ids) == len(set(ids)) and d["version"] == 1
    for t in d["thai_tasks"]:
        assert t.get("check") or t.get("same_check_as") in repo, t["id"]
        assert any("฀" <= c <= "๿" for c in t["prompt"]), t["id"]
    for t in d["repo_tasks"]:
        compile(t["check"], t["id"], "exec")
    assert len(d["code_tasks"]["ids"]) == 30 and len(set(d["code_tasks"]["ids"])) == 30
    assert (ROOT / "tools" / "agent_bench" / "fixture" / "inventory" / "store.py").exists()
