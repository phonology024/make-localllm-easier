"""The published catalog JSON (#164): up to date with catalog.py and valid against its schema."""
import json
import sys
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_catalog as bc  # noqa: E402

V1 = ROOT / "data" / "catalog" / "v1"


def load(name: str) -> dict:
    return json.loads((V1 / name).read_text(encoding="utf-8"))


def test_committed_catalog_matches_catalog_py():
    assert (V1 / "catalog.json").read_text(encoding="utf-8") == bc.render(), \
        "data/catalog/v1/catalog.json is out of date: run python tools/build_catalog.py"


def test_catalog_is_valid_against_schema():
    schema = load("schema.json")
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(load("catalog.json"))


def test_every_catalog_model_is_published_with_its_numbers():
    data = load("catalog.json")
    by_id = {m["id"]: m for m in data["models"]}
    assert set(by_id) == set(bc.catalog.MODELS)
    for key, m in bc.catalog.MODELS.items():
        out = by_id[key]
        assert out["scores"] == m["scores"] and out["sha256"] == m["sha256"] and out["rev"] == m["rev"]
        assert out["memory"]["weights_gib"] == m["gb"]
        assert out["speed"][0]["decode_tok_s"] == m["tok_s_9070xt"]
    hw = {h["id"] for h in data["hardware"]}
    assert all(s["hardware"] in hw for m in data["models"] for s in m["speed"])


def test_schema_rejects_broken_entries():
    schema, good = load("schema.json"), load("catalog.json")
    v = jsonschema.Draft202012Validator(schema)
    m = good["models"][0]
    for bad in ({**m, "sha256": "xyz"}, {**m, "scores": {"english": 80}}, {**m, "scores": {"en/global": 140}},
                {**m, "format": "exe"}, {k: x for k, x in m.items() if k != "memory"}):
        assert not v.is_valid({**good, "models": [bad]})


def test_rev_changes_with_the_numbers(monkeypatch):
    before = bc.build()["catalog_rev"]
    key = next(iter(bc.catalog.MODELS))
    changed = {**bc.catalog.MODELS, key: {**bc.catalog.MODELS[key], "tok_s_9070xt": 1}}
    monkeypatch.setattr(bc.catalog, "MODELS", changed)
    assert bc.build()["catalog_rev"] != before
