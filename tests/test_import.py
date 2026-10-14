"""`localllm import` (#110): Ollama / LM Studio / Jan models are matched by SHA-256 and hard-linked; the other apps'
files never change; dry run by default."""
import hashlib
import json
import os

import pytest

from localllm import catalog, importer

BODY = b"GGUF" + bytes(range(256)) * 64
SHA = hashlib.sha256(BODY).hexdigest()


@pytest.fixture
def cat(monkeypatch):
    monkeypatch.setattr(catalog, "MODELS", {"tiny": {"file": "tiny-Q4.gguf", "gb": len(BODY) / 2**30, "sha256": SHA}})


def _ollama(root, name="tiny", digest=SHA, data=BODY):
    (root / "blobs").mkdir(parents=True)
    (root / "blobs" / f"sha256-{digest}").write_bytes(data)
    man = root / "manifests" / "registry.ollama.ai" / "library" / name
    man.mkdir(parents=True)
    (man / "latest").write_text(json.dumps({"layers": [
        {"mediaType": "application/vnd.ollama.image.license", "digest": "sha256:" + "0" * 64},
        {"mediaType": "application/vnd.ollama.image.model", "digest": f"sha256:{digest}"}]}))


def test_ollama_blobs_are_matched_without_hashing(tmp_path, cat):
    _ollama(tmp_path / "ollama")
    found = importer.scan_ollama(tmp_path / "ollama")
    assert [(f.app, f.name, f.sha256) for f in found] == [("Ollama", "tiny:latest", SHA)]
    rows = importer.match(found, hash_file=lambda p: pytest.fail("hashed an Ollama blob"))
    assert rows[0][1] == "tiny"


def test_lm_studio_files_are_hashed_only_near_a_catalog_size(tmp_path, cat, monkeypatch):
    monkeypatch.setattr(importer, "SIZE_SLACK_GB", 0)
    d = tmp_path / "lms" / "pub" / "repo"
    d.mkdir(parents=True)
    (d / "renamed.gguf").write_bytes(BODY)
    (d / "big-other.gguf").write_bytes(b"x" * (len(BODY) * 3))
    (d / "mmproj-F16.gguf").write_bytes(BODY)                     # projectors are not models
    hashed = []
    rows = importer.match(importer.scan_folders([("LM Studio", tmp_path / "lms")]),
                          hash_file=lambda p: hashed.append(p.name) or importer.sha256_of(p))
    assert hashed == ["renamed.gguf"]
    assert {f.name: k for f, k in rows} == {"big-other.gguf": None, "renamed.gguf": "tiny"}


def test_dry_run_changes_nothing_and_apply_hard_links(tmp_path, cat):
    _ollama(tmp_path / "ollama")
    blob = tmp_path / "ollama" / "blobs" / f"sha256-{SHA}"
    before = (blob.stat().st_mtime, blob.read_bytes())
    models = tmp_path / "localllm" / "models"
    rows = importer.plan(importer.match(importer.scan_ollama(tmp_path / "ollama")), models)
    text = importer.describe(rows, applied=False)
    assert rows[0]["action"] == "link" and "would link" in text and "dry run: nothing changed" in text
    assert not models.exists()
    importer.apply(rows)
    dest = models / "tiny-Q4.gguf"
    assert rows[0]["action"] == "linked" and dest.read_bytes() == BODY
    assert os.stat(dest).st_ino == os.stat(blob).st_ino and os.stat(blob).st_nlink == 2      # a link, not a copy
    assert (blob.stat().st_mtime, blob.read_bytes()) == before                               # Ollama's file as it was
    again = importer.plan(importer.match(importer.scan_ollama(tmp_path / "ollama")), models)
    assert again[0]["action"] == "have"


def test_unknown_and_other_drive_rows(tmp_path, cat, monkeypatch):
    _ollama(tmp_path / "ollama", name="mystery", digest="ab" * 32, data=b"GGUF other")
    _ollama(tmp_path / "o2")
    rows = importer.plan(importer.match(importer.scan_ollama(tmp_path / "ollama") + importer.scan_ollama(tmp_path / "o2")),
                         tmp_path / "m")
    assert [r["action"] for r in rows] == ["unmeasured", "link"]
    monkeypatch.setattr(importer, "_same_volume", lambda a, b: False)
    rows = importer.plan(importer.match(importer.scan_ollama(tmp_path / "o2")), tmp_path / "m")
    text = importer.describe(rows, applied=False)
    assert rows[0]["action"] == "other-drive" and "LOCALLLM_MODELS=" in text


def test_nothing_found():
    assert importer.describe([], applied=False) == "No models from Ollama, LM Studio or Jan found on this PC."
