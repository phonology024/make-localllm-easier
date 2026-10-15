"""winget and Homebrew files (#46, #106): the formula's hash is the tarball PyPI actually serves."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("manifests", ROOT / "packaging" / "manifests.py")
manifests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manifests)

PYPI_071 = {"urls": [
    {"packagetype": "bdist_wheel", "digests": {"sha256": "cbbda34b6e172069abc78884b6d77f9b870c5c7a0f5942198d5048ae8195be28"}},
    {"packagetype": "sdist", "digests": {"sha256": "fb201722ae8af0a97afabeb486ea520c7dbff5e62fd4788937f93ccc8b6c4ec8"}}]}


def test_formula_hash_is_the_one_pypi_serves():
    sha = manifests.pypi_sdist_sha("0.7.1", get=lambda url: PYPI_071)
    assert sha == "fb201722ae8af0a97afabeb486ea520c7dbff5e62fd4788937f93ccc8b6c4ec8"           # the sdist, not the wheel
    rb = manifests.formula("0.7.1", sha)
    assert f'sha256 "{sha}"' in rb and "make_localllm_easier-0.7.1.tar.gz" in rb


def test_waits_for_the_publish_job_then_gives_up_clearly():
    answers = iter([OSError("404"), {"urls": []}, PYPI_071])
    t = [0.0]
    slept = []

    def get(url):
        a = next(answers)
        if isinstance(a, Exception):
            raise a
        return a

    sha = manifests.pypi_sdist_sha("0.7.1", get=get, wait_s=120, sleep=lambda s: (slept.append(s), t.__setitem__(0, t[0] + s)),
                                   clock=lambda: t[0])
    assert sha.startswith("fb2017") and slept == [30, 30]
    with pytest.raises(SystemExit, match="no sdist for 9.9.9"):
        manifests.pypi_sdist_sha("9.9.9", get=lambda url: {"urls": []}, wait_s=0)


def test_the_committed_0_7_1_submission_files_match_the_release():
    sub = ROOT / "packaging" / "submissions" / "0.7.1"
    rb = (sub / "homebrew" / "localllm.rb").read_text(encoding="utf-8")
    assert 'sha256 "fb201722ae8af0a97afabeb486ea520c7dbff5e62fd4788937f93ccc8b6c4ec8"' in rb
    inst = (sub / "winget" / "phonology024.localllm.installer.yaml").read_text(encoding="utf-8")
    # localllm-windows-x64.exe in the v0.7.1 release's SHA256SUMS
    assert "InstallerSha256: E260B76853335DA3FAC8401CD29DA5EC3EC496AA3B0B7B552AE0C645EBD5F413" in inst
    assert "releases/download/v0.7.1/localllm-windows-x64.exe" in inst
    for name in ("phonology024.localllm.yaml", "phonology024.localllm.locale.en-US.yaml"):
        assert "PackageVersion: 0.7.1" in (sub / "winget" / name).read_text(encoding="utf-8")
