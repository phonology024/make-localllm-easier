"""Write the winget manifests and the Homebrew formula for a release (#46).

  python packaging/manifests.py 0.7.1 --exe dist/localllm-windows-x64.exe [--pypi-wait 900]

The winget hash comes from the executable you pass (the same file that is uploaded). The Homebrew formula installs
the source tarball from PyPI, so its hash is the one PyPI lists for that version (waiting up to --pypi-wait seconds for
the publish job): a tarball built again elsewhere differs byte for byte, and 0.7.1's first formula failed that way.
`--sdist FILE` still works offline, and is checked against PyPI when PyPI has the version.
Output goes to build/manifests/; how to submit it is in packaging/README.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GH = "https://github.com/phonology024/make-localllm-easier"
ID = "phonology024.localllm"
PYPI = "https://pypi.org/pypi/make-localllm-easier/{version}/json"
DESC = "One command runs the best local AI your PC can handle, picked from measured scores in your language"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def winget(version: str, exe_sha: str) -> dict[str, str]:
    head = f"# yaml-language-server: $schema=https://aka.ms/winget-manifest.{{kind}}.1.6.0.schema.json\n"
    common = f"PackageIdentifier: {ID}\nPackageVersion: {version}\n"
    return {
        f"{ID}.yaml": head.format(kind="version") + common +
        "DefaultLocale: en-US\nManifestType: version\nManifestVersion: 1.6.0\n",
        f"{ID}.installer.yaml": head.format(kind="installer") + common +
        "InstallerType: portable\nCommands:\n  - localllm\nInstallers:\n  - Architecture: x64\n"
        f"    InstallerUrl: {GH}/releases/download/v{version}/localllm-windows-x64.exe\n"
        f"    InstallerSha256: {exe_sha.upper()}\nManifestType: installer\nManifestVersion: 1.6.0\n",
        f"{ID}.locale.en-US.yaml": head.format(kind="defaultLocale") + common +
        "PackageLocale: en-US\nPublisher: phonology024\nPackageName: localllm (make-localllm-easier)\n"
        f"PackageUrl: {GH}\nLicense: MIT\nLicenseUrl: {GH}/blob/main/LICENSE\nMoniker: localllm\n"
        f"ShortDescription: {DESC}\nTags:\n  - llm\n  - local-llm\n  - llama-cpp\n  - ai\n"
        f"ReleaseNotesUrl: {GH}/releases/tag/v{version}\nManifestType: defaultLocale\nManifestVersion: 1.6.0\n",
    }


def formula(version: str, sdist_sha: str) -> str:
    return f'''class Localllm < Formula
  include Language::Python::Virtualenv

  desc "{DESC}"
  homepage "{GH}"
  url "https://files.pythonhosted.org/packages/source/m/make-localllm-easier/make_localllm_easier-{version}.tar.gz"
  sha256 "{sdist_sha}"
  license "MIT"

  depends_on "python@3.12"

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match version.to_s, shell_output("#{{bin}}/localllm --version")
    assert_match "qwen3.8-27b-q3", shell_output("#{{bin}}/localllm list")
  end
end
'''


def _get(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def pypi_sdist_sha(version: str, get=_get, wait_s: float = 0, sleep=time.sleep, clock=time.monotonic) -> str:
    """SHA-256 of the sdist PyPI serves for `version`; polls every 30 s for up to wait_s while it is being published."""
    deadline = clock() + wait_s
    while True:
        try:
            found = [u["digests"]["sha256"] for u in get(PYPI.format(version=version))["urls"]
                     if u.get("packagetype") == "sdist"]
            if found:
                return found[0]
            why = "no source tarball listed"
        except (OSError, ValueError, KeyError) as e:
            why = str(e) or type(e).__name__
        if clock() >= deadline:
            raise SystemExit(f"PyPI has no sdist for {version} yet ({why}); publish it first or pass --sdist")
        sleep(30)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("version")
    ap.add_argument("--exe", type=Path, required=True, help="the Windows x64 executable that is uploaded")
    ap.add_argument("--sdist", type=Path, help="the source tarball as uploaded to PyPI (default: ask PyPI)")
    ap.add_argument("--pypi-wait", type=float, default=0, help="seconds to wait for the version to appear on PyPI")
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "manifests")
    a = ap.parse_args()
    wdir = a.out / "winget" / "manifests" / "p" / "phonology024" / "localllm" / a.version
    wdir.mkdir(parents=True, exist_ok=True)
    for name, text in winget(a.version, sha256(a.exe)).items():
        (wdir / name).write_text(text, encoding="utf-8", newline="\n")
    (a.out / "homebrew").mkdir(parents=True, exist_ok=True)
    sdist_sha = sha256(a.sdist) if a.sdist else pypi_sdist_sha(a.version, wait_s=a.pypi_wait)
    if a.sdist:
        try:
            listed = pypi_sdist_sha(a.version)
        except SystemExit:
            listed = sdist_sha                            # not on PyPI yet: nothing to compare with
        if listed != sdist_sha:
            raise SystemExit(f"{a.sdist} is not the tarball PyPI serves for {a.version} ({listed}); drop --sdist")
    (a.out / "homebrew" / "localllm.rb").write_text(formula(a.version, sdist_sha), encoding="utf-8", newline="\n")
    print(f"wrote {wdir} and {a.out / 'homebrew' / 'localllm.rb'}")


if __name__ == "__main__":
    main()
