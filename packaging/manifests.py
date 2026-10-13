"""Write the winget manifests and the Homebrew formula for a release (#46).

  python packaging/manifests.py 0.7.0 --exe dist/localllm-windows-x64.exe --sdist dist/make_localllm_easier-0.7.0.tar.gz

Hashes come from the files you pass (the same files that are uploaded), so a manifest can't point at a file it wasn't
made from. Output goes to build/manifests/; how to submit it is in packaging/README.md.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GH = "https://github.com/phonology024/make-localllm-easier"
ID = "phonology024.localllm"
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("version")
    ap.add_argument("--exe", type=Path, required=True, help="the Windows x64 executable that is uploaded")
    ap.add_argument("--sdist", type=Path, required=True, help="the source tarball that is on PyPI")
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "manifests")
    a = ap.parse_args()
    wdir = a.out / "winget" / "manifests" / "p" / "phonology024" / "localllm" / a.version
    wdir.mkdir(parents=True, exist_ok=True)
    for name, text in winget(a.version, sha256(a.exe)).items():
        (wdir / name).write_text(text, encoding="utf-8", newline="\n")
    (a.out / "homebrew").mkdir(parents=True, exist_ok=True)
    (a.out / "homebrew" / "localllm.rb").write_text(formula(a.version, sha256(a.sdist)), encoding="utf-8", newline="\n")
    print(f"wrote {wdir} and {a.out / 'homebrew' / 'localllm.rb'}")


if __name__ == "__main__":
    main()
