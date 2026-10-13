"""Build a self-contained `localllm` executable for this OS with PyInstaller (#46). No Python needed to run it.

  python -m pip install pyinstaller==6.11.1
  python packaging/build_exe.py            # -> dist/localllm(.exe), then a smoke test of the binary

The release workflow (.github/workflows/release.yml) runs this on Windows, macOS and Linux for every v* tag.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEP = ";" if os.name == "nt" else ":"
DATA = ["localllm/ui.html", "localllm/platforms.json"]


def main() -> None:
    args = [sys.executable, "-m", "PyInstaller", "--onefile", "--clean", "--noconfirm", "--name", "localllm",
            "--distpath", str(ROOT / "dist"), "--workpath", str(ROOT / "build" / "pyinstaller"),
            "--specpath", str(ROOT / "build"), "--paths", str(ROOT)]
    for d in DATA:
        args += ["--add-data", f"{ROOT / d}{SEP}localllm"]
    subprocess.run(args + [str(ROOT / "packaging" / "entry.py")], check=True)
    exe = ROOT / "dist" / ("localllm.exe" if os.name == "nt" else "localllm")
    for cmd in (["--version"], ["list"], ["connect", "cline"]):          # no GPU or network needed
        out = subprocess.run([str(exe), *cmd], capture_output=True, text=True, timeout=120)
        print(f"$ localllm {' '.join(cmd)}\n{out.stdout[:400]}")
        if out.returncode:
            sys.exit(f"smoke test failed: {out.stderr[-800:]}")
    print(f"ok: {exe} ({exe.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
