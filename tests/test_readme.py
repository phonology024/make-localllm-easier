"""The README and docs must show what the code prints and what we measured (#80)."""
import contextlib
import io
import re
import sys
from pathlib import Path

from localllm import bench, cli, runtime, tune

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import headline  # noqa: E402

README = (ROOT / "README.md").read_text(encoding="utf-8")
HOW = (ROOT / "docs" / "how-it-works.md").read_text(encoding="utf-8")      # the doctor sample lives here
# the PC the README sample comes from: RX 9070 XT (16304 MiB as llama.cpp lists it), 32 GB RAM, Thai Windows
DEV = {"id": "Vulkan0", "name": "AMD Radeon RX 9070 XT", "total_gb": 16304 / 1024, "free_gb": 15.0}


def _doctor(monkeypatch) -> str:
    monkeypatch.setattr(cli, "_machine", lambda: (Path("llama-server"), [DEV], DEV, 32.0))
    monkeypatch.setattr(runtime, "ram_available_gb", lambda: 17.5)
    monkeypatch.setattr(tune, "load", lambda: {})
    monkeypatch.setattr(runtime, "build_of", lambda server: None)
    monkeypatch.setattr(bench, "system_language", lambda: "th")
    monkeypatch.setattr("platform.system", lambda: "Windows")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cli.cmd_doctor(None)
    return out.getvalue()


def test_readme_doctor_sample_matches_real_output(monkeypatch):
    sample = HOW.split("## What `localllm doctor` tells you", 1)[1].split("```", 2)[1].strip("\n")
    real = _doctor(monkeypatch).split("\nRun it:", 1)[0].rstrip("\n")
    assert sample == real, "doctor sample is stale: paste the output of this test's _doctor() into docs/how-it-works.md"


def test_readme_and_docs_relative_links_resolve():
    """The README was split into docs/ pages (2026-10-14): every relative link still lands on a file."""
    for page in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        text = re.sub(r"```.*?```", "", page.read_text(encoding="utf-8"), flags=re.S)
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if re.match(r"(https?:|mailto:|#)", target):
                continue
            path = (page.parent / target.split("#", 1)[0]).resolve()
            assert path.exists(), f"{page.name}: broken link {target}"


def test_readme_ram_table_matches_measurements():
    for name, n in headline.numbers().items():
        row = next(line for line in README.splitlines() if line.startswith(f"| {name} |"))
        cells = [c.strip(" *") for c in row.strip("|").split("|")]
        assert cells[1] == f"{n['default_gb']:.2f} GB RAM" and cells[2] == f"{n['localllm_gb']:.2f} GB", row
        speeds = [float(s) for s in re.findall(r"[\d.]+", cells[3])]
        assert all(abs(s - n["localllm_tok_s"]) <= 1 and abs(s - n["default_tok_s"]) <= 1 for s in speeds), row
