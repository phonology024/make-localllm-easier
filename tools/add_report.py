"""Turn a shared `localllm report` (a "Share a result" issue, or a saved JSON) into a file under reports/ (#101).

  python tools/add_report.py --issue 130            # fetch the issue body with the GitHub CLI
  python tools/add_report.py ISSUE_BODY.md          # or a saved issue body / report JSON
  python tools/add_report.py ISSUE_BODY.md --check  # only say whether it would be accepted (CI comments this)

The report is checked with the leaderboard's own rules (schema, ranges, scores 0-100) plus the privacy check
(no secret-looking fields, absolute paths, home folders or e-mail addresses). Nothing is fixed silently: a report
that fails is refused with the reasons, for the person who shared it to correct. An accepted report is written as
reports/<gpu>-<date>.json (never overwriting another one); then run `python tools/build_leaderboard.py`.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import build_leaderboard  # noqa: E402

REPORTS = ROOT / "reports"
JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)


def parse(text: str) -> dict:
    """The report from an issue body (first ```json block) or from a bare JSON file."""
    m = JSON_BLOCK.search(text)
    rep = json.loads(m.group(1) if m else text)
    if not isinstance(rep, dict):
        raise ValueError("the report is not a JSON object")
    return rep


def file_name(rep: dict, folder: Path = REPORTS) -> Path:
    gpu = re.sub(r"[^a-z0-9]+", "-", str(rep.get("gpu") or "cpu-only").lower()).strip("-")[:40] or "gpu"
    date = rep.get("date") if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(rep.get("date"))) else "undated"
    path, n = folder / f"{gpu}-{date}.json", 2
    while path.exists():
        path, n = folder / f"{gpu}-{date}-{n}.json", n + 1
    return path


def check(text: str) -> tuple[dict | None, list[str]]:
    try:
        rep = parse(text)
    except ValueError as e:
        return None, [f"no readable report JSON ({e})"]
    return rep, build_leaderboard.validate(rep)


def issue_body(number: int) -> str:
    out = subprocess.run(["gh", "issue", "view", str(number), "--json", "body", "--jq", ".body"],
                         capture_output=True, text=True, encoding="utf-8", check=True)
    return out.stdout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("file", nargs="?", type=Path)
    src.add_argument("--issue", type=int)
    ap.add_argument("--check", action="store_true", help="only report whether it would be accepted")
    a = ap.parse_args()
    text = issue_body(a.issue) if a.issue else a.file.read_text(encoding="utf-8")
    rep, why = check(text)
    if why:
        print("Not accepted yet:\n" + "\n".join(f"- {w}" for w in why))
        sys.exit(1)
    if a.check:
        print(f"Looks good: {rep.get('gpu') or 'CPU only'}, {rep.get('os')}, llama.cpp {rep.get('llama_cpp')}. "
              "A maintainer adds it to the leaderboard.")
        return
    out = file_name(rep)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out.relative_to(ROOT)}; now run python tools/build_leaderboard.py and commit both")


if __name__ == "__main__":
    main()
