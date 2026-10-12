"""Monthly reach snapshot for ROADMAP 0.8 (#94): appends one row to docs/metrics.md.

  python tools/metrics.py            print the row
  python tools/metrics.py --write    append it to docs/metrics.md

Needs the GitHub CLI (`gh`, logged in) for issue counts. Sources: GitHub API (stars, forks, issues), pypistats.org
(PyPI downloads, last 30 days; pypistats keeps ~180 days of history, so take the snapshot every month).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = "phonology024/make-localllm-easier"
PACKAGE = "make-localllm-easier"
OUT = Path(__file__).resolve().parents[1] / "docs" / "metrics.md"
MAINTAINERS = {"phonology024"}


def _gh(*args: str):
    return json.loads(subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout)


def pypi_last_month() -> int | None:
    try:
        d = json.load(urllib.request.urlopen(f"https://pypistats.org/api/packages/{PACKAGE}/overall?mirrors=false",
                                             timeout=30))
    except OSError:
        return None
    cutoff = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 30 * 86400))
    return sum(r["downloads"] for r in d["data"] if r["date"] >= cutoff)


def snapshot() -> dict:
    repo = _gh("api", f"repos/{REPO}")
    issues = _gh("issue", "list", "-R", REPO, "--state", "all", "--limit", "1000", "--json", "author,labels")
    return {
        "date": time.strftime("%Y-%m-%d"),
        "stars": repo["stargazers_count"], "forks": repo["forks_count"],
        "pypi_30d": pypi_last_month(),
        "reports": sum(any(l["name"] == "results" for l in i["labels"]) for i in issues),
        "outside_issues": sum(i["author"]["login"] not in MAINTAINERS for i in issues),
    }


HEAD = ("| date | stars | forks | PyPI downloads (30 days) | `localllm report` issues | issues from outside the "
        "project |\n|---|---|---|---|---|---|\n")


def row(s: dict) -> str:
    return (f"| {s['date']} | {s['stars']} | {s['forks']} | {s['pypi_30d'] if s['pypi_30d'] is not None else '-'} "
            f"| {s['reports']} | {s['outside_issues']} |\n")


def main() -> None:
    line = row(snapshot())
    if "--write" not in sys.argv:
        print(line, end="")
        return
    text = OUT.read_text(encoding="utf-8")
    if HEAD not in text:
        sys.exit(f"{OUT} has no metrics table")
    OUT.write_text(text.rstrip("\n") + "\n" + line, encoding="utf-8", newline="\n")
    print(f"appended to {OUT}: {line}", end="")


if __name__ == "__main__":
    main()
