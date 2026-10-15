"""Weekly traction snapshot (#148): one JSON line per ISO week in data/metrics.jsonl (the monthly table for ROADMAP
0.8 stays in tools/metrics.py / docs/metrics.md).

  python tools/traction.py                 # append this week's line (skips a week that is already there)
  python tools/traction.py --print         # show it, write nothing
  python tools/traction.py --out PATH      # another file (the workflow keeps it on the `metrics` branch)

What goes in, all public numbers except GitHub traffic (which needs push access; GITHUB_TOKEN in Actions has it):
  github    stars, forks, watchers, open issues + PRs
  traffic   clones and views: 14-day count and uniques (GitHub keeps 14 days, so weekly lines miss nothing)
  pypi      downloads, last 7 and 30 days, without mirrors (pypistats.org; it keeps ~180 days)
  releases  asset downloads per release tag, and the total
  issues    `localllm report` issues (label results) and issues opened by people outside the project
  reports   shared results that made it into data/leaderboard.json
A source that fails is recorded as null with its error, never guessed. Only GITHUB_TOKEN is read from the environment.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "phonology024/make-localllm-easier"
PACKAGE = "make-localllm-easier"
MAINTAINERS = ("phonology024",)
API = "https://api.github.com"


def http_json(url: str):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "localllm-traction"}
    token = os.environ.get("GITHUB_TOKEN")
    if token and url.startswith(API):
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
        return json.load(r)


def _try(fn):
    try:
        return fn(), None
    except Exception as e:  # noqa: BLE001 - one dead source must not lose the others
        return None, f"{type(e).__name__}: {e}"[:200]


def _search_count(query: str, get) -> int:
    return get(f"{API}/search/issues?q={urllib.parse.quote(query)}&per_page=1")["total_count"]


def snapshot(get=http_json, today: dt.date | None = None,
             leaderboard: Path = ROOT / "data" / "leaderboard.json") -> dict:
    today = today or dt.date.today()
    year, week, _ = today.isocalendar()

    def github():
        r = get(f"{API}/repos/{REPO}")
        return {"stars": r["stargazers_count"], "forks": r["forks_count"], "watchers": r["subscribers_count"],
                "open_issues": r["open_issues_count"]}

    def traffic():
        out = {}
        for kind in ("clones", "views"):
            t = get(f"{API}/repos/{REPO}/traffic/{kind}")
            out[kind] = {"count": t["count"], "uniques": t["uniques"]}
        return out

    def pypi():
        rows = get(f"https://pypistats.org/api/packages/{PACKAGE}/overall?mirrors=false")["data"]
        since = lambda days: (today - dt.timedelta(days=days)).isoformat()   # noqa: E731
        return {f"last_{n}d": sum(r["downloads"] for r in rows if since(n) < r["date"] <= today.isoformat())
                for n in (7, 30)}

    def releases():
        rel = get(f"{API}/repos/{REPO}/releases?per_page=100")
        per = {r["tag_name"]: sum(a["download_count"] for a in r.get("assets", [])) for r in rel}
        return {"per_tag": per, "total": sum(per.values())}

    def issues():
        outside = " ".join(f"-author:{m}" for m in MAINTAINERS)
        return {"report_issues": _search_count(f"repo:{REPO} is:issue label:results", get),
                "outside_issues": _search_count(f"repo:{REPO} is:issue {outside}", get)}

    def reports():
        return json.loads(leaderboard.read_text(encoding="utf-8")).get("reports", 0)

    line, errors = {"week": f"{year}-W{week:02d}", "date": today.isoformat()}, {}
    for name, fn in (("github", github), ("traffic", traffic), ("pypi", pypi), ("releases", releases),
                     ("issues", issues), ("reports", reports)):
        line[name], err = _try(fn)
        if err:
            errors[name] = err
    if errors:
        line["errors"] = errors
    return line


def append(line: dict, path: Path) -> bool:
    """Add the line unless this ISO week is already in the file. True if written."""
    weeks = set()
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            try:
                weeks.add(json.loads(raw)["week"])
            except (ValueError, KeyError, TypeError):
                continue
    if line["week"] in weeks:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(line, sort_keys=True) + "\n")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "metrics.jsonl")
    ap.add_argument("--print", action="store_true", help="show the snapshot, write nothing")
    a = ap.parse_args()
    line = snapshot()
    print(json.dumps(line, indent=1, sort_keys=True))
    if not a.print:
        print("written" if append(line, a.out) else f"{line['week']} is already in {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
