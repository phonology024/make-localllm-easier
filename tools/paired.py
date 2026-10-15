"""Compare two runner jobs item by item (exact McNemar test), not only by their totals (#65, #168).

  python tools/paired.py JOB_A JOB_B [--results tools/runner_results.jsonl]

Uses the per-item outcomes bench.run records for right/wrong suites ("outcomes" + "items_sha"). Only tests both jobs
ran on the same items (same items_sha) are compared. Two models on the same 800 questions mostly get the same ones
right; McNemar looks only at the questions where they differ, so it can tell a real gap from noise where two
separate margins cannot.
"""
from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

RESULTS = Path(__file__).with_name("runner_results.jsonl")


def latest(results: Path, job_id: str) -> dict:
    """{test: {"outcomes", "items_sha", ...}} from the newest line of this job that has scores."""
    found = {}
    for line in results.read_text(encoding="utf-8").splitlines():
        r = json.loads(line) if line.strip() else {}
        if r.get("job", {}).get("id") == job_id and r.get("results", {}).get(job_id):
            found = r["results"][job_id]
    return {k: v for k, v in found.items() if isinstance(v, dict) and "outcomes" in v}


def mcnemar(a: str, b: str) -> dict:
    """a, b: "1"/"0" per item. b_only / a_only: items only one of them got right; p: exact two-sided."""
    a_only = sum(x == "1" and y == "0" for x, y in zip(a, b))
    b_only = sum(x == "0" and y == "1" for x, y in zip(a, b))
    n, k = a_only + b_only, min(a_only, b_only)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"n": len(a), "a": a.count("1"), "b": b.count("1"), "a_only": a_only, "b_only": b_only, "p": p}


def compare(ra: dict, rb: dict) -> dict:
    out = {}
    for test in sorted(set(ra) & set(rb)):
        x, y = ra[test], rb[test]
        if x["items_sha"] != y["items_sha"] or len(x["outcomes"]) != len(y["outcomes"]):
            out[test] = {"skipped": "different items"}
        else:
            out[test] = mcnemar(x["outcomes"], y["outcomes"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("job_a")
    ap.add_argument("job_b")
    ap.add_argument("--results", type=Path, default=RESULTS)
    a = ap.parse_args()
    ra, rb = latest(a.results, a.job_a), latest(a.results, a.job_b)
    rows = compare(ra, rb)
    if not rows:
        sys.exit(f"no test with per-item outcomes in both {a.job_a} and {a.job_b}")
    for test, r in rows.items():
        if "skipped" in r:
            print(f"{test:16} skipped: {r['skipped']}")
            continue
        verdict = "different" if r["p"] < 0.05 else "tie"
        print(f"{test:16} {a.job_a} {r['a']}/{r['n']}  {a.job_b} {r['b']}/{r['n']}  "
              f"only A right {r['a_only']}, only B right {r['b_only']}  p={r['p']:.3f}  {verdict}")


if __name__ == "__main__":
    main()
