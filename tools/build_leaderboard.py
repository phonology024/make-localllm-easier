"""Build the public leaderboard data and the platform matrix from `localllm report` files (#73, #99).

  python tools/build_leaderboard.py [--reports reports/] [--check]

Inputs: the catalog (our own measurements, RX 9070 XT on Windows) and every report in `reports/` (saved JSON, or an
issue body with the JSON in a ```json block; see reports/README.md). Outputs, all regenerated together:

  data/leaderboard.json     one entry per (GPU, model, llama.cpp build), newest report wins, with flags
  localllm/platforms.json   OS x GPU vendor -> tested by maintainers / community reports (n) / untested (for `doctor`)
  docs/how-it-works.md      the platform table between <!-- platforms:start --> and <!-- platforms:end -->

A report that fails validation is skipped with the reason. --check exits 1 if any output would change (for CI).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from localllm import catalog  # noqa: E402
from localllm.report import os_family, vendor  # noqa: E402
import merge_reports  # noqa: E402

SCHEMA = 1                      # data/leaderboard.json format; bump on any breaking change
REPORT_SCHEMAS = {1}            # `localllm report` schemas we can read
START, END = "<!-- platforms:start -->", "<!-- platforms:end -->"
OSES = ["Windows", "Linux", "macOS"]
VENDORS = ["AMD", "NVIDIA", "Intel", "Apple"]
NA = {("Windows", "Apple"), ("Linux", "Apple")}
# what we run ourselves; anything else needs a community report
MAINTAINER = {"gpu": "AMD Radeon RX 9070 XT", "vram_gb": 15.9, "ram_gb": 32.0, "os": "Windows 11",
              "llama_cpp": "b11476"}
MAX_TOK_S = 400                 # decode faster than this on a 20B+ model is almost surely a typo or a wrong file
LIMITS = {"vram_gb": (0, 1024), "ram_gb": (0, 4096)}


def validate(rep: dict) -> list[str]:
    """Reasons to reject a report; empty when it is usable."""
    why = []
    if rep.get("schema") not in REPORT_SCHEMAS:
        why.append(f"unknown report schema {rep.get('schema')!r}")
    for k, (lo, hi) in LIMITS.items():
        v = rep.get(k)
        if not isinstance(v, (int, float)) or not lo <= v <= hi:
            why.append(f"{k} {v!r} outside {lo}-{hi}")
    if not isinstance(rep.get("os"), str):
        why.append("no os")
    for name, res in (rep.get("results") or {}).items():
        for test, v in (res or {}).items():
            acc = v.get("acc") if isinstance(v, dict) else None
            if not test.startswith("_") and not (isinstance(acc, (int, float)) and 0 <= acc <= 100):
                why.append(f"{name} {test}: score {acc!r} not 0-100")
    return why


def _key(model: str) -> str:
    """Catalog key for a GGUF file name or eval run name, else the name itself."""
    for k, m in catalog.MODELS.items():
        if model in (k, m["file"]):
            return k
    return model


def entries(reports: list[dict]) -> list[dict]:
    out = []
    for k, m in catalog.MODELS.items():
        out.append({"source": "maintainers", **MAINTAINER, "vendor": "AMD", "os_family": "Windows", "model": k,
                    "tok_s": m["tok_s_9070xt"], "scores": dict(sorted(m["scores"].items())), "flags": []})
    for row in merge_reports.rows(reports):
        flags = []
        if row["tok_s"] and row["tok_s"] > MAX_TOK_S:
            flags.append(f"decode {row['tok_s']} tok/s looks too fast, check the model file")
        if not row["tok_s"] and not row["scores"]:
            continue
        out.append({"source": "community", "gpu": row["gpu"], "vram_gb": row["vram"], "ram_gb": row["ram"],
                    "os": row["os"], "llama_cpp": row["build"], "vendor": vendor(row["gpu"]),
                    "os_family": os_family(row["os"]), "model": _key(row["model"]), "tok_s": row["tok_s"],
                    **({"host": row["host"]} if row.get("host") else {}),
                    "scores": dict(sorted(row["scores"].items())), "flags": flags})
    return out


def platforms(reports: list[dict]) -> dict:
    counts: dict[tuple, int] = {}
    for r in reports:
        cell = (os_family(r.get("os")), vendor(r.get("gpu")))
        if all(cell):
            counts[cell] = counts.get(cell, 0) + 1
    m = {}
    for o in OSES:
        for v in VENDORS:
            if (o, v) in NA or (o == "macOS" and v != "Apple"):
                continue
            n = counts.get((o, v), 0)
            m[f"{o}/{v}"] = {"maintainers": (o, v) == ("Windows", "AMD"), "community_reports": n}
    return m


def label(cell: dict | None) -> str:
    if not cell:
        return "untested"
    bits = (["tested by maintainers"] if cell["maintainers"] else []) + \
        ([f"community reports ({cell['community_reports']})"] if cell["community_reports"] else [])
    return ", ".join(bits) or "untested"


def table(m: dict) -> str:
    lines = ["| | " + " | ".join(VENDORS) + " |", "|---" * (len(VENDORS) + 1) + "|"]
    for o in OSES:
        lines.append(f"| {o} | " + " | ".join(label(m[f"{o}/{v}"]) if f"{o}/{v}" in m else "-"
                                                for v in VENDORS) + " |")
    return "\n".join(lines)


def update_readme(readme: str, tbl: str) -> str:
    if START not in readme or END not in readme:
        raise SystemExit(f"docs/how-it-works.md has no {START} ... {END} section")
    head, rest = readme.split(START, 1)
    return f"{head}{START}\n{tbl}\n{END}{rest.split(END, 1)[1]}"


def load_reports(folder: Path) -> tuple[list[dict], list[str]]:
    reports, skipped = [], []
    files = sorted(folder.glob("*.json")) + sorted(folder.glob("*.md")) if folder.exists() else []
    for f in files:
        if f.name.lower() == "readme.md":
            continue
        try:
            rep = merge_reports.load(f)
        except (OSError, ValueError) as e:
            skipped.append(f"{f.name}: not readable ({e})")
            continue
        why = validate(rep)
        if why:
            skipped.append(f"{f.name}: " + "; ".join(why))
        else:
            reports.append(rep)
    return reports, skipped


def build(reports: list[dict]) -> dict[str, str]:
    """{relative path: new content} for every output."""
    m = platforms(reports)
    board = {"schema": SCHEMA, "reports": len(reports), "entries": entries(reports), "platforms": m}
    readme = (ROOT / "docs" / "how-it-works.md").read_text(encoding="utf-8")
    dump = lambda o: json.dumps(o, ensure_ascii=False, indent=1) + "\n"  # noqa: E731
    return {"data/leaderboard.json": dump(board), "localllm/platforms.json": dump(m),
            "docs/how-it-works.md": update_readme(readme, table(m))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reports", type=Path, default=ROOT / "reports")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    reports, skipped = load_reports(a.reports)
    for s in skipped:
        print(f"skipped {s}")
    stale = []
    for rel, text in build(reports).items():
        path = ROOT / rel
        old = path.read_text(encoding="utf-8") if path.exists() else None
        if old == text:
            continue
        stale.append(rel)
        if not a.check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
    if a.check:
        if stale:
            sys.exit(f"out of date: {', '.join(stale)}. Run python tools/build_leaderboard.py")
        print(f"up to date ({len(reports)} reports)")
        return
    print(f"{len(reports)} reports -> {', '.join(stale) or 'nothing changed'}")


if __name__ == "__main__":
    main()
