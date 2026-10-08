"""Merge `localllm report` files into the per-GPU table in README.md (#39).

  python tools/merge_reports.py REPORT.json [REPORT2.json ... | DIR]  [--readme README.md] [--check]

A report can be the saved JSON or a GitHub issue body that contains it in a ```json block. Rows are keyed by GPU,
model and llama.cpp build; a newer report replaces an older one. The table lives between the markers
<!-- gpu-table:start --> and <!-- gpu-table:end -->. --check exits 1 when the README would change (for CI).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from localllm import catalog  # noqa: E402

START, END = "<!-- gpu-table:start -->", "<!-- gpu-table:end -->"
LANGS_SHOWN = 4


def load(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"```json\s*(\{.*\})\s*```", text, re.S)
    return json.loads(m.group(1) if m else text)


def rows(reports: list[dict]) -> list[dict]:
    """One row per (GPU, model, llama.cpp build): tuned decode tok/s plus that model's eval scores."""
    out: dict[tuple, dict] = {}
    for r in sorted(reports, key=lambda r: r.get("date", "")):
        gpu = r.get("gpu") or f"CPU ({r.get('cpu') or r.get('arch')})"
        for model, builds in (r.get("tune") or {}).items():
            for build, t in builds.items():
                out[(gpu, model, build)] = {"gpu": gpu, "vram": r.get("vram_gb"), "ram": r.get("ram_gb"),
                                            "os": r.get("os"), "build": build, "model": model,
                                            "tok_s": t.get("tok_s"), "settings": _settings(t), "scores": {}}
        for name, res in (r.get("results") or {}).items():
            scores = {k: v.get("acc") for k, v in res.items() if not k.startswith("_") and isinstance(v, dict)}
            hit = [row for (g, m, _b), row in out.items() if g == gpu and _same_model(m, name)]
            for row in hit or [out.setdefault((gpu, name, r.get("llama_cpp")), {
                    "gpu": gpu, "vram": r.get("vram_gb"), "ram": r.get("ram_gb"), "os": r.get("os"),
                    "build": r.get("llama_cpp"), "model": name, "tok_s": None, "settings": "", "scores": {}})]:
                row["scores"].update(scores)
    return sorted(out.values(), key=lambda x: (str(x["gpu"]), str(x["model"])))


def _settings(t: dict) -> str:
    bits = [f"{k}={v}" for k, v in (t.get("env") or {}).items()] + ([f"MTP {t['mtp']}"] if t.get("mtp") else [])
    return ", ".join(bits) or "defaults"


def _same_model(tuned_file: str, eval_name: str) -> bool:
    """An eval run named after a catalog key (`--name qwen3.8-27b-q3`) belongs to that key's GGUF file."""
    if eval_name in catalog.MODELS:
        return catalog.MODELS[eval_name]["file"] == tuned_file
    a, b = (re.sub(r"[^a-z0-9]", "", s.lower().rsplit(".gguf", 1)[0]) for s in (tuned_file, eval_name))
    return bool(a and b) and (a in b or b in a)


def table(rs: list[dict]) -> str:
    head = "| GPU | VRAM | RAM | OS | llama.cpp | model | decode tok/s | kept settings | scores |\n" \
           "|---|---|---|---|---|---|---|---|---|"
    lines = [head]
    for r in rs:
        sc = sorted(r["scores"].items())
        shown = ", ".join(f"{k} {v}" for k, v in sc[:LANGS_SHOWN]) + (f" (+{len(sc) - LANGS_SHOWN})"
                                                                       if len(sc) > LANGS_SHOWN else "")
        lines.append(f"| {r['gpu']} | {r['vram']} GB | {r['ram']} GB | {r['os']} | {r['build'] or '-'} | "
                     f"{r['model']} | {r['tok_s'] or '-'} | {r['settings'] or '-'} | {shown or '-'} |")
    return "\n".join(lines)


def update(readme: str, tbl: str) -> str:
    if START not in readme or END not in readme:
        raise SystemExit(f"README has no {START} ... {END} section")
    head, rest = readme.split(START, 1)
    return f"{head}{START}\n{tbl}\n{END}{rest.split(END, 1)[1]}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("reports", nargs="+", type=Path)
    ap.add_argument("--readme", type=Path, default=Path(__file__).resolve().parents[1] / "README.md")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    files = [f for p in a.reports for f in (sorted(p.glob("*.json")) + sorted(p.glob("*.md")) if p.is_dir() else [p])]
    new = update(a.readme.read_text(encoding="utf-8"), table(rows([load(f) for f in files])))
    if a.check:
        sys.exit(0 if new == a.readme.read_text(encoding="utf-8") else 1)
    a.readme.write_text(new, encoding="utf-8")
    print(f"{a.readme}: {len(files)} reports merged")


if __name__ == "__main__":
    main()
