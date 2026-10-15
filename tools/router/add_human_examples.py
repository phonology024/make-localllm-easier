"""Human-written routing examples (#40): check a contribution and add it to tools/router/human_test.jsonl.

  python tools/router/add_human_examples.py --issue 151          # fetch the issue body with the GitHub CLI, add it
  python tools/router/add_human_examples.py BODY.md              # a saved issue body
  python tools/router/add_human_examples.py BODY.md --check      # only say what would be accepted (CI comments this)

The "Routing examples" issue form gives a language and one example per line, `label | text` (or label, a tab, text);
labels are general / math / code / translate, as defined in tools/router/README.md. Each line is checked:
- the label is one of the four and the text is 3-500 characters;
- no URL, e-mail address or long digit run (phone or ID numbers): these are often private, and never needed;
- not already in human_test.jsonl, shared_test.jsonl or generated.jsonl (case, spacing and punctuation ignored), so
  the human test set stays independent of the AI-written data it is meant to check;
- a script that clearly doesn't match the stated language is a warning, not a refusal (people mix languages).
Accepted lines are written with source "human" and the issue number, so a contributor can ask for theirs removed.
Score both routers on it: `python tools/router/train_router.py MODEL --test tools/router/human_test.jsonl` and
`python tools/router/eval_laya.py tools/router/human_test.jsonl`.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from localllm import router  # noqa: E402

D = Path(__file__).resolve().parent
HUMAN = D / "human_test.jsonl"
OTHERS = (D / "shared_test.jsonl", D / "generated.jsonl")
LABELS = ("general", "math", "code", "translate")
LANG = re.compile(r"[a-z]{2,3}")
PRIVATE = [(re.compile(r"https?://|www\.", re.I), "a link"),
           (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "an e-mail address"),
           (re.compile(r"(?:\d[\s-]?){9,}"), "a long number (phone or ID?)")]
SCRIPT_LANGS = {lang for lang, _ in router.SCRIPTS}


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith(("P", "Z", "C")))


def known(paths=(HUMAN, *OTHERS)) -> set[str]:
    seen = set()
    for p in paths:
        if p.exists():
            for raw in p.read_text(encoding="utf-8").splitlines():
                try:
                    seen.add(norm(json.loads(raw)["text"]))
                except (ValueError, KeyError, TypeError):
                    continue
    return seen


def parse(body: str) -> tuple[str | None, list[str]]:
    """(language code, example lines) from the issue form's sections, or from a plain file of lines."""
    sections = dict(re.findall(r"^### (.+?)\s*\n(.*?)(?=^### |\Z)", body, re.S | re.M))
    lang = (sections.get("Language") or "").strip().split()[0:1]
    text = sections.get("Examples", body)
    text = re.sub(r"^```\w*\s*$", "", text, flags=re.M)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and ln.strip() != "_No response_"]
    return (lang[0].lower() if lang else None), lines


def check(lang: str | None, lines: list[str], seen: set[str]) -> tuple[list[dict], list[str], list[str]]:
    """(accepted rows, refused lines with the reason, warnings)."""
    rows, refused, warnings = [], [], []
    if not lang or not LANG.fullmatch(lang):
        return [], [f"language: give an ISO code like th, hi, pt (got {lang!r})"], []
    batch = set()
    for n, line in enumerate(lines, 1):
        m = re.match(r"^\s*(\w+)\s*(?:\||\t)\s*(.+)$", line)
        if not m:
            refused.append(f"line {n}: write `label | text`")
            continue
        label, text = m.group(1).lower(), m.group(2).strip()
        why = []
        if label not in LABELS:
            why.append(f"label {label!r} is not one of {', '.join(LABELS)}")
        if not 3 <= len(text) <= 500:
            why.append("the text should be 3-500 characters")
        why += [f"has {what}" for rx, what in PRIVATE if rx.search(text)]
        key = norm(text)
        if key in seen or key in batch:
            why.append("already in the test or training data")
        if why:
            refused.append(f"line {n}: " + "; ".join(why))
            continue
        guess = router.detect_language(text)
        if lang in SCRIPT_LANGS and guess != lang or guess in SCRIPT_LANGS and guess != lang:
            warnings.append(f"line {n}: looks like {guess}, not {lang} (kept; mixed languages are fine)")
        batch.add(key)
        rows.append({"text": text, "label": label, "lang": lang, "source": "human"})
    return rows, refused, warnings


def report(rows, refused, warnings) -> str:
    out = [f"{len(rows)} example(s) accepted" + (f", {len(refused)} not:" if refused else ".")]
    out += [f"- {r}" for r in refused]
    if warnings:
        out += ["", "Warnings:"] + [f"- {w}" for w in warnings]
    if rows:
        by = {lab: sum(r["label"] == lab for r in rows) for lab in LABELS}
        out += ["", "By label: " + ", ".join(f"{k} {v}" for k, v in by.items() if v)]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("file", nargs="?", type=Path)
    src.add_argument("--issue", type=int)
    ap.add_argument("--check", action="store_true", help="only report, write nothing")
    a = ap.parse_args()
    body = (subprocess.run(["gh", "issue", "view", str(a.issue), "--json", "body", "--jq", ".body"], check=True,
                           capture_output=True, text=True, encoding="utf-8").stdout if a.issue
            else a.file.read_text(encoding="utf-8"))
    lang, lines = parse(body)
    rows, refused, warnings = check(lang, lines, known())
    print(report(rows, refused, warnings))
    if a.check or not rows:
        sys.exit(0 if rows and not refused else 1)
    with open(HUMAN, "a", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps({**r, **({"issue": a.issue} if a.issue else {})}, ensure_ascii=False) + "\n")
    print(f"added {len(rows)} to {HUMAN.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
