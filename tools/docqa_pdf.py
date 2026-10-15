"""How much does a PDF cost retrieval? (#113) The docqa documents, printed to PDF by Edge or Chrome (as people's PDFs are:
real subset fonts, ToUnicode maps), read back through pypdf, and searched with the same questions as the TXT run.

  python tools/docqa_pdf.py --langs en,th,ja,hi,ar,zh,es [--browser PATH]

A passage counts as found when it holds the needle's name and its five-digit number: PDF text can lose marks (Thai tone
marks, Hindi conjuncts), so the TXT run's exact-sentence match would call a found passage a miss. The TXT side is
scored the same way here, so the two columns compare like for like. CPU only (multilingual-e5-small); needs pypdf
(installed, or the pinned wheel localllm downloads) and a Chromium-based browser for printing.
"""
from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from localllm import bench, docs  # noqa: E402
from localllm.runtime import HOME  # noqa: E402

BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe", "google-chrome", "chromium", "msedge"]


def browser(given: str | None) -> str:
    for b in ([given] if given else BROWSERS):
        if b and (Path(b).is_file() or shutil.which(b)):
            return b
    sys.exit("no Chromium-based browser found: pass --browser PATH")


def to_pdf(text: str, lang: str, exe: str, work: Path) -> bytes:
    page = work / "doc.html"
    rtl = ' dir="rtl"' if lang in ("ar", "he", "fa", "ur") else ""
    body = "".join(f"<p>{html.escape(p)}</p>" for p in text.split("\n\n"))
    page.write_text(f'<!doctype html><meta charset="utf-8"><html lang="{lang}"{rtl}><body>{body}</body></html>',
                    encoding="utf-8")
    out = work / "doc.pdf"
    out.unlink(missing_ok=True)
    subprocess.run([exe, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={out}",
                    page.as_uri()], capture_output=True, timeout=120)
    if not out.is_file():
        sys.exit(f"{exe} did not print a PDF")
    return out.read_bytes()


def found(hits: list[dict], name: str, num: str) -> int | None:
    return next((h["n"] for h in hits if num in h["text"] and name in h["text"]), None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--langs", default="en,th,ja,hi,ar,zh,es")
    ap.add_argument("--browser")
    a = ap.parse_args()
    exe, lib = browser(a.browser), docs.Library(root=HOME / "bench" / "docqa-index")
    print(f"{'lang':5}{'TXT R@1':>9}{'R@5':>7}{'PDF R@1':>10}{'R@5':>7}   n")
    with tempfile.TemporaryDirectory() as tmp:
        for lang in a.langs.split(","):
            items = bench.load("docqa", lang)
            score = {"txt": [], "pdf": []}
            ids = {}
            for it in items:
                name = next(n for n in bench.NEEDLE_NAMES if n in it["needle"])
                num = re.search(r"\d{5}", it["needle"]).group(0)
                for kind in ("txt", "pdf"):
                    key = (kind, it["doc"])
                    if key not in ids:
                        data = it["text"].encode("utf-8") if kind == "txt" else to_pdf(it["text"], lang, exe, Path(tmp))
                        ids[key] = lib.add(f"docqa-{lang}-{it['doc']}.{kind}", data, allow_download=True)["id"]
                    score[kind].append(found(lib.search(it["q"], [ids[key]], k=5), name, num))
            n = len(items)
            r = {k: (100 * sum(x == 1 for x in v) / n, 100 * sum(x is not None for x in v) / n) for k, v in score.items()}
            print(f"{lang:5}{r['txt'][0]:8.1f}%{r['txt'][1]:6.1f}%{r['pdf'][0]:9.1f}%{r['pdf'][1]:6.1f}%  {n}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
