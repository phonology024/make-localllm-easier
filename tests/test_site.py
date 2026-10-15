"""The website builds from the repo data, every internal link resolves, and nothing is missing (#74)."""
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "public"


def _build(site_url: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "site" / "build.py")], check=True, capture_output=True,
                   env={**os.environ, "SITE_URL": site_url})


# a root domain, and GitHub Pages, which serves a project site under /<repo>/
@pytest.mark.parametrize("site_url,base", [("https://example.org", ""),
                                           ("https://phonology024.github.io/make-localllm-easier", "/make-localllm-easier")])
def test_site_builds_and_internal_links_resolve(site_url, base):
    _build(site_url)
    static = {f.name for f in (ROOT / "site" / "static").glob("*")}       # copied as-is (e.g. Search Console's file)
    pages = [p for p in OUT.rglob("*.html") if p.name not in static]
    assert len(pages) >= 12 and (OUT / "sitemap.xml").exists() and (OUT / "robots.txt").exists()
    for page in pages:
        html = page.read_text(encoding="utf-8")
        assert "<title>" in html and 'name="description"' in html, page
        for href in re.findall(r'(?:href|src)="(/[^"#]*)', html):
            assert href.startswith(base + "/"), f"{page}: link {href} misses the base path {base}"
            target = OUT / href[len(base):].lstrip("/")
            assert target.exists() or (target / "index.html").exists(), f"{page}: broken link {href}"
    assert f"<loc>{site_url}/leaderboard/</loc>" in (OUT / "sitemap.xml").read_text(encoding="utf-8")


def test_one_static_page_per_language_and_gpu():
    _build("https://example.org")
    data = json.loads((ROOT / "data" / "leaderboard.json").read_text(encoding="utf-8"))
    langs = {k.split("/")[0] for e in data["entries"] for k in e["scores"]}
    for lang in langs:
        html = (OUT / "leaderboard" / lang / "index.html").read_text(encoding="utf-8")
        main = html.split("<main", 1)[1].split("</main>", 1)[0]
        e = next(e for e in data["entries"] if any(k.startswith(lang + "/") for k in e["scores"]))
        assert f"<code>{e['model']}</code>" in main and "<script" not in main      # readable without JavaScript
        assert any(f"{v:.1f}" in main for k, v in e["scores"].items() if k.startswith(lang + "/"))
    for gpu in {e["gpu"] for e in data["entries"]}:
        slug = re.sub(r"[^a-z0-9]+", "-", gpu.lower()).strip("-")
        assert gpu in (OUT / "leaderboard" / "gpu" / slug / "index.html").read_text(encoding="utf-8")
    board = (OUT / "leaderboard" / "index.html").read_text(encoding="utf-8")
    assert all(f'href="/leaderboard/{lang}/"' in board for lang in langs)


def test_report_fields_are_escaped_on_static_pages(monkeypatch):
    """GPU and model names come from community reports: they must never reach the HTML unescaped."""
    spec = importlib.util.spec_from_file_location("site_build", ROOT / "site" / "build.py")   # not pypa's `build`
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    monkeypatch.setattr(build, "pages", {})
    evil = "<img src=x onerror=alert(1)>"
    data = {"entries": [{"source": "community", "gpu": evil, "model": evil, "tok_s": 1, "os": evil, "llama_cpp": "b1",
                         "scores": {"th/global": 50.0}, "flags": []}]}
    build.language_page(data, "th")
    build.gpu_page(data, evil)
    assert "/leaderboard/gpu/img-src-x-onerror-alert-1/" in build.pages
    for html in build.pages.values():
        assert evil not in html and "&lt;img src=x onerror=alert(1)&gt;" in html
    # language codes name output folders and links: anything but a plain code is dropped, never written
    assert build.score_langs({"scores": {"TH/global": 1.0, "pt-br/global": 1.0}}) == {"pt-br"}
    for bad in ("../../x", 'x"onclick=alert(1)', "th/../.."):
        data["entries"][0]["scores"] = {f"{bad}/global": 50.0, "th/global": 50.0}
        assert build.score_langs(data["entries"][0]) == {"th"}
        with pytest.raises(ValueError):
            build.language_page(data, bad)
        build.gpu_page(data, evil)
        assert bad not in build.pages[f"/leaderboard/gpu/{build.slug(evil)}/"]


def test_compare_page_names_its_sources():
    _build("https://example.org")
    page = (OUT / "compare" / "index.html").read_text(encoding="utf-8")
    assert "<th>Jan</th>" in page and "<h2>Sources</h2>" in page
    assert page.count('rel="nofollow"') >= 8 and "docs.ollama.com/gpu" in page and "lmstudio.ai/docs" in page
