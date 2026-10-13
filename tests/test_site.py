"""The website builds from the repo data, every internal link resolves, and nothing is missing (#74)."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_site_builds_and_internal_links_resolve(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "site" / "build.py")], check=True, capture_output=True)
    out = ROOT / "site" / "public"
    static = {f.name for f in (ROOT / "site" / "static").glob("*")}       # copied as-is (e.g. Search Console's file)
    pages = [p for p in out.rglob("*.html") if p.name not in static]
    assert len(pages) >= 12 and (out / "sitemap.xml").exists() and (out / "robots.txt").exists()
    for page in pages:
        html = page.read_text(encoding="utf-8")
        assert "<title>" in html and 'name="description"' in html, page
        for href in re.findall(r'(?:href|src)="(/[^"#]*)', html):
            target = out / href.lstrip("/")
            assert target.exists() or (target / "index.html").exists(), f"{page}: broken link {href}"
