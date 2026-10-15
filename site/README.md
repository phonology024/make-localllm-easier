# Website

Static pages generated from the repo's data (catalog, `data/leaderboard.json`, `tools/footprint_results.jsonl`),
so the site and the README can't disagree.

```
python site/build.py                                   # -> site/public/
python -m http.server 8765 --directory site/public     # look at it on http://127.0.0.1:8765
python site/make_og.py                                 # redraw the social preview (needs matplotlib)
```

## Deploy (GitHub Pages)

`.github/workflows/pages.yml` builds the site and publishes it to <https://phonology024.github.io/make-localllm-easier/>
on every push to `main` that touches the site or its data (catalog, `data/leaderboard.json`, footprint results), and on
demand from the Actions tab. Links are written root-relative and get the `/make-localllm-easier` base path at build
time, so the same build also works at a root domain (`SITE_URL=https://example.org python site/build.py`).

Once, by a repo admin: Settings > Pages > Build and deployment > Source = **GitHub Actions** (or
`gh api -X POST repos/phonology024/make-localllm-easier/pages -f build_type=workflow`).

Pages: one per language (`/leaderboard/<lang>/`) and one per GPU (`/leaderboard/gpu/<gpu>/`) are generated from
`data/leaderboard.json`, so every new measurement or merged `localllm report` adds or updates them on the next push.

## Before and after the first deploy (checklist)

- `python -m pytest -q tests/test_site.py` (build + every internal link, at a root domain and under a base path) and
  look at it locally.
- Google Search Console: add the URL-prefix property `https://phonology024.github.io/make-localllm-easier/`, choose
  "HTML file"; the file in `site/static/` is copied to the site root on every build. Verify, then submit `sitemap.xml`.
- GitHub repo: About > Website = the github.io URL; Settings > Social preview = `site/assets/og.png`.
- No analytics: GitHub Pages has none built in, and the site ships no trackers.
