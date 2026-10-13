# Website

Static pages generated from the repo's data (catalog, `data/leaderboard.json`, `tools/footprint_results.jsonl`),
so the site and the README can't disagree.

```
python site/build.py                                   # -> site/public/
python -m http.server 8765 --directory site/public     # look at it on http://127.0.0.1:8765
python site/make_og.py                                 # redraw the social preview (needs matplotlib)
```

## Deploy (Cloudflare Pages, default address)

The site lives at the free `*.pages.dev` address, no custom domain:

```
npx wrangler login                                     # once
npx wrangler pages project create make-localllm-easier --production-branch main   # once
python site/build.py
npx wrangler pages deploy site/public --project-name make-localllm-easier --branch main
```

If the project name is taken, pick another and build with `SITE_URL=https://<name>.pages.dev python site/build.py`
so canonical links and the sitemap point at the right address.

After the first deploy: add the URL-prefix property in Google Search Console, verify it, submit `/sitemap.xml`.

## Before and after the first deploy (checklist)

- `python -m pytest -q tests/test_site.py` (build + every internal link) and look at it locally.
- Analytics: in the Cloudflare dashboard, Pages project > Metrics > enable **Web Analytics**. Cloudflare injects its
  cookieless beacon at deploy time, so the site code stays free of trackers; the footer's "no tracking" note in
  `assets/site.js` refers to our own code.
- Google Search Console: add the URL-prefix property `https://make-localllm-easier.pages.dev/`, choose "HTML file",
  put the `googleXXXX.html` file it gives you into `site/static/` (copied to the site root on every build), rebuild,
  deploy, verify, then submit `/sitemap.xml`.
- GitHub repo: About > Website = the pages.dev URL; Settings > Social preview = `site/assets/og.png`.
- Pages: 18 (home, features, how it works, use cases, compare, get started, leaderboard, 7 guides, Thai home and Thai
  get started, 404).
