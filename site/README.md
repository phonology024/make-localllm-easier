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
