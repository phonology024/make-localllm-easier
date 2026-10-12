# Shared `localllm report` files

Each file here is one report from one PC. `tools/build_leaderboard.py` turns them into `data/leaderboard.json` (the
website's leaderboard) and the platform table in the main README.

## Add yours

1. `localllm tune` and `localllm eval --langs en,<your language>` on your GPU
2. `localllm report` prints the JSON, saves it under `~/.localllm/` and opens a prefilled issue
3. Either submit that issue, or open a PR that adds the JSON here as `reports/<gpu>-<date>.json`
   (for example `rtx-4070-2026-10-12.json`), then run `python tools/build_leaderboard.py` and commit what it changes

The report has no user names, paths or keys (see `localllm/report.py`). Read it before you send it anyway.

## What gets checked

- `schema` must be one we know (now: 1)
- `vram_gb`, `ram_gb` in a sane range, `os` present
- every score between 0 and 100
- decode above 400 tok/s gets a flag, not a rejection

Reports that fail are skipped with the reason printed. For the same GPU, model and llama.cpp build the newest report
wins.
