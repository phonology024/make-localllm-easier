# Shared `localllm report` files

Each file here is one report from one PC. `tools/build_leaderboard.py` turns them into `data/leaderboard.json` (the
website's leaderboard) and the platform table in the main README.

## Add yours

1. `localllm tune` and `localllm eval --langs en,<your language>` on your GPU
2. `localllm report` prints the JSON, saves it under `~/.localllm/` and opens the prefilled
   [Share a result](https://github.com/phonology024/make-localllm-easier/issues/new?template=results.yml) form
3. Submit it. A check comments on the issue within a minute: accepted, or what to fix (edit the issue and it re-checks)
4. A maintainer adds it with `python tools/add_report.py --issue <number>` and `python tools/build_leaderboard.py`.
   You can also open a PR that adds the JSON here as `reports/<gpu>-<date>.json` (for example
   `rtx-4070-2026-10-12.json`) plus what `build_leaderboard.py` changes; CI runs the same checks

The report has no user names, paths or keys (see `localllm/report.py`). Read it before you send it anyway.

## What gets checked

- `schema` must be one we know (now: 1)
- `vram_gb`, `ram_gb` in a sane range, `os` present
- every score between 0 and 100
- decode above 400 tok/s gets a flag, not a rejection
- privacy: no secret-looking field (key, token, password ...), no absolute path, no home folder, no e-mail address.
  `localllm report` already removes these; the check catches a report edited by hand or made by an old version, and
  refuses it rather than fixing it silently

Reports that fail are skipped with the reason printed. For the same GPU, model and llama.cpp build the newest report
wins.
