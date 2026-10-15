# Reach (ROADMAP 0.8)

One row a month, taken with `python tools/metrics.py --write`. "Issues from outside the project" counts issues opened
by anyone who is not a maintainer; `localllm report` issues are the ones labelled `results`.

| date | stars | forks | PyPI downloads (30 days) | `localllm report` issues | issues from outside the project |
|---|---|---|---|---|---|
| 2026-10-13 | 3 | 2 | 690 | 0 | 0 |

## Weekly snapshot

Every Monday the `metrics` workflow runs `tools/traction.py` and adds one JSON line to `data/metrics.jsonl` on the
`metrics` branch (main is protected, so the data has its own branch): stars, forks, watchers, GitHub clones and views
(14-day count and uniques), PyPI downloads (7 and 30 days, no mirrors), release asset downloads per tag,
`localllm report` issues, issues from outside the project and reports on the leaderboard. Read it with
`git fetch origin metrics && git show origin/metrics:data/metrics.jsonl`; run it by hand from the Actions tab
(*metrics* > *Run workflow*) or locally with `python tools/traction.py --print`.

## Targets

Set after the first full month (next snapshot 2026-11-13), so they come from a real baseline. Working numbers for the
first 30 days of promotion, to be replaced then: 100+ stars, 800+ PyPI downloads a month, 10+ `localllm report`
issues from hardware we don't own, 3+ outside contributors.

PyPI counts include CI and mirror bots, so read them as an upper bound. Stars and reports are the numbers that matter.
