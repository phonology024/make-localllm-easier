# Contributing

Thanks for looking. The most useful thing you can send is a measurement from hardware we don't have. Code and docs
are welcome too.

## 1. Share a report (10-30 minutes, no coding)

We measure on one PC (Windows 11, RX 9070 XT). Every other GPU, OS and language comes from people like you.

```
pip install make-localllm-easier
localllm tune                         # finds the fastest settings for your PC, a few minutes
localllm eval --langs en,<yours>      # e.g. en,de or en,th; a few minutes per language
localllm report                       # opens a prefilled GitHub issue; read it, then submit
```

The report has no user names, paths or keys. It shows up in the README's "Where it is tested" table and on the
website's leaderboard. Interesting results we haven't seen before: NVIDIA on Windows or Linux, Apple Silicon, Intel
Arc, 6-12 GB cards, CPU-only PCs.

Renting a cloud GPU for an hour? `tools/cloud_lab.sh` does all of the above unattended and marks the report as
`cloud:<provider>`.

Have a local exam set for your language that can be used commercially (Apache-2.0, MIT, CC-BY)? Open an issue with
the link; that's how ThaiExam got in.

## 2. Code or docs

```
git clone https://github.com/phonology024/make-localllm-easier
cd make-localllm-easier
pip install -e . pytest
python -m pytest -q tests
```

The tests don't need a GPU or a model download; they run on Windows, Linux and macOS in CI.

- Look for [`good first issue`](https://github.com/phonology024/make-localllm-easier/labels/good%20first%20issue) or
  [`help wanted`](https://github.com/phonology024/make-localllm-easier/labels/help%20wanted). Comment on the issue
  before you start so two people don't do the same work.
- `localllm/` uses the standard library only. Optional tools under `tools/` may use numpy, gguf or matplotlib.
- Keep it short: small functions, comments that say *why*, no new dependencies without asking.
- A change to a number in the README needs the measurement behind it (GPU, RAM, OS, llama.cpp build, command).
  `tests/test_readme.py` checks the `doctor` sample and the RAM table against the code and `tools/*.jsonl`.
- Anything in [docs/stability.md](docs/stability.md) can't be removed without the deprecation steps there.

## 3. Translations

The README exists in English and [Tamil](docs/README.ta.md). Add `docs/README.<lang>.md` and a link at the top of
`README.md`. Translate the text, not the commands or numbers.

## How decisions are made

One maintainer for now. Measurements decide: a feature that measures slower or worse doesn't ship, even if it's
clever, and a negative result is published like a positive one. Issues get an answer within a few days; ping the
issue if one doesn't.
