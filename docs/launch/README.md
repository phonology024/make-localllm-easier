# Launch kit (drafts, not posted)

Drafts for the maintainer to review, edit and post by hand (#147). Nothing here is posted by any tool or workflow.

| File | For |
|---|---|
| [show-hn.md](show-hn.md) | Hacker News "Show HN" |
| [reddit-localllama.md](reddit-localllama.md) | r/LocalLLaMA |
| [x-thread.md](x-thread.md) | a short X / Twitter thread |

## Fact sheet: every number the drafts use, and where it comes from

Re-check these against `main` on the day you post; if one changed, change the drafts.

| Claim | Number | Source |
|---|---|---|
| System RAM, same 30-turn chat, Qwen3.8-27B Q3 | 9.30 GB with llama.cpp defaults -> **2.37 GB** with localllm, 35.2 tok/s both | README "Less RAM, same speed" (checked against `tools/footprint_results.jsonl` by `tests/test_readme.py`) |
| Same for gemma-4-26B-A4B QAT | 9.24 GB -> **2.21 GB**, ~85 tok/s both | same |
| Test PC | one PC: Windows 11, AMD RX 9070 XT 16 GB, 32 GB RAM, llama.cpp b11476 (Vulkan) | docs/benchmarks.md, docs/how-it-works.md "Where it is tested" |
| Languages measured | 23 on Global-MMLU-Lite, 44 countries' own exams on INCLUDE, plus ThaiExam | docs/faq.md "Which languages are measured?" |
| 2-bit penalty | 8-13 points; Hindi, Arabic and Thai lose the most (13) | docs/benchmarks.md, finding 1 |
| Small model that fits beats a big one squeezed | Qwen3.5-9B at 4 bit beats Qwen3.8-27B at 2 bit in every language measured (Thai exams 61.1 vs 54.2) | docs/benchmarks.md "Smaller cards" |
| 70B at 2 bit vs a 26B MoE that fits | Thai 50 vs 78, Hindi 58-63 vs 69-70 | docs/benchmarks.md, finding 7 (#86) |
| Pick per language example | on a 16 GB card: gemma-4-26B-A4B QAT for Thai (exams 65.7, math 89.6) at ~85 tok/s; Qwen3.8-27B Q3 is +5.5 pts on Chinese knowledge | docs/benchmarks.md table, docs/how-it-works.md routing table |
| AMD without Resizable BAR | one env var makes Qwen3.8-27B 1.64x faster; gemma 7-9% slower, so it is set per model | docs/benchmarks.md, finding 3 |
| No-Python downloads | 0.7.1: Windows x64 .exe, macOS arm64 and x64, Linux x64, AppImage, SHA256SUMS; not code-signed yet | GitHub release v0.7.1, docs/faq.md "Do I need Python?" |
| Leaderboard | https://phonology024.github.io/make-localllm-easier/leaderboard/ (per language, per GPU) | site/build.py, pages.yml |
| Contributed GPUs so far | none yet: every row is from the maintainers' PC | data/leaderboard.json ("reports": 0) - say so, and ask |
| License | MIT; models keep their own licenses | README |

## Before posting

- Post when you can answer comments for a few hours.
- Hacker News: link the repo, not the website; plain title, no superlatives; first comment = the "why" paragraph.
- r/LocalLLaMA: lead with the numbers and the method; people will ask about other GPUs, so ask them for
  `localllm report`.
- Don't claim speed or quality on hardware we haven't measured. NVIDIA, Intel and Apple are "should work, unmeasured".
