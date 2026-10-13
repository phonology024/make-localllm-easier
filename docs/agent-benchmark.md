# Local-agent benchmark (#96): the rules, fixed before anything runs

We want to know whether a harness built with the runtime beats a general one (Hermes Agent, OpenCode, Aider) on the
same local model. We are a competitor and the referee, so the task set, the scoring and what counts as a win are
written down and committed here **before** any harness is run. Changing them after the first result means a new
version of the task set and a rerun of every harness.

## Task set (version 1, frozen 2026-10-14)

All in [`tools/agent_bench/tasks.json`](../tools/agent_bench/tasks.json):

| Part | Tasks | Judged by |
|---|---|---|
| Repo tasks | 6 changes to a small project ([`fixture/`](../tools/agent_bench/fixture)): fix a bug, add a method with a test, rename across files, a CLI flag, CSV import, a new module used by the CLI | a hidden check per task, run in the Docker sandbox on the files the agent left |
| Thai tasks | 4: three repo tasks asked in Thai, one whose result must contain a Thai message | the same hidden checks |
| Code tasks | 30 EvalPlus problems (14 HumanEval, 16 MBPP), listed by id | EvalPlus's extended tests, as in `eval --suites codegen` |
| Long session | the 30 questions of `tools/agent_session.py` on this repository, 32k context | answer names a file that holds the answer; no failure or spill |

Every check fails on the untouched fixture and passes on a reference solution (checked when the set was frozen).

## Metrics

Task success, tool-call error rate, wall time, generated tokens, peak RAM and VRAM (llama-server and the harness),
spill events. Three runs per harness and model; we report the mean, the min-max range and every failure.

## Setup

- Same llama-server for every harness: localllm's launch flags, `--profile max`, gemma-4-26B-A4B QAT and Qwen3.8-27B
  Q3, llama.cpp b11476.
- Each harness runs with its own recommended local-model settings, and the configs are published with the results.
  The other projects are invited to tune theirs.
- Harnesses run their tools in a container (the agent's commands never touch the host).

## What "beats" means

On a model, harness A beats harness B when A's success rate over repo + Thai + code tasks is higher by more than the
larger of the two run-to-run ranges, **and** A's peak RAM is lower. If ours loses, that is published the same way, and
the harness is fixed or dropped (the ship rule in #88).

## Status

- Task set and rules: this page (frozen).
- First measurement of the harness idea: `tools/harness_bench.py` (verify-and-retry on HumanEval+/MBPP+, #88).
- Not run yet: everything that executes model-written code needs Docker with Linux containers, which needs hardware
  virtualization turned on in the PC's firmware. The test PC has it off, so these runs wait for that.
