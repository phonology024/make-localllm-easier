# Big models on a small card: what the research says (Oct 2026)

Goal: get the quality of a 70B-300B model on a 16 GB GPU (our test PC: RX 9070 XT 16 GB + 32 GB RAM), or make small
models cooperate until they match it. This page is a literature scan, not our measurements. Everything marked
*(arithmetic)* is our own back-of-envelope calculation; everything else comes from the linked sources, many of which are
preprints, READMEs or blogs. Each idea has an issue that turns it into a measurement on our hardware.

## The hard limit

Weights cost bytes: parameters x bits / 8. The GPU is only fast while the weights it reads fit in its memory.

| model | bits | size *(arithmetic)* | fits 16 GB VRAM | fits 16 GB + 32 GB RAM |
|---|---|---|---|---|
| dense 70B | 1.58 | ~14 GB | barely, no room for context | yes |
| dense 70B | 2 | ~17.5 GB | no | yes, slow |
| dense 70B | ~4.8 (Q4_K_M) | ~42 GB | no | barely, ~2-3 tok/s class (RAM bandwidth) |
| MoE ~120B (gpt-oss class) | ~4 (MXFP4) | ~60 GB | no | no (needs ~64 GB RAM) |
| MoE ~235B | ~4.5 | ~130 GB | no | no |

So "70B-300B on 16 GB" cannot be done by quantization alone without paying in quality. The realistic levers, best first:

## 1. Mixture-of-experts + offload: keep quality, pay in speed

A 120B MoE only reads ~5B parameters per token. Attention and the KV cache stay on the GPU, the expert weights live in
system RAM, and `--n-cpu-moe` chooses how many layers' experts stay on the CPU side.
- llama.cpp's gpt-oss guide puts the VRAM floor for the 120B model at about 8 GB; system RAM is the real constraint
  (people pair a 16 GB card with 64 GB RAM) ([guide #15396](https://github.com/ggml-org/llama.cpp/discussions/15396),
  [tuning post](https://carteakey.dev/blog/local-inference/optimizing-gpt-oss-120b-local-inference/)).
- Reported gpt-oss-120b speeds on other cards: RTX 4070 + 64 GB DDR5 about 25 tok/s after tuning, RTX 3090 about 29 tok/s
  in llama.cpp but 8.5 in Ollama (the runtime matters), RTX 4080 12.5 tok/s in Ollama. We found no 16 GB measurement;
  expect roughly 10-25 tok/s *(estimate, unmeasured)*.
- A newer llama.cpp expert cache (`--moe-cache-mib`) keeps hot experts on the GPU: large gains on an RTX 5090, about
  1.29x on one 16 GB report ([comparison](https://www.developersdigest.tech/blog/llama-cpp-moe-offload-n-cpu-moe-vs-moe-cache-2026)).
- Cost on our PC: 32 GB RAM does not hold a 60 GB model, so this alone does not reach 120B for us.

## 2. Expert pruning (REAP) + quantization: make the MoE fit RAM + VRAM

REAP (Cerebras, ICLR 2026) removes whole experts using router-weighted activation and, in the paper, beats merging
across 20B-1T models; at 50% pruning plus 4-bit it reports 87.5% total size reduction on Kimi-K2
([paper](https://arxiv.org/html/2510.13999v3), [code](https://github.com/CerebrasResearch/reap)).
- *(arithmetic)* a 120B MoE at 50% pruned, 4-bit is ~30 GB, which fits 16 GB VRAM + 32 GB RAM. A 235B at 50% / 3-bit is
  ~50 GB, borderline. 300B+ does not fit without SSD streaming.
- Risk to measure: pruning keeps what the calibration set uses. A 2026 paper prunes experts *on purpose* to build
  translation specialists ([arXiv 2605.28042](https://arxiv.org/pdf/2605.28042)), which is the same effect that could
  silently remove a language we care about (Thai, Hindi, Arabic). Per-language scores before and after are the test.
- REAP only applies to MoE models; a dense 70B has no experts to prune.

## 3. Stream experts from SSD with prediction

Three-tier storage (VRAM / pinned RAM / NVMe) with a predictor that prefetches the experts the next layers will use:
- llama.cpp [PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294) streams routed experts from disk; its author
  targets unified-memory machines, and a reviewer notes heavy streaming needs a lot of VRAM to compensate.
- [Discussion #27149](https://github.com/ggml-org/llama.cpp/discussions/27149): the router runs before expert compute,
  giving a prefetch window; double buffering gave 2.09x on a 30B-A3B.
- Research: [VisMMOE](https://arxiv.org/pdf/2605.05899), [DuoServe-MoE](https://arxiv.org/html/2509.07379v2),
  [ProMoE](https://arxiv.org/html/2410.22134v3). A 500-line extension, [llama-moe-cache](https://github.com/ongunm/llama-moe-cache),
  claims a 120 GB model on a 12 GB GPU; that is self-reported and unverified.
- This is the only route to 300B-class on our PC, and the least mature. NVMe is far slower than RAM, so expect a large
  speed penalty.

## 4. Dense 70B at 1.5-2 bit: probably not worth it

- 4-bit costs little (Llama-3.1-70B 95.3% -> 92.1% on one clinical task, [study](https://arxiv.org/pdf/2603.26434)).
- We found no independent benchmark of a 70B squeezed to 1.58-2 bit after training. BitNet b1.58 matches full precision
  only when *trained* at 1.58 bit ([paper](https://arxiv.org/pdf/2402.17764)), which is not the same thing.
- Our own measurements agree: 2-bit costs 8-13 points and Thai/Hindi/Arabic lose most (README).
- ParetoQ: below 3 bits the representation changes a lot, so calibration cannot fix it ([paper](https://arxiv.org/pdf/2502.02631)).

## 5. Small models + more compute at test time

Instead of a bigger model, sample more and pick well (best-of-N, majority vote, a verifier, tools).
- Snell et al. 2024: with compute-optimal allocation a small model can beat a 14x larger one **on problems it already
  solves some of the time** ([paper](https://arxiv.org/abs/2408.03314)); "Can 1B beat 405B?" pushes the same line
  ([arXiv 2502.06703](https://arxiv.org/pdf/2502.06703)). A 2026 vision-language study shows 2B going from 35% to 64% with
  self-consistency on one benchmark ([arXiv 2606.28864](https://arxiv.org/pdf/2606.28864)).
- Counter-evidence (2026): on MMLU-Pro an 8B with debate, mixture-of-agents, self-consistency or self-refinement stayed
  below 53%, under a 70B with plain chain of thought (64.3%), at the same compute ([arXiv 2605.01566](https://arxiv.org/pdf/2605.01566)).
- Reading: it works when answers are **checkable** (math with a known answer, code with tests) and the small model is
  not hopeless; it does not make an 8B into a 70B on open knowledge questions. We already have a code sandbox and a
  math suite (MGSM), so we can test exactly the favourable case.

## 6. A harness around the model

Scaffolding (plan, call tools, run tests, retry from the error) changes results, but capability still dominates: one
2026 study holds the harness fixed because model capability drives success more than scaffold choice
([arXiv 2607.07946](https://arxiv.org/pdf/2607.07946)); another finds harness choice still moves scores
([arXiv 2609.26777](https://arxiv.org/pdf/2609.26777)). We found no solid numbers for 7-35B local models with a
verification loop, so that is something we would measure ourselves.

## What we would try first (order)

1. Measure gpt-oss-120b-class MoE with `--n-cpu-moe` and the expert cache on our PC: the speed and RAM floor, honestly
   (#82).
2. REAP-pruned 100-120B MoE at 3-4 bit that fits 48 GB, scored per language (#83).
3. Small model + verifier on checkable tasks vs the big model, per language and per second of wall time (#85).
4. SSD expert streaming for 235B-300B, after #82/#83 (#84).
5. KV cache compression (#49) stacks with all of the above: long context is where the KV cache, not the weights, runs
   out of memory.
