# Roadmap

**North star: the leanest local AI.** Use the least CPU, system RAM and GPU memory, and the smallest model files that
keep measured quality - beating other launchers on footprint, not just convenience. Where no published work answers a
question, we run the experiment ourselves and publish the numbers.

Every release ships a before/after number measured on real hardware (one headline figure + a bar chart, e.g.
"3x less RAM on the same chat"), so an upgrade is visible at a glance.

Guiding principle (borrowed from how the Elysia web framework cut memory): **don't load, don't copy, don't run what
isn't used** - skip unused model parts, share instead of duplicating caches, and specialise the launch per machine once.

## 0.1 - one command (released)
- `localllm`: detect GPU/RAM, pick the most accurate measured model for your language, download, start a tuned
  llama-server, open the chat page
- `localllm doctor`: which model sizes this GPU suits, speed estimates, how much text the chosen model holds
- `localllm eval`: Global-MMLU-Lite (23 languages) + INCLUDE (44 countries) + ThaiExam, logprob scoring
- Tuned launch: small-BAR fix, MTP drafting for Qwen3.8, single-slot unified KV, `-fit off`

## 0.2 - use less system RAM (released)
Result: 4x less RAM over a 30-turn chat (Qwen3.8 9.30 -> 2.37 GB, gemma-4 9.24 -> 2.21 GB), same speed, ~15% less CPU.
Evidence: llama-server defaults (`--cache-ram 8192` MiB prompt cache, `--ctx-checkpoints 32` per slot) took one Gemma 4
user from 0.7 GB to 18 GB of RAM and out-of-memory in three generations; with 0-1 checkpoints it stayed at 0.4-1.5 GB
(llama.cpp #21690, PR #16391).

- [x] `localllm chat`: terminal chat with streaming, history, /save, /think, Ctrl+C to stop (tested on gemma-4, 85-88 tok/s)
- [x] Measure RAM over a long chat, defaults vs tuned, Qwen3.8 + gemma-4 (#1)
- [x] Low-RAM profile by default for one user: `-np 1`, `--ctx-checkpoints` 0-4 (0-1 for hybrid/Gemma 4),
      `--cache-ram` 0-1024 sized from installed RAM; check the speed cost (#2)
- [x] KV cache `q8_0` by default (half the KV memory, ~0.05% perplexity); `q4_0` K only as an opt-in after measuring
      per language
- [x] Load mode: read weights straight into place (measured: mmap kept the whole file resident, 13.5 vs 6.1 GB even with experts in RAM)
- [x] Two-tier MoE estimate (VRAM + RAM) that warns when RAM is short - llama.cpp `--fit` assumes RAM is unlimited;
      choose `--n-cpu-moe` from free RAM (#4)
- [x] `doctor`: RAM the model will use and what stays free; max context that fits on the GPU (#3)
- [x] Don't load unused parts: skip the vision projector without images, skip the MTP head when not drafting
- [x] Measure and minimise CPU use: idle and busy llama-server CPU, thread count, busy-waiting (#33)
- [x] Detect Windows "shared GPU memory" spill (VRAM silently overflowing into RAM) and say so

## 0.3 - work with the cloud providers' APIs (released)
Result: one endpoint verified with the official OpenAI, Anthropic and Ollama SDKs + Gemini REST (streaming included), routing off by default.
Evidence: llama-server already serves Anthropic `/v1/messages` (tools, vision, thinking) next to OpenAI
`/v1/chat/completions`; Ollama >= 0.14 does too; LiteLLM routes/falls back across providers; RouteLLM's router keeps 95%
of GPT-4 quality while sending only 26% of requests to it.

- [x] One local endpoint: pass OpenAI and Anthropic APIs straight through to llama-server; thin shims for Ollama
      `/api/*` and Gemini `generateContent` (#5)
- [x] Hybrid routing, local first: forward to the user's own cloud key when the prompt is too long, needs a tool/model
      the PC can't run, or the local model is busy; cost and privacy shown before sending; keys in the OS keychain (#6)
- [x] Quality-aware routing: thresholds calibrated from *our measured per-language scores* (e.g. send hard Thai or math
      to the cloud when the local quant is below the quality floor) - nobody routes by language today
- [x] `localllm route --test`: same prompt local vs cloud - answer, latency, cost (#7)

## 0.4 - squeeze the GPU (released)
Result: `localllm tune` took Qwen3.8-27B from 21.7 to 55 tok/s on the same RX 9070 XT (2.5x); gemma-4 runs 124-128 tok/s with the per-model default.
Evidence: ReBAR-off fix 1.7x on RX 9070 XT (ours) and 2.7x on RX 7900 XTX (#27097); MTP +40% on RDNA4 (ours), 1.86x on
RTX 3090, but slower on Apple Metal; CUDA fusion + `GGML_CUDA_GRAPH_OPT=1` +17-42% on RTX 4090/5090; Vulkan vs ROCm
winner on RDNA4 differs between decode and prefill.

- [x] `localllm tune`: measure candidate settings with real llama-server runs, keep only >= 1.1x wins, cache per GPU +
      model + llama.cpp build (#8). Multi-backend sweep waits for a machine with more than one backend
- [x] Detect a small host-visible heap (Resizable BAR off) and set the Vulkan fix automatically, confirmed by A/B
- [ ] `GGML_CUDA_GRAPH_OPT=1` on single-GPU NVIDIA - implemented in `tune`, needs an NVIDIA owner to measure (#19)
- [x] MTP only where it measures faster: A/B draft length 2/3/5 per GPU, keep it on above 1.1x
- [ ] Pick the backend by workload: prefill-heavy (documents/RAG) vs decode-heavy (chat)
- [x] DeltaNet recurrent-state copy overhead: it was the host-visible memory, fixed by the small-BAR fix (state ops 8.2 -> 0.9 ms per token) (#9)
- [ ] Faster load: 16 MiB upload staging buffers (1.2 s faster on a 12 GB model) - patch ready on a fork, to be proposed upstream by the maintainer; `-fit off` already skips the dry-run (#10)
- [x] Before/after speed table per GPU in the release notes (#11)

## 0.5 - compression research (0.5.0 released; research items stay open)
Evidence: across 55 languages, 2-bit hurts non-Latin and low-resource languages most (Bengali -16 COMET vs ~-2 for
Japanese/French); language-specific imatrix helps only at 2-bit (~+3) and not at 4-bit (+-0.2) - matching our Thai null
result at 3.5 bpw. Below ~3 bits the weights are effectively restructured, so calibration alone can't fix it (ParetoQ).

- [x] Quality floor: never recommend below UD-Q2_K_XL-class quants; warn below it
- [x] Rank vendor QAT checkpoints (e.g. Gemma QAT) above post-training quants of the same model
- [x] Per-language metric: KL divergence / top-1 agreement vs a reference model, not English perplexity
      (`tools/kld_per_language.py`; Q3_K_XL vs UD-IQ2_S table in the README)
- [ ] Sensitivity-aware recipes: measure KLD per tensor, emit `--tensor-type` overrides, keep embeddings/output higher
      for non-Latin scripts
- [x] Mixed multilingual chat-format imatrix (EN+TH+HI+AR+code+math) vs EN-only vs single-language, at 2-bit
      (result: helps only when re-quantizing from Q8; built from BF16 it lost to Unsloth's UD-IQ2_S in 3 of 4 languages)
- [ ] Better 2-bit formats on the multilingual set: IQ2_KT / IQ2_KL (ik_llama.cpp) and EXL3 ~2.5 bpw
- [ ] LoRA self-distillation of a 2-bit 27B from its Q8 teacher (no one has measured this per language yet)
- [ ] Vocabulary trimming per language for GGUF (no tool existed): smaller embedding/output and faster output layer,
      most useful on 1-4B models. `tools/trim_vocab.py` done (#27, #36): Thai + English keeps 60-70% of the
      vocabulary with identical tokenization and the same top-1 next token at 100% of positions on Qwen3-0.6B,
      Qwen3.5-2B and Gemma 4 E2B; files 8-22% smaller, decode +8-12% on CPU. Next: GPU speed/VRAM and benchmark
      accuracy on the catalog models, then a `localllm` option
- [ ] Publish every measured quant with its per-language scores on Hugging Face

## 0.6 - smart router: the right local model for each message (Laya-style)
Idea from the maintainer's Laya router (Local Router Chat): a small, fast classifier reads each message and sends it to
the model that is best *for that request* - fast MoE for everyday chat, the stronger dense model for Chinese/Japanese or
hard reasoning, the user's cloud key only when nothing local is good enough. Unlike generic routers, ours decides from
**measured per-language/per-task scores** in the catalog. Prior evidence: a few-shot LLM router picked correctly on 47/48
Thai/Thai-English requests vs 20/48 for rules, but took ~2.3 s per message; RouteLLM keeps 95% of GPT-4 quality with 26%
strong-model calls.

- [x] Router latency budget < 50 ms: script + keyword language/task detection, microseconds, never the 27B model
      (an embedding router stays an option if rules prove too coarse)
- [ ] Two models resident at once (depends on 0.5 compression: e.g. a fast and a strong model that both fit in 16 GB),
      so routing never waits for a 5-6 s model swap; fall back to "stay on the current model" when a swap would be needed
- [ ] **Sub-second model switching** (research, our own experiments where nothing is published): the measured 2.7 s for a
      12 GB model is only ~4.5 GB/s, far below PCIe 5.0 x16 and page-cache read speed, so the loader is the bottleneck.
      Try in order: (1) specialists as LoRA adapters on one shared base, hot-swapped per request (milliseconds; extract
      LoRAs from same-base fine-tunes via SVD when none exist); (2) keep weights pinned in RAM in GPU-ready form and
      upload in large multi-threaded transfers (target < 1 s for 12 GB); (3) swap models inside one long-lived
      llama-server instead of restarting the process; (4) smaller files from 0.5. Measure each, upstream what works.
- [x] Lazy-load mode (like the maintainer's Creative Core) for PCs that can't hold two models: load a specialist on
      demand, unload when idle. Switch per *task phase*, not per message (hysteresis), keep recently used weights in the
      OS page cache when RAM allows (swap ~2.7 s upload vs ~4.3-6 s cold on a 12 GB model, measured), prefetch the
      likely next model into RAM while the current one answers, tell the user "switching to the code model (~4 s)" with a
      stay-here option. `localllm` picks resident-pair vs lazy-load from the PC's VRAM/RAM. Look at llama-swap first.
- [x] Routing table generated from catalog scores per language/task + measured tok/s, overridable per app (`--models`)
- [x] Lazy-load vs resident pair chosen from VRAM (`--models auto`); measured swap 8.6 s alternating (page cache can't
      hold both on 32 GB RAM), 4.1 s warm with 16 MiB upload buffers; WDDM VRAM oversubscription made both models 4-5x
      slower (rejected)
- [ ] Benchmark: answer quality and end-to-end latency vs a single model, on the multilingual suite + a routing test set
      (Thai/Thai-English set from the Laya research, extended to other languages); ship only if both improve.
      Status: on 16 GB only zh/es gain >= 3 pts and each swap costs 4-9 s, so routing stays opt-in, not the default
- [ ] Specialist pool for local-first routing: slots for code, math/reasoning, vision, embeddings (RAG + the router
      itself), speech-to-text (babelscribe), translation. A specialist joins the catalog only if it beats the generalist on
      its task by more than the benchmark margin (~5 points), fits next to the main model (or swaps fast), and isn't poor
      in the user's language (otherwise the generalist talks to the user and hands only the task to the specialist)
- [ ] `localllm eval` task suites beyond multiple choice: code (HumanEval+/LiveCodeBench-style), math (GSM8K/MATH-500),
      vision QA, translation - needed to measure specialists honestly. Done: math (MGSM, 11 languages, `--suites math`),
      translation (FLORES-101, 101 languages, chrF++ identical to sacreBLEU, `--suites translate`)
- [x] Show which model answered and why (`X-Localllm-Model`, shown in `localllm chat`); override with `--model`

## Later
- Shared prefix cache (block/radix, like vLLM/SGLang) instead of per-slot prompt copies - needs llama.cpp work
- More measured GPUs: `localllm eval` results from contributors feed the catalog
- Writing-quality evaluation (not just multiple choice)

Research behind this roadmap, with sources and numbers (Thai): [docs/research-th.md](docs/research-th.md).
