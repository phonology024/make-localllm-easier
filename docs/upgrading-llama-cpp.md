# Moving to a new llama.cpp build

Every number in the catalog was measured on one llama.cpp build, pinned in `localllm/runtime.py` (`LLAMA_BUILD`, with
the SHA-256 of each download). A new build can be faster, slower, or break a model, so the pin only moves after these
steps (#95).

1. **Smoke test on this PC.** `python tools/smoke_build.py bNNNNN` loads every catalog model that fits on the candidate
   and on the pinned build, asks one question, measures decode speed and checks the spill counter. It passes when every
   model loads, answers, doesn't spill, and is at most 5% slower (`--tolerance`). Results go to
   `tools/smoke_results.jsonl`; a failure means no bump.
2. **Read the release notes** between the two builds. If anything touches Vulkan, MoE, the KV cache, MTP or the server's
   cache options, the headline numbers need a re-run: `python tools/runner.py tools/runner_rebaseline.json` (set
   `"llama_cpp"` in the manifest to the new build) and `python tools/measure_footprint.py` for the RAM table.
3. **Pin it.** In `runtime.py` set `LLAMA_BUILD` and replace `LLAMA_SHA256` with the new assets' digests (GitHub lists
   them on the release; `gh release view bNNNNN -R ggml-org/llama.cpp --json assets` prints them). Set `"build"` on
   each catalog model whose numbers were re-measured. Models still carrying the old build show a note in
   `localllm doctor` ("measured on bOLD; this PC runs bNEW"), so stale scores are visible, not hidden.
4. **Re-run the cache-sensitive checks.** `localllm tune` results are stored per build, so they re-measure on their
   own. Run `pytest` (the README doctor sample and RAM table are checked against the data).

A user who sets `LOCALLLM_LLAMA_BUILD=latest` or `=bNNNNN` skips the pin on their own PC; `doctor` tells them when
their build differs from the one the numbers came from.
