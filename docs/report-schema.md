# Report and leaderboard formats

Both are versioned. Adding a field is not a breaking change; renaming or removing one, or changing its meaning, is,
and bumps `schema`. Readers must ignore fields they don't know.

## `localllm report` (schema 1)

Written by `localllm/report.py`, saved as `~/.localllm/report-<date>.json`.

| field | type | meaning |
|---|---|---|
| `schema` | int | 1 |
| `localllm` | str | localllm version that wrote it |
| `date` | str | `YYYY-MM-DD` |
| `os`, `arch`, `cpu` | str | from Python's `platform` module |
| `ram_gb` | float | installed system RAM |
| `language` | str | the PC's language (ISO 639-1) |
| `llama_cpp` | str or null | llama.cpp build, e.g. `b11476` |
| `gpu`, `vram_gb`, `driver` | str / float / str | the GPU localllm would use |
| `devices` | list | every GPU llama.cpp sees: `{name, vram_gb}` |
| `pick` | str or null | the catalog model `localllm` would pick here |
| `tune` | object | `{model file: {llama.cpp build: {gpu, env, mtp, tok_s}}}` from `localllm tune` |
| `host` | str, optional | `cloud` or `cloud:<provider>` for a rented machine (`LOCALLLM_REPORT_HOST`, set by `tools/cloud_lab.sh`); absent for your own PC |
| `results` | object | `{run name: {"lang/suite": {acc, ...}, "_meta": {...}}}` from `localllm eval`. Your local `results.json` and the measurement runner's lines also keep `outcomes` (per item, `"1"`/`"0"`) and `items_sha` for right/wrong suites, for `tools/paired.py`; shared reports leave them out |

No user names, home folders, paths or keys: `report.scrub` removes them before anything is printed or saved.

## `data/leaderboard.json` (schema 1)

Written by `tools/build_leaderboard.py` from the catalog and `reports/`.

```
{
 "schema": 1,
 "reports": 3,                      // reports that passed validation
 "entries": [{
   "source": "maintainers" | "community",
   "gpu": "...", "vendor": "AMD" | "NVIDIA" | "Intel" | "Apple" | null,
   "vram_gb": 15.9, "ram_gb": 32.0, "os": "Windows 11", "os_family": "Windows" | "Linux" | "macOS" | null,
   "llama_cpp": "b11476",
   "model": "gemma4-26b-a4b-qat",   // catalog key when we know the file, else the name in the report
   "tok_s": 90,                     // decode tok/s, null if not tuned
   "scores": {"th/math": 89.6, ...}, // accuracy %, chrF++ for */translate
   "host": "cloud:runpod",          // only for rows measured on a rented cloud machine
   "flags": ["..."]                 // reasons to double-check this row
 }],
 "platforms": {"Windows/AMD": {"maintainers": true, "community_reports": 0}, ...}
}
```

## `data/catalog/v1/catalog.json` (catalog version 1)

The measured catalog (`localllm/catalog.py`) as JSON for other tools, written by `tools/build_catalog.py` and checked
against the JSON Schema [data/catalog/v1/schema.json](../data/catalog/v1/schema.json) in the tests. The website serves
both at `https://phonology024.github.io/make-localllm-easier/catalog/v1/catalog.json` and `.../catalog/v1/schema.json`.
License: MIT, the same as the code.

```
{
 "catalog_version": 1,
 "catalog_rev": "ae434cf45eaea48d",   // changes whenever any model entry changes
 "license": "MIT",
 "hardware": [{"id": "rx9070xt-windows-vulkan", "gpu": "AMD Radeon RX 9070 XT", "vram_gb": 15.9, ...}],
 "models": [{
   "id": "gemma4-26b-a4b-qat", "format": "gguf",
   "hf_repo": "...", "file": "...gguf", "rev": "<40-hex commit>", "sha256": "<64 hex>",
   "bits_per_weight": 4.2, "qat": true, "max_ctx": 262144,
   "moe": {"layers": 30, "expert_gib_per_layer": 0.4} | null,
   "memory": {"weights_gib": 13.3, "kv_kib_per_token_q8": 10.9, "fixed_cache_gib": 0.11, "checkpoint_gib": 0.11,
              "cpu_mapped_gib": 0.51 | null},
   "vision": {"mmproj_file": "...", "mmproj_gib": 1.11, "mmproj_sha256": "..."} | null,
   "speed": [{"hardware": "rx9070xt-windows-vulkan", "llama_cpp": "b11476", "decode_tok_s": 90,
              "confidence": "measured" | "estimated", "mtp_draft_tokens": 0,
              "moe_offload": [{"layers_in_ram": 8, "decode_tok_s": 45}, ...]}],
   "scores": {"th/regional": 65.7, ...},  // accuracy %, chrF++ for */translate and */summarize
   "note": "..."
 }]
}
```

Version 1 only gains optional fields. Renaming, removing or retyping a field means a new `v2/` folder next to `v1/`,
and `v1/` keeps being published.

