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
| `results` | object | `{run name: {"lang/suite": {acc, ...}, "_meta": {...}}}` from `localllm eval` |

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
   "flags": ["..."]                 // reasons to double-check this row
 }],
 "platforms": {"Windows/AMD": {"maintainers": true, "community_reports": 0}, ...}
}
```
