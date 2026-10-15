"""Publish the measured model catalog as versioned JSON (#164).

  python tools/build_catalog.py [--check]

Input: localllm/catalog.py, the only place the numbers are typed in. Output: data/catalog/v1/catalog.json, which follows
the hand-written contract data/catalog/v1/schema.json. site/build.py copies both to <site>/catalog/v1/ so other tools
can read them over HTTPS. The output holds no dates or versions, so it changes only when the measurements change;
`catalog_rev` (a hash of the models) changes with them. A breaking change to the format goes to v2/, never into v1/.
--check exits 1 if catalog.json is out of date (for CI).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from localllm import catalog  # noqa: E402
from build_leaderboard import MAINTAINER  # noqa: E402

VERSION = 1
OUT_DIR = ROOT / "data" / "catalog" / f"v{VERSION}"
SITE_URL = "https://phonology024.github.io/make-localllm-easier"
REPO = "https://github.com/phonology024/make-localllm-easier"
LICENSE = "MIT"
# the one PC the maintainers measure on (see docs/how-it-works.md, "Where it is tested")
HARDWARE = {"id": "rx9070xt-windows-vulkan", "gpu": MAINTAINER["gpu"], "vram_gb": MAINTAINER["vram_gb"],
            "ram_gb": MAINTAINER["ram_gb"], "os": MAINTAINER["os"], "runtime": "llama.cpp", "backend": "vulkan"}


def model_entry(key: str, m: dict) -> dict:
    speed = {"hardware": HARDWARE["id"], "llama_cpp": m["build"], "decode_tok_s": m["tok_s_9070xt"],
             "confidence": "measured", "mtp_draft_tokens": 2 if m.get("mtp") else 0}
    if m.get("tok_s_offload"):
        speed["moe_offload"] = [{"layers_in_ram": n, "decode_tok_s": t} for n, t in sorted(m["tok_s_offload"].items())]
    return {
        "id": key,
        "format": "gguf",
        "hf_repo": m["repo"], "file": m["file"], "rev": m["rev"], "sha256": m["sha256"],
        "bits_per_weight": m["bpw"], "qat": m["qat"], "max_ctx": m["max_ctx"],
        "moe": {"layers": m["moe"]["layers"], "expert_gib_per_layer": m["moe"]["expert_gb_per_layer"]}
        if m.get("moe") else None,
        "memory": {"weights_gib": m["gb"], "kv_kib_per_token_q8": m["kv_kb_per_token"],
                   "fixed_cache_gib": m["fixed_cache_gb"], "checkpoint_gib": m["checkpoint_gb"],
                   "cpu_mapped_gib": m.get("cpu_mapped_gb")},
        "vision": {"mmproj_file": m["mmproj"], "mmproj_gib": m["mmproj_gb"], "mmproj_sha256": m["mmproj_sha256"]}
        if m.get("mmproj") else None,
        "speed": [speed],
        "scores": dict(sorted(m["scores"].items())),
        "note": m["note"],
    }


def build() -> dict:
    models = [model_entry(k, m) for k, m in catalog.MODELS.items()]
    rev = hashlib.sha256(json.dumps(models, sort_keys=True).encode()).hexdigest()[:16]
    return {
        "$schema": f"{SITE_URL}/catalog/v{VERSION}/schema.json",
        "catalog_version": VERSION,
        "catalog_rev": rev,
        "license": LICENSE,
        "source": REPO,
        "scores_are": "0-100 from `localllm eval`, keyed language/suite (en/global, th/regional, ...): accuracy in percent, "
                      "chrF++ for */translate and */summarize",
        "hardware": [HARDWARE],
        "models": models,
    }


def render() -> str:
    return json.dumps(build(), indent=1, ensure_ascii=False) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if catalog.json is out of date")
    a = ap.parse_args()
    out = OUT_DIR / "catalog.json"
    text = render()
    if a.check:
        if not out.exists() or out.read_text(encoding="utf-8") != text:
            sys.exit(f"{out.relative_to(ROOT)} is out of date: run python tools/build_catalog.py")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"{len(catalog.MODELS)} models -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
