"""Per-language quantization damage: KL divergence and top-1 agreement of a quant vs a reference model (issue #22).

English perplexity hides multilingual loss, so we measure each language separately with llama.cpp's
`llama-perplexity --kl-divergence`:

  python tools/kld_per_language.py base  --ref REF.gguf  --texts DIR      # once: save the reference logits per language
  python tools/kld_per_language.py score --model QUANT.gguf --texts DIR   # per quant: KLD + "same top token" per language

DIR holds one UTF-8 text file per language (th.txt, hi.txt, ...). Results go to tools/kld_results.jsonl.
The reference should be the most precise file that runs (BF16, else Q8_0; partial GPU offload is fine).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

PPL = Path.home() / "tools" / "llama.cpp-b11457" / "vulkan" / "llama-perplexity.exe"
OUT = Path(__file__).with_name("kld_results.jsonl")


def parse(out: str) -> dict:
    def grab(pat):
        m = re.search(pat, out)
        return float(m.group(1)) if m else None
    return {"kld_mean": grab(r"Mean\s+KLD:\s+([\d.]+)"), "kld_p99": grab(r"99\.0%\s+KLD:\s+([\d.]+)"),
            "same_top_pct": grab(r"Same top p:\s+([\d.]+)"), "ppl": grab(r"Mean PPL\(Q\)\s+:\s+([\d.]+)")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["base", "score"])
    ap.add_argument("--ref"); ap.add_argument("--model"); ap.add_argument("--texts", required=True)
    ap.add_argument("--chunks", type=int, default=8); ap.add_argument("--ctx", type=int, default=512)
    a = ap.parse_args()
    common = ["-c", str(a.ctx), "--chunks", str(a.chunks), "-fa", "on", "-dev", "Vulkan0"]
    for txt in sorted(Path(a.texts).glob("*.txt")):
        base = txt.with_suffix(".kld")
        if a.mode == "base":
            out = subprocess.run([str(PPL), "-m", a.ref, "-f", str(txt), "--kl-divergence-base", str(base), *common],
                                 capture_output=True, text=True, errors="replace")
            print(f"{txt.stem}: reference logits saved ({base.stat().st_size / 2**30:.2f} GB)" if base.exists()
                  else f"{txt.stem}: FAILED\n{out.stdout[-800:]}{out.stderr[-800:]}")
            continue
        out = subprocess.run([str(PPL), "-m", a.model, "--kl-divergence-base", str(base), "--kl-divergence", *common,
                              "-ngl", "999"], capture_output=True, text=True, errors="replace")
        r = {"model": Path(a.model).name, "lang": txt.stem, **parse(out.stdout + out.stderr)}
        print(json.dumps(r))
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
