"""The README's headline RAM numbers, from one place: tools/footprint_results.jsonl (#80).

  python tools/headline.py          print the numbers the README table must show
  python tools/headline.py chart    redraw docs/img/ram-30-turns.png from them (needs matplotlib)

tests/test_readme.py fails when the README table and these numbers disagree.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "tools" / "footprint_results.jsonl"
CHART = ROOT / "docs" / "img" / "ram-30-turns.png"
# README name -> (llama.cpp defaults run, localllm run) in footprint_results.jsonl, both 30 turns on the RX 9070 XT
RUNS = {"Qwen3.8-27B Q3": ("qwen-default", "qwen-tuned"),
        "gemma-4-26B-A4B QAT": ("gemma-default", "gemma-tuned")}


def numbers() -> dict[str, dict]:
    """{name: {default_gb, localllm_gb, default_tok_s, localllm_tok_s}} after the 30th turn."""
    runs = {}
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            runs[r["label"]] = r["final"]
    out = {}
    for name, (base, ours) in RUNS.items():
        b, o = runs[base], runs[ours]
        assert b["turn"] == o["turn"] == 30, name
        out[name] = {"default_gb": round(b["working_set_gb"], 2), "localllm_gb": round(o["working_set_gb"], 2),
                     "default_tok_s": b["tok_s"], "localllm_tok_s": o["tok_s"]}
    return out


def chart(out: Path = CHART) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = numbers()
    names = list(n)[::-1]
    fig, ax = plt.subplots(figsize=(12.8, 6.4), dpi=100)
    y = range(len(names))
    h = 0.36
    for off, key, color, label in ((h / 2, "default_gb", "#9ca5b3", "llama.cpp defaults"),
                                   (-h / 2, "localllm_gb", "#21a06b", "localllm")):
        vals = [n[k][key] for k in names]
        bars = ax.barh([i + off for i in y], vals, height=h, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.text(v + 0.12, b.get_y() + h / 2, f"{v:.2f} GB", va="center", fontsize=12)
    ax.set_yticks(list(y), names, fontsize=13)
    ax.set_xlim(0, max(v["default_gb"] for v in n.values()) * 1.24)
    ax.set_xlabel("System RAM after a 30-turn chat (GB, lower is better)", fontsize=13)
    ax.set_title("About 4x less RAM, same speed (RX 9070 XT, 32 GB RAM)", fontsize=14, loc="left")
    ax.tick_params(axis="x", labelsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=13)
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    if sys.argv[1:] == ["chart"]:
        chart()
    else:
        print(json.dumps(numbers(), indent=1))
