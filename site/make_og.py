"""Draw site/assets/og.png (1200x630 social preview) from the headline numbers. Needs matplotlib."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import headline  # noqa: E402

n = headline.numbers()["gemma-4-26B-A4B QAT"]
fig = plt.figure(figsize=(12, 6.3), dpi=100)
fig.patch.set_facecolor("#111214")
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1200); ax.set_ylim(0, 630)
mono = {"family": "monospace"}
ax.text(80, 520, "$ localllm", color="#3cc488", fontsize=34, fontdict=mono, weight="bold")
ax.text(80, 420, "Run the best local LLM", color="#e8e8e4", fontsize=35, weight="bold")
ax.text(80, 350, "your PC can handle, in one command.", color="#e8e8e4", fontsize=35, weight="bold")
ax.text(80, 270, "Picked from measured scores in your language. AMD, NVIDIA, Intel, Apple.", color="#a1a5ac",
        fontsize=19)
for y, gb, c, label in ((170, n["default_gb"], "#5f6772", "llama.cpp defaults"),
                        (120, n["localllm_gb"], "#3cc488", "localllm")):
    w = gb / n["default_gb"] * 620
    ax.add_patch(plt.Rectangle((80, y), w, 34, color=c))
    ax.text(80 + w + 14, y + 8, f"{gb:.1f} GB  {label}", color="#e8e8e4", fontsize=18, fontdict=mono)
ax.text(80, 60, "System RAM after a 30-turn chat, RX 9070 XT. Free, MIT licensed.", color="#a1a5ac", fontsize=16)
out = Path(__file__).with_name("assets") / "og.png"
fig.savefig(out, facecolor=fig.get_facecolor())
print(f"wrote {out}")
