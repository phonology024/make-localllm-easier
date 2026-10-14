"""`localllm report --share`: a card with this PC's result, to post anywhere (#118).

One SVG, 1200 x 630 (the size social sites preview), written next to the report in ~/.localllm/. It shows the GPU,
the model localllm picked for it, the decode speed `localllm tune` measured here, the score in the user's language
and the system RAM the model uses, plus the link to that GPU's leaderboard page. It is drawn from the report, which
has no names, paths or keys, and every text is escaped. Standard library only; nothing is uploaded.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

from . import catalog

SITE = "https://phonology024.github.io/make-localllm-easier"
W, H = 1200, 630


def gpu_slug(gpu: str) -> str:
    """Same slug as the website's per-GPU pages (site/build.py)."""
    return re.sub(r"[^a-z0-9]+", "-", gpu.lower()).strip("-") or "unknown"


def _mean(scores: dict, lang: str) -> float | None:
    vals = [v for k, v in scores.items() if k.split("/")[0] == lang and not k.endswith("/translate")
            and isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 1) if vals else None


def facts(rep: dict) -> dict:
    """What the card says, from a `localllm report`. Only numbers this PC measured are called measured here."""
    key = rep.get("pick") if rep.get("pick") in catalog.MODELS else None
    m = catalog.MODELS.get(key or "", {})
    lang = rep.get("language") or "en"
    tok = None
    for file, builds in (rep.get("tune") or {}).items():
        if m and file == m.get("file"):
            vals = [b.get("tok_s") for b in builds.values() if isinstance(b, dict)]
            vals = [v for v in vals if isinstance(v, (int, float))]
            tok = max(vals) if vals else tok
    results = {k: v for k, v in (rep.get("results") or {}).items() if not k.startswith("_") and isinstance(v, dict)}
    mine = results.get(key) or (next(iter(results.values())) if len(results) == 1 else None)
    score, source = None, ""
    if mine:
        score = _mean({k: v.get("acc") for k, v in mine.items() if isinstance(v, dict)}, lang)
        source = "measured on this PC" if score is not None else ""
    if score is None and m:
        score, source = _mean(m["scores"], lang), "catalog score"
    ram = None
    if m:
        from . import runtime, sizing
        args = runtime.server_args(Path(m["file"]), None, 8080, catalog.MIN_CTX, m["mtp"],
                                   ram_total_gb=rep.get("ram_gb") or 32.0)
        ram = sizing.ram_estimate_gb(m, args)["total_gb"]
    gpu = rep.get("gpu") or "CPU only"
    return {"gpu": gpu, "vram_gb": rep.get("vram_gb") or 0, "ram_gb": rep.get("ram_gb") or 0, "os": rep.get("os") or "",
            "model": key, "tok_s": tok, "score": score, "score_source": source, "lang": lang, "ram_used_gb": ram,
            "url": f"{SITE}/leaderboard/gpu/{gpu_slug(gpu)}/"}


def _text(x: int, y: int, s: str, size: int, weight: int = 400, fill: str = "#e6e8e3", anchor: str = "start") -> str:
    return (f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}">{html.escape(s, quote=False)}</text>')


def _short(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def render(f: dict) -> str:
    """The card as SVG text."""
    stats = [(f"{f['tok_s']:g} tok/s" if f["tok_s"] is not None else "not tuned yet",
              "decode speed, measured here" if f["tok_s"] is not None else "run: localllm tune"),
             (f"{f['score']:.1f}%" if f["score"] is not None else "-",
              f"score in {f['lang']} ({f['score_source']})" if f["score"] is not None else f"not measured in {f['lang']}"),
             (f"{f['ram_used_gb']:.1f} GB" if f["ram_used_gb"] is not None else "-", "system RAM used (est.)")]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
             'font-family="system-ui, -apple-system, Segoe UI, Roboto, Noto Sans, sans-serif">',
             f'<rect width="{W}" height="{H}" fill="#141614"/>',
             f'<rect x="0" y="0" width="{W}" height="8" fill="#3dbb84"/>',
             _text(60, 90, "localllm", 34, 700, "#3dbb84"),
             _text(60, 170, _short(f["gpu"], 40), 56, 700),
             _text(60, 215, f"{f['vram_gb']:g} GB VRAM · {f['ram_gb']:g} GB RAM · {_short(f['os'], 30)}", 26, 400, "#9aa096"),
             _text(60, 280, f"runs {f['model']}" if f["model"] else "no measured model fits this GPU yet", 32, 600)]
    for i, (big, small) in enumerate(stats):
        x = 60 + i * 370
        parts += [f'<rect x="{x}" y="320" width="340" height="170" rx="16" fill="#1c1f1c" stroke="#2e332e"/>',
                  _text(x + 24, 405, big, 50, 700), _text(x + 24, 455, small, 21, 400, "#9aa096")]
    parts += [_text(60, 570, f["url"].replace("https://", ""), 22, 400, "#9aa096"),
              _text(W - 60, 570, "measured, not guessed", 22, 600, "#3dbb84", "end"), "</svg>"]
    return "\n".join(parts) + "\n"


def write(rep: dict, folder: Path) -> tuple[Path, str]:
    f = facts(rep)
    out = folder / f"share-{rep.get('date') or 'today'}.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(f), encoding="utf-8")
    return out, f["url"]
