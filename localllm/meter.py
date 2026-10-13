"""Savings meter (#97): how many tokens this PC answered locally, per month, and what they would have cost.

The gateway reads the token counts llama-server reports at the end of every local chat answer (OpenAI `usage` /
`timings`, Anthropic `usage`) and adds them up in ~/.localllm/savings.json. `cache_n` is the part of the prompt that
was re-used from the prompt cache instead of being read again: agents resend a big fixed prefix every call, so this
shows whether that reuse works.

Dollars are an estimate from prices *you* set (`localllm savings --price 3 15` = $3 in / $15 out per million tokens,
for the cloud model you would otherwise use). No prices ship with localllm: they change, and a made-up number would
be worse than none.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

from . import runtime

def _last(text: str, field: str) -> int | None:
    found = re.findall(rf'"{field}"\s*:\s*(\d+)', text)
    return int(found[-1]) if found else None


def parse_usage(tail: bytes) -> dict | None:
    """Token counts from the end of a response; the last value of a field wins (streams repeat them). `usage` gives the
    whole prompt; llama-server's `timings` split it into prompt_n (read now) + cache_n (re-used from the cache)."""
    text = tail.decode("utf-8", "replace")
    cached = _last(text, "cache_n")
    if cached is None:
        cached = _last(text, "cache_read_input_tokens")
    total = _last(text, "prompt_tokens")
    if total is None:
        total = _last(text, "input_tokens")
    if total is None and _last(text, "prompt_n") is not None:
        total = _last(text, "prompt_n") + (cached or 0)
    out = _last(text, "completion_tokens")
    for f in ("output_tokens", "predicted_n"):
        out = _last(text, f) if out is None else out
    if total is None and out is None:
        return None
    return {"input": total or 0, "output": out or 0, "cached": cached or 0}


class Meter:
    def __init__(self, path: Path | None = None):
        self.path = path or runtime.HOME / "savings.json"
        self.lock = threading.Lock()

    def load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def add(self, usage: dict, month: str | None = None) -> None:
        month = month or time.strftime("%Y-%m")
        with self.lock:
            d = self.load()
            m = d.setdefault("months", {}).setdefault(month, {"requests": 0, "input": 0, "cached": 0, "output": 0})
            m["requests"] += 1
            for k in ("input", "cached", "output"):
                m[k] += int(usage.get(k, 0))
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
            tmp.replace(self.path)

    def set_price(self, input_per_m: float, output_per_m: float, label: str = "") -> None:
        with self.lock:
            d = self.load()
            d["price"] = {"input_per_m": input_per_m, "output_per_m": output_per_m, "label": label}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(d, indent=1), encoding="utf-8")

    def month(self, month: str | None = None) -> dict:
        """This month's totals, plus `saved_usd` when a price is set."""
        d = self.load()
        m = dict(d.get("months", {}).get(month or time.strftime("%Y-%m"),
                                          {"requests": 0, "input": 0, "cached": 0, "output": 0}))
        p = d.get("price")
        if p:
            m["saved_usd"] = round((m["input"] * p["input_per_m"] + m["output"] * p["output_per_m"]) / 1e6, 2)
            m["price_label"] = p.get("label", "")
        return m


def line(m: dict) -> str:
    """One line for doctor, /status and the UI."""
    if not m["requests"]:
        return "nothing answered locally yet this month"
    reuse = f", {100 * m['cached'] / m['input']:.0f}% of prompt tokens from the cache" if m["input"] else ""
    money = (f" - about ${m['saved_usd']:.2f} at your prices{' (' + m['price_label'] + ')' if m.get('price_label') else ''}"
             if "saved_usd" in m else " (`localllm savings --price IN OUT` to see it in dollars)")
    return (f"this month: {m['requests']} answers, {m['input']:,} tokens in / {m['output']:,} out answered locally"
            f"{reuse}{money}")
