"""Local-first routing. Every chat request stays on this PC unless the user turned routing on in
~/.localllm/route.json AND a rule matches. Keys are never stored by localllm: they come from environment variables (or
the OS keychain via the optional `keyring` package) named in the config.

Example ~/.localllm/route.json
{
  "enabled": true,
  "provider": "openrouter",
  "providers": {
    "openrouter": {"kind": "openai", "url": "https://openrouter.ai/api", "key_env": "OPENROUTER_API_KEY",
                   "model": "anthropic/claude-sonnet-4"},
    "anthropic":  {"kind": "anthropic", "url": "https://api.anthropic.com", "key_env": "ANTHROPIC_API_KEY",
                   "model": "claude-sonnet-4-5"}
  },
  "rules": {"max_local_prompt_tokens": 6000, "language_floor": 60, "cloud_model_names": true},
  "local_model": "qwen3.8-27b-q3"
}
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from .runtime import HOME

CONFIG = HOME / "route.json"
SCRIPTS = [("th", r"[฀-๿]"), ("hi", r"[ऀ-ॿ]"), ("ar", r"[؀-ۿ]"), ("ja", r"[぀-ヿ]"),
           ("ko", r"[가-힯]"), ("zh", r"[一-鿿]"), ("ru", r"[Ѐ-ӿ]"), ("he", r"[֐-׿]"),
           ("el", r"[Ͱ-Ͽ]"), ("bn", r"[ঀ-৿]"), ("ta", r"[஀-௿]")]
CLOUD_NAMES = re.compile(r"^(gpt-|o\d|claude-|gemini-|anthropic/|openai/|google/)", re.I)


def detect_language(text: str) -> str:
    """Cheap script-based guess (no model): enough to look up per-language scores."""
    counts = {lang: len(re.findall(rx, text)) for lang, rx in SCRIPTS}
    lang, n = max(counts.items(), key=lambda kv: kv[1])
    return lang if n >= 3 else "en"


MATH = re.compile(r"(\d\s*[-+*/×÷^=]\s*\d|\b(calculate|how many|how much|total|percent|average|solve|equation)\b|"
                  r"คำนวณ|เท่าไร|เท่าไหร่|กี่|多少|几|计算|いくつ|何個|計算|كم|कितन)", re.I)


TRANSLATE = re.compile(r"\b(translate|translation|traduce|traduis|traduza|übersetze|vertaal|terjemahkan|dịch)\b|"
                       r"แปล|翻译|翻譯|翻訳|번역|ترجم|अनुवाद|перевед", re.I)


def detect_task(text: str) -> str:
    """Cheap task guess (no model): 'translate', 'math' for word problems and arithmetic, else 'general'."""
    if TRANSLATE.search(text[:200]):          # the request is usually stated up front
        return "translate"
    return "math" if len(re.findall(r"\d+", text)) >= 2 and MATH.search(text) else "general"


# Who decides the task, in order. The keyword rules only commit when they find a translate/math cue; the embedding
# classifier (taskclf) always answers. Shared test (tools/router, 144 messages, 15 languages), committed head trained
# with the generated requests, CI CPU runners: keyword rules first 98.6%, embedding only 97.9%, keyword rules alone 56.9%.
TASK_ORDER = ("keyword", "embedding")


def classify_task(text: str, clf=None, order=TASK_ORDER) -> tuple[str, str]:
    """(task, how): how = 'keyword', 'embedding p=0.87', or 'keyword fallback because <reason>' when the embedding
    classifier `clf` (taskclf.Classifier) is missing or failed. clf=None means keyword rules only."""
    from .taskclf import Unavailable
    kw, failed = detect_task(text), None
    for step in order:
        if step == "keyword" and kw != "general":
            break
        if step == "embedding" and clf is not None:
            try:
                label, p = clf.classify(text)
                return label, f"embedding p={p:.2f}"
            except Unavailable as e:
                failed = str(e)
    return kw, f"keyword fallback because {failed}" if failed else "keyword"


@dataclass
class Provider:
    name: str
    kind: str          # "openai" (OpenAI-compatible) or "anthropic"
    url_base: str
    key: str
    model: str

    def serves(self, path: str) -> bool:
        return (self.kind == "openai" and path == "/v1/chat/completions") or \
               (self.kind == "anthropic" and path == "/v1/messages")

    def headers(self, path: str) -> dict:
        if self.kind == "anthropic":
            return {"x-api-key": self.key, "anthropic-version": "2023-06-01"}
        return {"Authorization": f"Bearer {self.key}"}

    def adapt(self, body: bytes, path: str) -> bytes:
        d = json.loads(body or b"{}")
        d["model"] = self.model
        return json.dumps(d).encode()


@dataclass
class Decision:
    label: str                      # shown to the user in X-Localllm-Route
    provider: Provider | None = None


def load_config() -> dict:
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"enabled": False}


def _key(env_name: str) -> str | None:
    if os.environ.get(env_name):
        return os.environ[env_name]
    try:
        import keyring  # optional
        return keyring.get_password("localllm", env_name)
    except Exception:
        return None


def providers(cfg: dict) -> dict[str, Provider]:
    out = {}
    for name, p in (cfg.get("providers") or {}).items():
        key = _key(p.get("key_env", ""))
        if key:
            out[name] = Provider(name, p.get("kind", "openai"), p["url"], key, p.get("model", ""))
    return out


def _text(body: dict) -> str:
    parts = []
    for m in body.get("messages", []):
        c = m.get("content")
        parts.append(c if isinstance(c, str) else " ".join(x.get("text", "") for x in c or [] if isinstance(x, dict)))
    if isinstance(body.get("system"), str):
        parts.append(body["system"])
    return "\n".join(parts)


def _last_user(body: dict) -> str:
    for m in reversed(body.get("messages", [])):
        if m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else " ".join(x.get("text", "") for x in c or [] if isinstance(x, dict))
    return ""


def decide(body: dict, path: str, cfg: dict | None = None) -> Decision:
    cfg = load_config() if cfg is None else cfg
    if not cfg.get("enabled"):
        return Decision("local")
    provs = providers(cfg)
    usable = [p for p in provs.values() if p.serves(path)]
    preferred = provs.get(cfg.get("provider", ""))
    cloud = preferred if preferred and preferred.serves(path) else (usable[0] if usable else None)
    if not cloud:
        return Decision("local (no cloud key configured for this API)")
    rules = cfg.get("rules") or {}
    if rules.get("cloud_model_names", True) and CLOUD_NAMES.match(str(body.get("model", ""))):
        return Decision(f"cloud:{cloud.name} (app asked for {body['model']})", cloud)
    tokens = len(_text(body)) // 3
    if tokens > rules.get("max_local_prompt_tokens", 10 ** 9):
        return Decision(f"cloud:{cloud.name} (prompt ~{tokens} tokens > local limit)", cloud)
    floor = rules.get("language_floor")
    if floor:
        from . import catalog
        lang = detect_language(_last_user(body))
        local = cfg.get("local_model")
        scores = catalog.MODELS.get(local, {}).get("scores", {}) if local else {}
        if any(k.startswith(lang + "/") for k in scores) and catalog.score(local, lang) < floor:
            return Decision(f"cloud:{cloud.name} ({lang} score {catalog.score(local, lang):.0f} < floor {floor})", cloud)
    return Decision("local")


SWITCH_POINTS = 3.0   # swapping models costs seconds on one GPU: only switch for a clear accuracy gain


def pick_local(text: str, keys: list[str], current: str | None, vram_gb: float, ram_free_gb: float = 0.0,
               clf=None) -> tuple[str, str]:
    """Smart local routing (0.6): the measured-best model for this message's language and task among `keys`, with
    hysteresis - stay on the loaded model unless the other one scores >= SWITCH_POINTS higher. The reason names the
    language, the task and what decided it. clf: taskclf.Classifier, or None for keyword rules only.
    Returns (model key, reason)."""
    from . import catalog
    lang, (task, how) = detect_language(text), classify_task(text, clf)
    tag = f"{lang} {task} by {how}"
    best = catalog.pick(vram_gb, lang, ram_free_gb, candidates=keys, task=task) or current or keys[0]
    if current and best != current and current in keys:
        gain = catalog.score(best, lang, task) - catalog.score(current, lang, task)
        if gain < SWITCH_POINTS:
            return current, f"{tag}: kept loaded model ({best} only {gain:+.1f} pts)"
        return best, f"{tag}: {gain:+.1f} pts vs {current}"
    return best, f"{tag}: best measured"


def compare(prompt: str, local_url: str, cfg: dict | None = None) -> list[dict]:
    """`localllm route --test`: the same prompt to the local model and to each configured cloud provider."""
    import time
    import urllib.request
    cfg = load_config() if cfg is None else cfg
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 400,
            "chat_template_kwargs": {"enable_thinking": False}}
    targets = [("local", local_url.rstrip("/") + "/v1/chat/completions", {}, body, None)]
    for p in providers(cfg).values():
        if p.kind == "openai":
            targets.append((p.name, p.url_base.rstrip("/") + "/v1/chat/completions", p.headers("/v1/chat/completions"),
                            {"model": p.model, "messages": body["messages"], "max_tokens": 400},
                            (cfg["providers"][p.name].get("price_in"), cfg["providers"][p.name].get("price_out"))))
    rows = []
    for name, url, hdrs, b, price in targets:
        t0 = time.time()
        try:
            req = urllib.request.Request(url, data=json.dumps(b).encode(),
                                         headers={"Content-Type": "application/json", **hdrs})
            d = json.load(urllib.request.urlopen(req, timeout=300))
            u = d.get("usage", {})
            cost = None
            if price and all(x is not None for x in price):
                cost = (u.get("prompt_tokens", 0) * price[0] + u.get("completion_tokens", 0) * price[1]) / 1e6
            rows.append({"target": name, "seconds": round(time.time() - t0, 2), "cost_usd": cost,
                         "tokens": u.get("completion_tokens"), "answer": d["choices"][0]["message"].get("content", "")})
        except Exception as e:  # report, don't crash the comparison
            rows.append({"target": name, "error": str(e)})
    return rows
