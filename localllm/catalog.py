"""Models we have measured end to end. `scores` = accuracy (%) from `localllm eval` keyed "lang/suite" (see bench.py).
gb = weights in GiB; kv_kb_per_token = KV cache per token at q8 (from the GGUF attention layout).
Speeds: llama-server decode tok/s on an RX 9070 XT 16 GB (Windows, Vulkan) with the tuned launch in runtime.py.
mmproj = the vision projector in the same repo (loaded only with --vision), mmproj_gb its size.
Add a model only after measuring it with `localllm eval`."""

MODELS = {
    "qwen3.8-27b-q3": {
        "repo": "unsloth/Qwen3.8-27B-GGUF", "file": "Qwen3.8-27B-UD-Q3_K_XL.gguf", "gb": 12.2, "bpw": 3.8, "qat": False,
        "kv_kb_per_token": 34.8, "fixed_cache_gb": 0.15, "checkpoint_gb": 0.15, "max_ctx": 262144, "tok_s_9070xt": 50, "mtp": True, "vk_fix": True,
        "mmproj": "mmproj-F16.gguf", "mmproj_gb": 0.86,
        "cpu_mapped_gb": 0.51,  # measured: ~521 MiB of the 12.2 GiB model stays CPU-mapped (large 248k vocab)
        "scores": {"en/global": 81.5, "zh/global": 76.2, "zh/regional": 74.7, "es/global": 80.2, "es/regional": 76.8, "hi/global": 69.0, "hi/regional": 74.3, "ar/global": 70.8, "ar/regional": 71.2, "ja/global": 73.5, "ja/regional": 87.6, "th/regional": 67.1,
                   "en/math": 94.4, "th/math": 87.2, "zh/math": 84.4,
                   "th/translate": 54.1, "zh/translate": 49.1, "ja/translate": 46.4, "hi/translate": 58.1, "ar/translate": 59.0},
        "note": "dense 27B; built-in MTP head drafts 2 tokens",
    },
    "gemma4-26b-a4b-qat": {
        "repo": "unsloth/gemma-4-26B-A4B-it-qat-GGUF", "file": "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf", "gb": 13.3, "bpw": 4.2, "qat": True,
        "kv_kb_per_token": 10.9, "fixed_cache_gb": 0.11, "checkpoint_gb": 0.11, "max_ctx": 262144, "tok_s_9070xt": 90, "mtp": False, "vk_fix": False,
        "mmproj": "mmproj-F16.gguf", "mmproj_gb": 1.11,
        "moe": {"layers": 30, "expert_gb_per_layer": 0.4},  # from the GGUF: 11.96 GiB of experts over 30 layers
        "tok_s_offload": {8: 45, 13: 36, 18: 31},          # measured: layers' experts in RAM -> decode tok/s
        "scores": {"en/global": 82.2, "zh/global": 73.5, "zh/regional": 66.5, "es/global": 74.5, "es/regional": 75.2, "hi/global": 69.5, "hi/regional": 71.0, "ar/global": 71.5, "ar/regional": 73.6, "ja/global": 74.5, "ja/regional": 81.9, "th/regional": 65.7,
                   "en/math": 96.8, "th/math": 89.6, "zh/math": 88.8,
                   "th/translate": 55.6, "zh/translate": 47.7, "ja/translate": 48.0, "hi/translate": 62.8, "ar/translate": 61.9},
        "note": "MoE with ~4B active params: fastest",
    },
    "qwen3.8-27b-iq2": {
        "repo": "unsloth/Qwen3.8-27B-GGUF", "file": "Qwen3.8-27B-UD-IQ2_S.gguf", "gb": 7.8, "bpw": 2.5, "qat": False,
        "kv_kb_per_token": 34.8, "fixed_cache_gb": 0.15, "checkpoint_gb": 0.15, "max_ctx": 262144, "tok_s_9070xt": 40, "mtp": False, "vk_fix": True,
        "mmproj": "mmproj-F16.gguf", "mmproj_gb": 0.86,
        "scores": {"en/global": 74.2, "zh/global": 67.8, "zh/regional": 67.8, "es/global": 70.8, "es/regional": 69.2, "hi/global": 56.2, "hi/regional": 55.5, "ar/global": 60.8, "ar/regional": 57.2, "ja/global": 65.8, "ja/regional": 77.9, "th/regional": 54.2},
        "note": "for 10-12 GB cards only: 2-bit costs 8-13 points, most in Hindi, Arabic, Thai",
    },
}


TASK_SUITES = {"general": ("global", "regional"), "math": ("math",), "translate": ("translate",)}


def score(key: str, lang: str | None = None, task: str = "general") -> float:
    """Mean accuracy over the user's language tests for this task if we have them, else over every language."""
    suites = TASK_SUITES.get(task, TASK_SUITES["general"])
    s = {k: v for k, v in MODELS[key]["scores"].items() if k.split("/")[1] in suites}
    if not s and task != "general":
        return score(key, lang)
    mine = [v for k, v in s.items() if lang and k.startswith(lang + "/")]
    pool = mine or list(s.values())
    return sum(pool) / len(pool) if pool else 0.0


MIN_CTX = 8192   # a model only "fits" if it also leaves room for an 8k-token conversation


def fits(key: str, vram_gb: float) -> bool:
    from .sizing import context_tokens
    return context_tokens(vram_gb, MODELS[key]) >= MIN_CTX


def cpu_moe_layers(key: str, vram_gb: float, ram_free_gb: float) -> int | None:
    """MoE models that don't fit the card whole: how many layers' experts to keep in RAM (llama.cpp --n-cpu-moe)
    so the rest plus an 8k context fit on the GPU. 0 = fits whole; None = not possible on this PC."""
    from .sizing import context_tokens
    m = MODELS[key]
    if fits(key, vram_gb):
        return 0
    if "moe" not in m:
        return None
    per = m["moe"]["expert_gb_per_layer"]
    for n in range(1, m["moe"]["layers"] + 1):
        if n * per > 0.7 * ram_free_gb:
            return None
        if context_tokens(vram_gb, {**m, "gb": m["gb"] - n * per}) >= MIN_CTX:
            return n
    return None


QUALITY_FLOOR_BPW = 2.7   # below ~UD-Q2_K_XL-class quants accuracy falls off a cliff (non-English most: -13 pts at 2.5 bpw)


def below_floor(key: str) -> bool:
    """Post-training quants below the floor; vendor QAT checkpoints are trained for their bit-width and exempt."""
    m = MODELS[key]
    return m.get("bpw", 99) < QUALITY_FLOOR_BPW and not m.get("qat")


TIE_POINTS = 2.0   # accuracy gaps this small are inside the benchmark's margin: prefer the faster model


OFFLOAD_SPEED = 0.5   # placeholder share of full-GPU speed with experts in RAM; replaced by measurements per model


def speed(key: str, vram_gb: float, ram_free_gb: float = 0.0) -> float:
    """Expected decode tok/s on this card: measured speed, scaled down when experts have to stay in RAM."""
    m = MODELS[key]
    n = cpu_moe_layers(key, vram_gb, ram_free_gb) or 0
    if not n:
        return m["tok_s_9070xt"]
    measured = m.get("tok_s_offload", {})          # {layers in RAM: tok/s} measured on the reference card
    if measured:
        nearest = min(measured, key=lambda k: abs(int(k) - n))
        return measured[nearest]
    return m["tok_s_9070xt"] * OFFLOAD_SPEED


def pick(vram_gb: float, lang: str | None = None, ram_free_gb: float = 0.0, candidates=None,
         task: str = "general") -> str | None:
    """Most accurate model (for `lang` when measured) that runs on this PC - whole on the GPU, or a MoE with some
    experts in RAM - with an 8k context; near-ties go to the faster one."""
    ok = [k for k in (candidates or MODELS) if cpu_moe_layers(k, vram_gb, ram_free_gb) is not None]
    if not ok:
        return None
    best = max(score(k, lang, task) for k in ok)
    near = [k for k in ok if score(k, lang, task) >= best - TIE_POINTS]
    # on a tie, a vendor QAT checkpoint beats a post-training quant of similar accuracy, then the faster one wins
    return max(near, key=lambda k: (bool(MODELS[k].get("qat")), speed(k, vram_gb, ram_free_gb)))
