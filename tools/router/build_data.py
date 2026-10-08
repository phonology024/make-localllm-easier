"""Training data for the localllm task router: labelled for free by where each text comes from.

  math       MGSM (11 languages, CC-BY-SA-4.0) + GSM8K train (MIT)
  code       MBPP (CC-BY-4.0) prompts, CRUXEval (MIT) "what does this return" questions, request phrasings in 12 languages
  translate  FLORES-101 sentences (CC-BY-SA-4.0) wrapped in translation requests written in 12 languages
  general    Aya (Apache-2.0) and OpenAssistant oasst2 (Apache-2.0) user prompts in many languages

Writes data/train.jsonl and data/heldout.jsonl ({"text", "label", "lang", "src"}). Split by source item, 85/15.
"""
import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).with_name("data")
ROWS = "https://datasets-server.huggingface.co/rows?dataset={ds}&config={cfg}&split={split}&offset={off}&length=100"
random.seed(7)


def rows(ds, cfg, split, limit=10**9, offset=0):
    cache = OUT / "cache" / f"{ds.replace('/', '_')}-{cfg}-{split}-{limit}-{offset}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    cache.parent.mkdir(parents=True, exist_ok=True)
    got, complete = _rows(ds, cfg, split, limit, offset)
    if complete:                              # a partial answer is used once but fetched again next time
        cache.write_text(json.dumps(got, ensure_ascii=False), encoding="utf-8")
    return got


def _get(u, tries=8):
    """One datasets-server page; anonymous requests get throttled (429), so back off up to a minute, Retry-After first."""
    for attempt in range(tries):
        try:
            return json.load(urllib.request.urlopen(u, timeout=120))
        except Exception as e:
            err = e
            after = getattr(e, "headers", None) and e.headers.get("Retry-After")
            wait = min(60, float(after) if after and after.isdigit() else 5 * 2 ** attempt)
            print(f"    {e} - retry in {wait:.0f}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"{u}: {err}")


def _rows(ds, cfg, split, limit, offset):
    out, off = [], offset
    while len(out) < limit:
        u = ROWS.format(ds=urllib.parse.quote(ds, safe=""), cfg=urllib.parse.quote(cfg), split=split, off=off)
        try:
            d = _get(u)
        except RuntimeError as e:
            print(f"  gave up on {ds} {cfg} {split} at row {off} ({e}): keeping {len(out)} rows", flush=True)
            return out, False
        out += [r["row"] for r in d["rows"]]
        off += 100
        if off >= d.get("num_rows_total", 0) or not d["rows"]:
            break
        time.sleep(0.3)                       # pace requests
    return out[:limit], True


# ---- templates (written per language; {x} = payload) ---------------------------------------------------------------
LANG_NAME = {"en": {"en": "English", "th": "Thai", "zh": "Chinese", "ja": "Japanese", "es": "Spanish", "fr": "French",
                    "de": "German", "hi": "Hindi", "ar": "Arabic", "ko": "Korean", "vi": "Vietnamese", "id": "Indonesian"},
             "th": {"en": "ภาษาอังกฤษ", "th": "ภาษาไทย", "zh": "ภาษาจีน", "ja": "ภาษาญี่ปุ่น", "fr": "ภาษาฝรั่งเศส"},
             "zh": {"en": "英文", "th": "泰语", "zh": "中文", "ja": "日语", "fr": "法语"},
             "ja": {"en": "英語", "th": "タイ語", "zh": "中国語", "ja": "日本語", "fr": "フランス語"}}
TR_TEMPLATES = {
    "en": ["Translate this into {L}: {x}", "Can you translate the following to {L}?\n{x}", "How do you say this in {L}? \"{x}\"",
           "Please translate to {L}:\n\n{x}", "{x}\n\nTranslate the above into {L}."],
    "th": ["ช่วยแปลเป็น{L}หน่อย: {x}", "แปลประโยคนี้เป็น{L}ให้หน่อยครับ\n{x}", "{x}\n\nแปลเป็น{L}ให้ที", "ประโยคนี้ภาษา{L}พูดว่ายังไง \"{x}\""],
    "zh": ["请把这段话翻译成{L}：{x}", "帮我翻译成{L}：\n{x}", "{x}\n\n请翻译成{L}。", "这句话用{L}怎么说？“{x}”"],
    "ja": ["次の文を{L}に翻訳してください：{x}", "{L}に訳して：\n{x}", "{x}\n\nこれを{L}に翻訳して。"],
    "es": ["Traduce esto al inglés: {x}", "¿Puedes traducir lo siguiente al inglés?\n{x}"],
    "fr": ["Traduis ceci en anglais : {x}", "Peux-tu traduire ce texte en anglais ?\n{x}"],
    "de": ["Übersetze das bitte ins Englische: {x}", "Kannst du das ins Englische übersetzen?\n{x}"],
    "hi": ["इसका अंग्रेज़ी में अनुवाद करें: {x}", "कृपया इसे अंग्रेज़ी में अनुवाद कीजिए:\n{x}"],
    "ar": ["ترجم هذا إلى الإنجليزية: {x}", "من فضلك ترجم النص التالي إلى الإنجليزية:\n{x}"],
    "ko": ["이 문장을 영어로 번역해 주세요: {x}", "영어로 번역해줘:\n{x}"],
    "vi": ["Dịch câu này sang tiếng Anh: {x}", "Hãy dịch đoạn sau sang tiếng Anh:\n{x}"],
    "id": ["Terjemahkan ini ke bahasa Inggris: {x}", "Tolong terjemahkan kalimat berikut ke bahasa Inggris:\n{x}"],
}
CODE_TEMPLATES = {
    "en": ["{x}", "Write a Python function: {x}", "Can you help me code this? {x}", "{x} Please include the code."],
    "th": ["ช่วยเขียนโค้ด Python: {x}", "เขียนฟังก์ชันให้หน่อย {x}", "โจทย์โค้ด: {x} ช่วยเขียนให้ที"],
    "zh": ["请用Python写一个函数：{x}", "帮我写代码：{x}"],
    "ja": ["Pythonで関数を書いてください：{x}", "次の処理をコードにして：{x}"],
    "es": ["Escribe una función en Python: {x}"], "fr": ["Écris une fonction Python : {x}"],
    "de": ["Schreib eine Python-Funktion: {x}"], "hi": ["एक Python फ़ंक्शन लिखिए: {x}"],
    "ar": ["اكتب دالة بايثون: {x}"], "ko": ["파이썬 함수를 작성해 주세요: {x}"], "vi": ["Viết một hàm Python: {x}"],
    "id": ["Tulis fungsi Python: {x}"],
}
CRUX_TEMPLATES = {"en": "```python\n{c}\n```\nWhat does f({i}) return?", "th": "```python\n{c}\n```\nf({i}) คืนค่าอะไร",
                  "zh": "```python\n{c}\n```\nf({i}) 返回什么？", "ja": "```python\n{c}\n```\nf({i}) の戻り値は？"}
FLORES = {"en": "eng", "th": "tha", "zh": "zho_simpl", "ja": "jpn", "es": "spa", "fr": "fra", "de": "deu", "hi": "hin",
          "ar": "ara", "ko": "kor", "vi": "vie", "id": "ind"}
AYA = {"tha": "th", "eng": "en", "zho": "zh", "jpn": "ja", "spa": "es", "fra": "fr", "deu": "de", "hin": "hi",
       "arb": "ar", "kor": "ko", "vie": "vi", "ind": "id", "por": "pt", "rus": "ru", "tur": "tr", "msa": "ms"}


def main():
    OUT.mkdir(exist_ok=True)
    items = []   # (group id, record)

    # math
    for lang in ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]:
        for i, r in enumerate(rows("juletxara/mgsm", lang, "test")):
            items.append((f"mgsm{i}", {"text": r["question"], "label": "math", "lang": lang, "src": "mgsm"}))
    for i, r in enumerate(rows("openai/gsm8k", "main", "train", 600)):
        items.append((f"gsm{i}", {"text": r["question"], "label": "math", "lang": "en", "src": "gsm8k"}))

    # code
    for i, r in enumerate(rows("google-research-datasets/mbpp", "full", "train") + rows("google-research-datasets/mbpp", "full", "test")):
        lang = random.choice(list(CODE_TEMPLATES))
        items.append((f"mbpp{i}", {"text": random.choice(CODE_TEMPLATES[lang]).format(x=r["text"]), "label": "code",
                                   "lang": lang, "src": "mbpp"}))
    for i, r in enumerate(rows("cruxeval-org/cruxeval", "default", "test", 500)):
        lang = random.choice(list(CRUX_TEMPLATES))
        items.append((f"crux{i}", {"text": CRUX_TEMPLATES[lang].format(c=r["code"], i=r["input"]), "label": "code",
                                   "lang": lang, "src": "cruxeval"}))

    # translate: sentence in language A, request written in language B (often A == B, or English)
    flores = {k: [r["sentence"] for r in rows("gsarti/flores_101", v, "dev", 400)] for k, v in FLORES.items()}
    for i in range(1100):
        req = random.choice(list(TR_TEMPLATES))
        src = random.choice([req, req, "en"] + list(FLORES))
        if src == req and req == "en":
            src = random.choice([k for k in FLORES if k != "en"])
        target = "en" if src != "en" else random.choice(["th", "zh", "ja", "fr"])
        names = LANG_NAME.get(req, LANG_NAME["en"])
        if target not in names:
            target = "en"
        text = random.choice(TR_TEMPLATES[req]).format(L=names[target], x=flores[src][i % len(flores[src])])
        items.append((f"flores{i}", {"text": text, "label": "translate", "lang": req, "src": "flores"}))

    # general: Aya (pages across the whole set) + oasst2 root prompts
    for off in random.sample(range(0, 200000, 100), 30):
        for i, r in enumerate(rows("CohereLabs/aya_dataset", "default", "train", 100, off)):
            if r["language_code"] in AYA:
                items.append((f"aya{off}_{i}", {"text": r["inputs"], "label": "general", "lang": AYA[r["language_code"]],
                                                "src": "aya"}))
    for off in random.sample(range(0, 128000, 100), 40):
        for i, r in enumerate(rows("OpenAssistant/oasst2", "default", "train", 100, off)):
            if r["role"] == "prompter" and r["parent_id"] is None and not r.get("deleted"):
                items.append((f"oa{off}_{i}", {"text": r["text"], "label": "general", "lang": r["lang"], "src": "oasst2"}))
    # knowledge questions (no options) in many languages: Global-MMLU-Lite (Apache-2.0), ThaiExam (Apache-2.0)
    for lang in ["ar", "bn", "de", "en", "es", "fr", "hi", "id", "it", "ja", "ko", "pt", "sw", "yo", "zh"]:
        for i, r in enumerate(random.sample(rows("CohereLabs/Global-MMLU-Lite", lang, "test"), 90)):
            items.append((f"gmmlu{lang}{i}", {"text": r["question"], "label": "general", "lang": lang, "src": "global-mmlu"}))
    for s in ["onet", "tgat", "a_level"]:
        url = f"https://huggingface.co/datasets/typhoon-ai/thai_exam/resolve/main/data/{s}/{s}_test.jsonl"
        for i, line in enumerate(urllib.request.urlopen(url).read().decode("utf-8").splitlines()):
            if line.strip():
                items.append((f"thx{s}{i}", {"text": json.loads(line)["question"], "label": "general", "lang": "th",
                                             "src": "thaiexam"}))

    # balance: math is plentiful, keep ~1300
    math = [it for it in items if it[1]["label"] == "math"]
    keep_math = set(id(x) for x in random.sample(math, min(1300, len(math))))
    items = [it for it in items if it[1]["label"] != "math" or id(it) in keep_math]

    random.shuffle(items)
    groups = sorted({g for g, _ in items})
    held = set(random.sample(groups, len(groups) * 15 // 100))
    for name, keep in (("train", lambda g: g not in held), ("heldout", lambda g: g in held)):
        recs = [r for g, r in items if keep(g)]
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        from collections import Counter
        print(name, len(recs), dict(Counter(r["label"] for r in recs)))


if __name__ == "__main__":
    main()
