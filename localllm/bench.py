"""Multilingual multiple-choice benchmark for any OpenAI-compatible server (llama-server, Ollama, LM Studio, vLLM ...).

Two kinds of test per language, so a score means something wherever you live:
  global    Global-MMLU-Lite (CohereLabs, Apache-2.0): the same 400 questions translated into 23 languages,
            so languages and models compare like for like
  regional  INCLUDE-lite-44 (CohereLabs, Apache-2.0): real exams written in each country (~250 per language,
            44 languages); ThaiExam (typhoon-ai, Apache-2.0) fills in Thai
Zero-shot, thinking off, one token: the answer is the option letter with the highest log-probability. Fast
(prompt processing only) and deterministic. Data is downloaded at eval time and cached, never redistributed.

Task suites (opt-in with --suites, they generate text so they are slower):
  math      MGSM (Shi et al. 2022, CC-BY-SA-4.0): the same 250 grade-school word problems in 11 languages; the model
            reasons in text (thinking off) and the final number is compared exactly
  translate FLORES-101 devtest (Goyal et al. 2021, CC-BY-SA-4.0): the same sentences in 101 languages; the first 100
            are translated English -> language and language -> English, scored with chrF++ (0-100, higher is better)
  vision    MaXM (Changpinyo et al. 2023, CC-BY-4.0; Crossmodal-3600 images): questions about photos taken in each
            language's own region, in 7 languages incl. Thai; a short answer counts when it matches a reference after
            normalising case, punctuation and spaces (or contains it, for answers phrased as a short sentence).
            Needs a server started with the vision projector (`localllm serve --vision`)
"""
from __future__ import annotations

import ast
import base64
import json
import locale
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

from .runtime import HOME

ROWS = "https://datasets-server.huggingface.co/rows?dataset={ds}&config={cfg}&split={split}&offset={off}&length=100"
GLOBAL_LANGS = ["ar", "bn", "cs", "cy", "de", "en", "es", "fr", "hi", "hu", "id", "it", "ja", "ko", "my", "or", "pt",
                "sk", "sq", "sw", "tg", "yo", "zh"]
INCLUDE = {"sq": "Albanian", "ar": "Arabic", "hy": "Armenian", "az": "Azerbaijani", "eu": "Basque", "be": "Belarusian",
           "bn": "Bengali", "bg": "Bulgarian", "zh": "Chinese", "hr": "Croatian", "nl": "Dutch", "et": "Estonian",
           "fi": "Finnish", "fr": "French", "ka": "Georgian", "de": "German", "el": "Greek", "he": "Hebrew",
           "hi": "Hindi", "hu": "Hungarian", "id": "Indonesian", "it": "Italian", "ja": "Japanese", "kk": "Kazakh",
           "ko": "Korean", "lt": "Lithuanian", "ms": "Malay", "ml": "Malayalam", "ne": "Nepali",
           "mk": "North Macedonian", "fa": "Persian", "pl": "Polish", "pt": "Portuguese", "ru": "Russian",
           "sr": "Serbian", "es": "Spanish", "tl": "Tagalog", "ta": "Tamil", "te": "Telugu", "tr": "Turkish",
           "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek", "vi": "Vietnamese"}
THAIEXAM = "https://huggingface.co/datasets/typhoon-ai/thai_exam/resolve/main/data/{s}/{s}_test.jsonl"
SYSTEM = "Answer the multiple-choice question. Reply with only the letter of the correct option."
MGSM_LANGS = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
MATH_SYSTEM = "Solve the problem step by step, briefly. End with a last line of the form 'Answer: <number>'."
FLORES = {"af": "afr", "am": "amh", "ar": "ara", "bg": "bul", "bn": "ben", "ca": "cat", "cs": "ces", "cy": "cym",
          "da": "dan", "de": "deu", "el": "ell", "es": "spa", "et": "est", "fa": "fas", "fi": "fin", "fr": "fra",
          "gu": "guj", "he": "heb", "hi": "hin", "hr": "hrv", "hu": "hun", "hy": "hye", "id": "ind", "is": "isl",
          "it": "ita", "ja": "jpn", "jv": "jav", "ka": "kat", "kk": "kaz", "km": "khm", "kn": "kan", "ko": "kor",
          "lo": "lao", "lt": "lit", "lv": "lav", "mk": "mkd", "ml": "mal", "mn": "mon", "mr": "mar", "ms": "msa",
          "my": "mya", "ne": "npi", "nl": "nld", "no": "nob", "pa": "pan", "pl": "pol", "pt": "por", "ro": "ron",
          "ru": "rus", "sk": "slk", "sl": "slv", "sr": "srp", "sv": "swe", "sw": "swh", "ta": "tam", "te": "tel",
          "th": "tha", "tl": "tgl", "tr": "tur", "uk": "ukr", "ur": "urd", "uz": "uzb", "vi": "vie", "yo": "yor",
          "zh": "zho_simpl", "zu": "zul"}
TRANSLATE_N = 100
MAXM = {"en": "en", "fr": "fr", "hi": "hi", "he": "iw", "ro": "ro", "th": "th", "zh": "zh"}   # ISO -> dataset split
VISION_SYSTEM = "Look at the image and answer the question in {lang} with one word or a short phrase. Reply with the answer only."
SUITES = ("global", "regional", "math", "translate", "vision")


def system_language() -> str:
    try:
        loc = locale.getlocale()[0] or ""
    except ValueError:
        loc = ""
    if "_" in loc or len(loc) == 2:
        return loc.split("_")[0].lower()[:2]
    names = {n.lower(): c for c, n in INCLUDE.items()} | {"thai": "th", "english": "en"}
    return names.get(loc.split("_")[0].lower(), "en")


def available(lang: str, suites: tuple[str, ...] = ("global", "regional")) -> list[str]:
    have = {"global": lang in GLOBAL_LANGS, "regional": lang in INCLUDE or lang == "th", "math": lang in MGSM_LANGS,
            "translate": lang in FLORES and lang != "en", "vision": lang in MAXM}
    return [s for s in suites if have[s]]


def _get_json(url: str, tries: int = 6):
    """datasets-server answers 429/5xx under load: back off and retry rather than lose the run."""
    for i in range(tries):
        try:
            return json.load(urllib.request.urlopen(url, timeout=120))
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or i == tries - 1:
                raise
        except urllib.error.URLError:
            if i == tries - 1:
                raise
        time.sleep(min(60, 2 ** (i + 1)))


def _rows(ds: str, cfg: str, split: str = "test", page: int = 100, limit: int = 0) -> list[dict]:
    """All rows of a split, or the first `limit`."""
    out, off = [], 0
    while True:
        u = ROWS.format(ds=urllib.parse.quote(ds), cfg=urllib.parse.quote(cfg), split=split, off=off)
        n = min(page, limit - len(out)) if limit else page
        d = _get_json(u.replace("length=100", f"length={n}"))
        out += [r["row"] for r in d["rows"]]
        off += n
        if off >= d.get("num_rows_total", 0) or not d["rows"] or (limit and len(out) >= limit):
            return out


def load(suite: str, lang: str, limit: int = 0) -> list[dict]:
    """[{'q': question, 'opts': [..], 'ans': index}] cached under ~/.localllm/bench/. `limit` is honoured for the
    vision suite only, whose rows carry photos (a whole split is ~100 MB through the rows API)."""
    limit = limit if suite == "vision" else 0
    cache = HOME / "bench" / (f"{suite}-{lang}-first{limit}.json" if limit else f"{suite}-{lang}.json")
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    items = []
    if suite == "global":
        for r in _rows("CohereLabs/Global-MMLU-Lite", lang):
            opts = [r[f"option_{c}"] for c in "abcd"]
            items.append({"q": r["question"], "opts": opts, "ans": "ABCD".index(r["answer"].strip().upper())})
    elif suite == "regional" and lang == "th":
        for s in ["onet", "ic", "tgat", "tpat1", "a_level"]:
            for line in urllib.request.urlopen(THAIEXAM.format(s=s)).read().decode("utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    keys = [c for c in "abcde" if r.get(c)]
                    items.append({"q": r["question"], "opts": [r[c] for c in keys],
                                  "ans": keys.index(r["answer"].strip().lower())})
    elif suite == "translate" and lang in FLORES:
        en = [r["sentence"] for r in _rows("gsarti/flores_101", "eng", "devtest")[:TRANSLATE_N]]
        xx = [r["sentence"] for r in _rows("gsarti/flores_101", FLORES[lang], "devtest")[:TRANSLATE_N]]
        items = [{"src": a, "ref": b, "to": lang} for a, b in zip(en, xx)] + \
                [{"src": b, "ref": a, "to": "en"} for a, b in zip(en, xx)]
    elif suite == "math" and lang in MGSM_LANGS:
        for r in _rows("juletxara/mgsm", lang):
            items.append({"q": r["question"], "ans": int(r["answer_number"])})
    elif suite == "vision" and lang in MAXM:
        img_dir = HOME / "bench" / "maxm"
        img_dir.mkdir(parents=True, exist_ok=True)
        # rows carry the photo, so small pages, and only as many as the run asks for
        for r in _rows("floschne/maxm", "default", MAXM[lang], page=10, limit=limit):
            path = img_dir / f"{r['image_id']}.jpg"
            if not path.exists():
                raw = (r.get("image") or {}).get("bytes")
                data = bytes(raw) if isinstance(raw, list) else \
                    urllib.request.urlopen(r["image_url"], timeout=120).read()     # cell truncated: fetch the photo
                path.write_bytes(data)
            refs = sorted({a for a in (r.get("answers") or []) + (r.get("processed_answers") or []) if a})
            items.append({"q": r["question"], "img": path.name, "refs": refs, "lang": lang})
    elif suite == "regional":
        for r in _rows("CohereLabs/include-lite-44", INCLUDE[lang]):
            opts = r["choices"] if isinstance(r["choices"], list) else ast.literal_eval(r["choices"])
            items.append({"q": r["question"], "opts": opts, "ans": int(r["answer"])})
    else:
        raise ValueError(f"no {suite} test for '{lang}'")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return items


def ask(url: str, item: dict) -> int | None:
    letters = "ABCDEFGHIJ"[: len(item["opts"])]
    opts = "\n".join(f"{l}. {o}" for l, o in zip(letters, item["opts"]))
    body = {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"{item['q']}\n\n{opts}\n\nAnswer ({'/'.join(letters)}):"}],
            "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=600))
    best, best_lp = None, -1e9
    for t in d["choices"][0]["logprobs"]["content"][0]["top_logprobs"]:
        tok = t["token"].strip().upper().strip(".()")
        if len(tok) == 1 and tok in letters and t["logprob"] > best_lp:
            best, best_lp = letters.index(tok), t["logprob"]
    return best


def final_number(text: str) -> float | None:
    """The answer of a worked solution: the number after the last 'Answer:', else the last number in the text."""
    import re
    tail = text.rsplit("Answer", 1)[-1] if "Answer" in text else text
    nums = re.findall(r"-?\d[\d,]*\.?\d*", tail.replace(" ", " ").replace("**", ""))
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def ask_math(url: str, item: dict) -> bool:
    body = {"messages": [{"role": "system", "content": MATH_SYSTEM}, {"role": "user", "content": item["q"]}],
            "max_tokens": 600, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    text = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"].get("content") or ""
    n = final_number(text)
    return n is not None and abs(n - item["ans"]) < 1e-6


def _ngrams(seq, n: int) -> dict:
    out = {}
    for i in range(len(seq) - n + 1):
        g = tuple(seq[i:i + n])
        out[g] = out.get(g, 0) + 1
    return out


PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def _words(sent: str) -> list[str]:
    """Words with one leading/trailing ASCII punctuation mark split off, exactly like sacreBLEU's chrF++."""
    out = []
    for w in sent.split():
        if len(w) > 1 and w[-1] in PUNCT:
            out += [w[:-1], w[-1]]
        elif len(w) > 1 and w[0] in PUNCT:
            out += [w[0], w[1:]]
        else:
            out.append(w)
    return out


def chrf(hyp: str, ref: str, char_order: int = 6, word_order: int = 2, beta: float = 2.0) -> float:
    """chrF++ (Popovic 2017; sacreBLEU's defaults): F-beta over character 1-6-grams (spaces removed) and word 1-2-grams,
    averaged over n-gram orders. 0-100. Works for scripts without spaces (Thai, Chinese, Japanese) via the characters."""
    hc, rc = hyp.replace(" ", ""), ref.replace(" ", "")
    hw, rw = _words(hyp), _words(ref)
    precs, recs = [], []
    for seq_h, seq_r, orders in ((hc, rc, char_order), (hw, rw, word_order)):
        for n in range(1, orders + 1):
            h, r = _ngrams(seq_h, n), _ngrams(seq_r, n)
            match = sum(min(c, r.get(g, 0)) for g, c in h.items())
            if h and r:
                precs.append(match / sum(h.values()))
                recs.append(match / sum(r.values()))
    if not precs:
        return 0.0
    p, r = sum(precs) / len(precs), sum(recs) / len(recs)
    return 0.0 if p + r == 0 else round(100 * (1 + beta ** 2) * p * r / (beta ** 2 * p + r), 2)


LANG_NAMES = {"en": "English", "zh": "Simplified Chinese"}


def ask_translate(url: str, item: dict) -> float:
    to = LANG_NAMES.get(item["to"]) or INCLUDE.get(item["to"]) or {"th": "Thai"}.get(item["to"], item["to"])
    body = {"messages": [{"role": "system", "content": f"Translate the user's text into {to}. Reply with the translation only."},
                         {"role": "user", "content": item["src"]}],
            "max_tokens": 400, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    text = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"].get("content") or ""
    return chrf(text.strip(), item["ref"])


def _norm(text: str) -> str:
    text = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in unicodedata.normalize("NFKC", text))
    return " ".join(text.lower().split())


def vqa_match(pred: str, refs: list[str]) -> bool:
    """Same as a reference after normalising, or a short reply that contains one ("it is red" for "red")."""
    p = _norm(pred)
    for r in map(_norm, refs):
        if r and (p == r or (r in p and len(p) <= len(r) + 20)):
            return True
    return False


def ask_vision(url: str, item: dict) -> bool:
    lang = LANG_NAMES.get(item["lang"]) or INCLUDE.get(item["lang"]) or \
        {"th": "Thai", "he": "Hebrew", "ro": "Romanian"}.get(item["lang"], item["lang"])
    img = base64.b64encode((HOME / "bench" / "maxm" / item["img"]).read_bytes()).decode()
    body = {"messages": [{"role": "system", "content": VISION_SYSTEM.format(lang=lang)},
                         {"role": "user", "content": [
                             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
                             {"type": "text", "text": item["q"]}]}],
            "max_tokens": 32, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    text = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"].get("content") or ""
    return vqa_match(text.strip(), item["refs"])


def _save(name: str, res: dict) -> None:
    out = HOME / "results.json"
    allres = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    allres[name] = {**allres.get(name, {}), **res}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(allres, ensure_ascii=False, indent=1), encoding="utf-8")


def run(url: str, name: str, langs: list[str], limit: int = 0, suites: tuple[str, ...] = ("global", "regional")) -> dict:
    """Scores are saved after every test, so an interrupted run keeps what it finished."""
    res, t0 = {}, time.time()
    for lang in langs:
        mine = available(lang, suites)
        if not mine:
            print(f"  {lang}: no {'/'.join(suites)} benchmark yet (contributions welcome)")
        for suite in mine:
            items = load(suite, lang, limit)
            items = items[:limit] if limit else items
            if suite == "translate":          # a chrF++ score, not a right/wrong count
                scores = [ask_translate(url, it) for it in items]
                acc = round(sum(scores) / len(scores), 1)
                res[f"{lang}/{suite}"] = {"acc": acc, "chrf": acc, "n": len(items)}
                print(f"  {lang:3} {suite:9} chrF++ {acc:5.1f}  (en<->{lang}, {len(items)} sentences)", flush=True)
                _save(name, {**res, "_meta": {"model": name, "seconds": round(time.time() - t0), "limit": limit}})
                continue
            check = {"math": ask_math, "vision": ask_vision}.get(suite)
            ok = sum(check(url, it) if check else ask(url, it) == it["ans"] for it in items)
            acc = round(100 * ok / len(items), 1)
            res[f"{lang}/{suite}"] = {"acc": acc, "correct": ok, "n": len(items)}
            margin = round(196 * (acc / 100 * (1 - acc / 100) / len(items)) ** 0.5, 1)
            print(f"  {lang:3} {suite:9} {acc:5.1f}%  ±{margin}  ({ok}/{len(items)})", flush=True)
            _save(name, {**res, "_meta": {"model": name, "seconds": round(time.time() - t0), "limit": limit}})
    print(f"saved to {HOME / 'results.json'}  (share it: open a PR adding your GPU's numbers)")
    return res
