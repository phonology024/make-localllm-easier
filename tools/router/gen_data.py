"""Generate short, everyday-style requests per task class in many languages with the local LLM (gemma-4 via localllm).
The shared test set is never shown to the model; exact duplicates of it are dropped. Output: generated.jsonl (committed)
Needs a GPU that runs gemma-4 (not possible in CI). Model folders: $LOCALLLM_MODELS, else the maintainer's."""
import json, os, random, re, sys, time, urllib.request
from pathlib import Path

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D.parents[1]))          # this repo, so `localllm` imports without installing it
WIN_MODELS = [r"C:\QUANT_FLEET_MASTER\08_Local_Creative_Core\models",
              r"C:\QUANT_FLEET_MASTER\08_Local_Creative_Core\models\candidates"]
if not os.environ.get("LOCALLLM_MODELS") and Path(WIN_MODELS[0]).exists():
    os.environ["LOCALLLM_MODELS"] = os.pathsep.join(WIN_MODELS)
from localllm import cli, runtime  # noqa: E402

LANGS = {"th": "Thai", "en": "English", "zh": "Simplified Chinese", "ja": "Japanese", "es": "Spanish", "fr": "French",
         "de": "German", "hi": "Hindi", "ar": "Arabic", "ko": "Korean", "vi": "Vietnamese", "id": "Indonesian",
         "pt": "Portuguese", "ru": "Russian", "tr": "Turkish"}
KINDS = {
    "math": ["quick arithmetic or percentage questions people ask a chatbot", "short word problems about money, time, speed or shopping",
             "algebra, equations, derivatives or probability questions", "unit conversion or averages that need calculating"],
    "code": ["requests to write a small function or script in any programming language", "questions about an error message, often with a short code snippet",
             "how-to questions about CSS, SQL, git, Docker, regex or a framework", "requests to review, refactor, explain or test a short piece of code"],
    "translate": ["requests to translate a sentence into another language", "questions asking how to say a phrase in another language",
                  "requests to translate an email, message or song lyric", "questions asking what a foreign word or phrase means"],
    "general": ["everyday questions, advice and recommendations", "knowledge questions about history, science or culture, sometimes containing years or numbers that need no calculation",
                "requests for creative writing, emails, letters or summaries", "questions ABOUT programming languages, math or languages that only need an explanation, not code, a calculation or a translation"],
}
PROMPT = ("Write 12 different realistic messages that a user might send to an AI assistant, in {lang}. Type: {kind}. "
          "Vary length (3 to 40 words), tone and topic. Mix in English technical terms only where natives would. "
          "Output ONLY a JSON array of 12 strings.")


def main():
    test = {json.loads(l)["text"] for l in open(D / "shared_test.jsonl", encoding="utf-8")}
    server, _d, dev, _r = cli._machine()
    proc, url = cli._launch("gemma4-26b-a4b-qat", server, dev, 8192, runtime.ram_available_gb())
    out = open(D / "generated.jsonl", "w", encoding="utf-8")
    n = 0
    try:
        for code, lang in LANGS.items():
            for label, kinds in KINDS.items():
                for kind in kinds:
                    body = {"messages": [{"role": "user", "content": PROMPT.format(lang=lang, kind=kind)}],
                            "max_tokens": 1500, "temperature": 0.9, "chat_template_kwargs": {"enable_thinking": False}}
                    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                                 headers={"Content-Type": "application/json"})
                    try:
                        text = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"]
                        items = json.loads(re.search(r"\[.*\]", text, re.S).group(0))
                    except Exception as e:
                        print("skip", code, label, e); continue
                    for s in items:
                        if isinstance(s, str) and 3 < len(s) < 600 and s not in test:
                            out.write(json.dumps({"text": s, "label": label, "lang": code, "src": "gen-gemma4"},
                                                 ensure_ascii=False) + "\n"); n += 1
                out.flush()
            print(code, n, flush=True)
    finally:
        out.close(); proc.terminate(); proc.wait(30)
    print("total", n)


if __name__ == "__main__":
    main()
