"""Speculative decoding with a small draft model, per language (#50): decode tok/s and acceptance rate.

  python tools/draft_bench.py --model gemma4-26b-a4b-qat --draft PATH/gemma-4-E2B-it-UD-Q4_K_XL.gguf

Starts the catalog model with localllm's launch flags, once without a draft and then once per draft setting
(--spec-draft-n-max x --spec-draft-p-min), and asks the same generation prompts in each language at temperature 0.
Records decode tok/s, drafted and accepted tokens (llama-server's timings), and the server's VRAM, so the draft's memory
cost is in the same row. One JSON line per setting in tools/draft_results.jsonl. A setting is worth shipping in
`localllm tune` only where it is >= 1.1x faster in the user's language, not only in English.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from localllm import catalog, cli, runtime  # noqa: E402

PORT = 18150
PROMPTS = {
    "en": ["Explain how a refrigerator keeps food cold, in about 150 words.",
           "Write a short story about a lighthouse keeper who finds a message in a bottle."],
    "th": ["อธิบายว่าตู้เย็นทำให้อาหารเย็นได้อย่างไร ประมาณ 150 คำ",
           "เขียนเรื่องสั้นเกี่ยวกับคนเฝ้าประภาคารที่พบข้อความในขวด"],
    "zh": ["用大约150字解释冰箱是如何让食物保持低温的。", "写一个关于灯塔看守人在瓶子里发现一封信的短篇故事。"],
    "hi": ["लगभग 150 शब्दों में समझाइए कि रेफ्रिजरेटर भोजन को ठंडा कैसे रखता है।",
           "एक प्रकाशस्तंभ रक्षक के बारे में एक छोटी कहानी लिखिए जिसे बोतल में एक संदेश मिलता है।"],
    "ar": ["اشرح في حوالي 150 كلمة كيف تحافظ الثلاجة على برودة الطعام.",
           "اكتب قصة قصيرة عن حارس منارة يجد رسالة في زجاجة."],
    "ja": ["冷蔵庫が食べ物を冷たく保つ仕組みを150語程度で説明してください。",
           "瓶の中の手紙を見つけた灯台守についての短い物語を書いてください。"],
}


def start(args: list[str], env: dict) -> subprocess.Popen:
    server = runtime.find_server()
    log = open(Path(__file__).with_name("draft_server.log"), "ab")
    p = subprocess.Popen([str(server), *args], env=env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(1200):
        if p.poll() is not None:
            raise RuntimeError("llama-server exited, see draft_server.log")
        try:
            if b'"ok"' in urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).read():
                return p
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("did not load")


def measure(langs: list[str]) -> dict:
    out = {}
    for lang in langs:
        speeds, drafted, accepted = [], 0, 0
        for prompt in ["Warm up."] + PROMPTS[lang]:
            body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 256, "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False}}
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            t = json.load(urllib.request.urlopen(req, timeout=900)).get("timings", {})
            if prompt != "Warm up.":
                speeds.append(t.get("predicted_per_second", 0))
                drafted += t.get("draft_n", 0) or 0
                accepted += t.get("draft_n_accepted", 0) or 0
        out[lang] = {"tok_s": round(sum(speeds) / len(speeds), 1),
                     "accept": round(accepted / drafted, 3) if drafted else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="gemma4-26b-a4b-qat", choices=list(catalog.MODELS))
    ap.add_argument("--draft", type=Path, required=True)
    ap.add_argument("--n-max", default="2,4,6")
    ap.add_argument("--p-min", default="0,0.75")
    ap.add_argument("--langs", default="en,th,zh,hi,ar,ja")
    a = ap.parse_args()
    _server, _devs, dev, _ram = cli._machine()
    base = runtime.server_args(cli._model_path(a.model), dev["id"], PORT, 8192, False)
    env = runtime.server_env(None, catalog.MODELS[a.model].get("vk_fix", True))
    settings = [None] + [(int(n), float(p)) for n in a.n_max.split(",") for p in a.p_min.split(",")]
    base_speed = None
    for s in settings:
        extra = [] if s is None else ["-md", str(a.draft), "-ngld", "999", "-devd", dev["id"],
                                      "--spec-draft-n-max", str(s[0]), "--spec-draft-p-min", str(s[1])]
        try:
            p = start(base + extra, env)
        except RuntimeError as e:
            print(f"!! {s}: {e}", flush=True)
            continue
        try:
            mem = runtime.process_memory(p.pid)
            res = measure(a.langs.split(","))
        finally:
            p.terminate()
            p.wait(30)
        if s is None:
            base_speed = res
        rec = {"date": time.strftime("%Y-%m-%d"), "model": a.model, "draft": None if s is None else a.draft.name,
               "n_max": None if s is None else s[0], "p_min": None if s is None else s[1],
               "llama_cpp": runtime.build_of(runtime.find_server()), "vram_gb": mem.get("vram_gb"), "langs": res,
               "speedup": {l: round(v["tok_s"] / base_speed[l]["tok_s"], 2) for l, v in res.items()}
               if base_speed else None}
        with open(ROOT / "tools" / "draft_results.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{'no draft' if s is None else f'n_max {s[0]} p_min {s[1]}':22} " +
              "  ".join(f"{l} {v['tok_s']}" + (f" ({v['accept']:.0%})" if v["accept"] is not None else "")
                        for l, v in res.items()), flush=True)


if __name__ == "__main__":
    main()
