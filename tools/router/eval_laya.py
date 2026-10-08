"""Laya (multilingual checkpoint, zero-shot choice) on the shared router test set. Run with the Laya app's venv."""
import json, time
from collections import defaultdict
from pathlib import Path

from laya import Router

D = Path(__file__).parent
test = [json.loads(l) for l in open(D / "shared_test.jsonl", encoding="utf-8")]
questions = {"task": {"type": "choice", "instructions": "What kind of task is the user's request? ประเภทของคำขอ",
                      "criteria": {
                          "general": "everyday questions, knowledge, advice, chat, creative or business writing; คำถามทั่วไป ความรู้ คำแนะนำ งานเขียน",
                          "math": "calculate or solve a math, arithmetic, percentage or word problem; คำนวณหรือแก้โจทย์คณิตศาสตร์",
                          "code": "write, fix, explain, review or run program code, SQL, scripts, CSS; เขียน แก้ หรืออธิบายโค้ด",
                          "translate": "translate text between languages or ask how to say something in another language; แปลภาษา"}}}
router = Router(default="multilingual", max_loaded=1)
router.predict({"request": "warm up"}, questions)
pred, lat = [], []
for r in test:
    t = time.perf_counter()
    out = router.predict({"request": r["text"]}, questions)
    lat.append(1000 * (time.perf_counter() - t))
    pred.append(out.get("answers", {}).get("task", {}).get("choice"))
acc = 100 * sum(p == r["label"] for p, r in zip(pred, test)) / len(test)
per = defaultdict(list)
for p, r in zip(pred, test):
    per[r["label"]].append(p == r["label"])
lat.sort()
print(f"Laya shared test: {acc:.1f}%  per class { {k: round(100 * sum(v) / len(v), 1) for k, v in per.items()} }")
print(f"latency median {lat[len(lat) // 2]:.0f} ms, p95 {lat[int(len(lat) * .95)]:.0f} ms")
(D / "data").mkdir(exist_ok=True)
json.dump(pred, open(D / "data" / "laya_pred.json", "w"))
