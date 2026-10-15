"""Human-written routing examples (#40): the issue form is parsed, checked and added without private or copied text."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "router"))
import add_human_examples as ah  # noqa: E402

FORM = """### Language

th

### Examples

```text
math | ถ้าซื้อของ 3 ชิ้น ชิ้นละ 45 บาท ลด 10% ต้องจ่ายเท่าไหร่
code | เขียน python อ่านไฟล์ csv แล้วหาค่าเฉลี่ยคอลัมน์ราคาให้หน่อย
translate | "ขอบคุณที่รอ" ภาษาญี่ปุ่นพูดว่าไง
general|แนะนำหนังสือสำหรับคนเริ่มลงทุนหน่อย
```

### Publishing

- [X] I wrote these myself, they contain nothing private, and they may be published in the repo (CC0).
"""


def test_form_is_parsed_and_accepted():
    lang, lines = ah.parse(FORM)
    assert lang == "th" and len(lines) == 4
    rows, refused, warnings = ah.check(lang, lines, set())
    assert refused == [] and warnings == [] and [r["label"] for r in rows] == ["math", "code", "translate", "general"]
    assert all(r["lang"] == "th" and r["source"] == "human" for r in rows)


def test_refusals_say_why():
    lines = ["sports | who won", "math | 9", "general | call me on 081 234 5678 please",
             "general | see https://example.com for details", "code | mail me at a@b.co the script",
             "no separator here", "general | ok"]
    rows, refused, _ = ah.check("en", lines, set())
    assert rows == []
    joined = "\n".join(refused)
    for why in ("label 'sports'", "a long number", "a link", "an e-mail address", "write `label | text`",
                "3-500 characters"):
        assert why in joined, why
    assert ah.check("thai", ["math | 2 + 2 = ?"], set())[1][0].startswith("language:")


def test_copies_of_the_ai_written_data_and_repeats_are_refused():
    shared = json.loads((ah.D / "shared_test.jsonl").read_text(encoding="utf-8").splitlines()[0])["text"]
    seen = ah.known()
    rows, refused, _ = ah.check("th", [f"general | {shared}  ", f"general | {shared.upper()}!",
                                       "math | ครึ่งหนึ่งของ 50 คือ", "math | ครึ่งหนึ่งของ 50 คือ?"], seen)
    assert len(rows) == 1 and len(refused) == 3 and all("already in" in r for r in refused)


def test_a_mismatched_script_is_a_warning_not_a_refusal():
    rows, refused, warnings = ah.check("th", ["general | what's a good laptop for students"], set())
    assert len(rows) == 1 and refused == [] and "looks like en" in warnings[0]
    rows, _, warnings = ah.check("es", ["general | ¿qué portátil me recomiendas para estudiar?"], set())
    assert len(rows) == 1 and warnings == []                                    # Latin-script languages: no guess


def test_check_mode_writes_nothing(tmp_path):
    body = tmp_path / "body.md"
    body.write_text(FORM, encoding="utf-8")
    before = ah.HUMAN.read_text(encoding="utf-8") if ah.HUMAN.exists() else None
    r = subprocess.run([sys.executable, str(ah.D / "add_human_examples.py"), str(body), "--check"],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0 and "4 example(s) accepted" in r.stdout and "math 1" in r.stdout
    assert (ah.HUMAN.read_text(encoding="utf-8") if ah.HUMAN.exists() else None) == before
