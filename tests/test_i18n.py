"""#108: the chat page and the main CLI lines in 7 languages; no missing or unused strings; Arabic right to left."""
import json
import re
import string
import urllib.request
from pathlib import Path

import pytest

from localllm import gateway, i18n, workspace

ROOT = Path(__file__).resolve().parents[1]
PAGE = (ROOT / "localllm" / "ui.html").read_text(encoding="utf-8")
CLI = (ROOT / "localllm" / "cli.py").read_text(encoding="utf-8")
DYNAMIC = ("fit_", "suite_")          # the page builds these names: T("fit_" + m.fit), L["suite_" + s]


def used_keys() -> set[str]:
    page = set(re.findall(r'data-i18n(?:-[a-z]+)?="(\w+)"', PAGE)) | set(re.findall(r'\bT\("(\w+)"', PAGE))
    found = page | set(re.findall(r'i18n\.t\("(\w+)"', CLI))
    return {k for k in found if not k.endswith("_")}          # "fit_" + ...: see DYNAMIC


def fields(text: str) -> set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(text) if f}


def test_every_used_key_exists_and_every_key_is_used():
    keys = set(i18n.STRINGS)
    assert used_keys() - keys == set(), "used but missing in i18n.STRINGS"
    unused = {k for k in keys - used_keys() if not (k.startswith(DYNAMIC) and any(f'"{p}" +' in PAGE for p in DYNAMIC))}
    assert unused == set(), "in i18n.STRINGS but used nowhere"
    assert all(f'"{p}" + ' in PAGE for p in DYNAMIC)


def test_translations_are_complete_and_keep_their_placeholders():
    for key, entry in i18n.STRINGS.items():
        assert set(entry) <= set(i18n.LANGS) and entry.get("en"), key
        for lang, text in entry.items():
            assert fields(text) == fields(entry["en"]), (key, lang)        # {gb}, {lang}, ... in every language
    missing = {lang: [k for k, e in i18n.STRINGS.items() if lang not in e] for lang in i18n.LANGS}
    assert all(v == [] for v in missing.values()), missing


def test_missing_translation_falls_back_to_english(monkeypatch):
    monkeypatch.setitem(i18n.STRINGS, "only_en", {"en": "only in English {x}"})
    assert i18n.t("only_en", "th", x=1) == "only in English 1"
    assert i18n.table("th")["strings"]["only_en"] == "only in English {x}"
    assert i18n.table("xx")["strings"]["send"] == "Send"


def test_language_choice_order(monkeypatch):
    monkeypatch.setattr("localllm.bench.system_language", lambda: "ja")
    monkeypatch.delenv("LOCALLLM_LANG", raising=False)
    assert i18n.ui_lang() == "ja" and i18n.ui_lang("th-TH") == "th" and i18n.ui_lang("de-DE") == "ja"
    monkeypatch.setenv("LOCALLLM_LANG", "ar")
    assert i18n.ui_lang("th-TH") == "ar"                                    # --lang wins
    monkeypatch.setattr("localllm.bench.system_language", lambda: "de")
    monkeypatch.delenv("LOCALLLM_LANG")
    assert i18n.ui_lang() == "en" and i18n.pick("zh_CN") == "zh"


def test_page_gets_its_strings_and_arabic_is_rtl(tmp_path, monkeypatch):
    from test_gateway import _up
    monkeypatch.delenv("LOCALLLM_LANG", raising=False)
    ui = {"sessions": workspace.Sessions(tmp_path / "s"), "catalog": workspace.catalog_info([], "en", False)}
    s, up = _up()
    g = gateway.serve(up, port=0, model_name="m", api_key="k", trust_loopback=False, ui=ui)
    base = f"http://127.0.0.1:{g.server_address[1]}/localllm/i18n"
    ar = json.load(urllib.request.urlopen(base + "?lang=ar-EG", timeout=10))       # no key needed, like the page
    assert ar["lang"] == "ar" and ar["dir"] == "rtl" and ar["strings"]["send"] == i18n.STRINGS["send"]["ar"]
    th = json.load(urllib.request.urlopen(base + "?lang=th", timeout=10))
    assert th["dir"] == "ltr" and th["strings"]["new_chat"] == "แชทใหม่" and not any(k.startswith("cli_") for k in th["strings"])
    g.shutdown(); s.shutdown()
    assert "document.documentElement.dir" in PAGE and "[dir=rtl] aside" in PAGE and "border-inline-end" in PAGE


@pytest.mark.parametrize("lang", i18n.LANGS)
def test_cli_lines_render_in_every_language(lang):
    assert "C:\\x" in i18n.t("cli_confirm_download", lang, key="m", gb=1.0, dir="C:\\x")
    assert "http://127.0.0.1:8080" in i18n.t("cli_ready", lang, url="http://127.0.0.1:8080")
