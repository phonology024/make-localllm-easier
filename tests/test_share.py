"""`localllm report --share` (#118): an SVG card of the result, from the report only, escaped, offline."""
import copy
import re
from pathlib import Path

from localllm import catalog, cli, runtime, share, sizing

G = "gemma4-26b-a4b-qat"
REP = {"schema": 1, "localllm": "0.7.1", "date": "2026-10-15", "os": "Linux 6.8.0", "arch": "x86_64",
       "ram_gb": 64.0, "language": "de", "llama_cpp": "b11476", "gpu": "NVIDIA GeForce RTX 4090", "vram_gb": 24.0,
       "pick": G, "tune": {"gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf": {"b11476": {"env": {}, "mtp": 0, "tok_s": 140.0}}},
       "results": {G: {"de/global": {"acc": 78.0}, "de/translate": {"acc": 60.0}, "_meta": {}}}}


def texts(svg: str) -> list[str]:
    return re.findall(r"<text[^>]*>(.*?)</text>", svg)


def _ram(rep):
    m = catalog.MODELS[G]
    args = runtime.server_args(Path(m["file"]), None, 8080, catalog.MIN_CTX, m["mtp"], ram_total_gb=rep["ram_gb"])
    return sizing.ram_estimate_gb(m, args)["total_gb"]


def test_card_snapshot():
    svg = share.render(share.facts(REP))
    assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630"')
    assert texts(svg) == [
        "localllm", "NVIDIA GeForce RTX 4090", "24 GB VRAM · 64 GB RAM · Linux 6.8.0", f"runs {G}",
        "140 tok/s", "decode speed, measured here",
        "78.0%", "score in de (measured on this PC)",                 # translation (chrF++) is not averaged in
        f"{_ram(REP):.1f} GB", "system RAM used (est.)",
        "phonology024.github.io/make-localllm-easier/leaderboard/gpu/nvidia-geforce-rtx-4090/", "measured, not guessed"]


def test_without_tune_or_eval_it_says_so_instead_of_borrowing_numbers():
    rep = {**copy.deepcopy(REP), "tune": {}, "results": {}}
    t = texts(share.render(share.facts(rep)))
    assert "not tuned yet" in t and "run: localllm tune" in t                  # no speed from another GPU
    want = share._mean(catalog.MODELS[G]["scores"], "de")
    assert (f"{want:.1f}%" in t and "score in de (catalog score)" in t) if want is not None else "not measured in de" in t


def test_report_text_is_escaped_and_nothing_personal_is_added():
    rep = {**copy.deepcopy(REP), "gpu": '<script>alert(1)</script>"&', "os": "Windows 11 <b>"}
    svg = share.render(share.facts(rep))
    assert "<script>" not in svg and "&lt;script&gt;" in svg and "&lt;b&gt;" in svg
    assert "/leaderboard/gpu/script-alert-1-script/" in svg
    assert set(re.findall(r"[\w.-]+@[\w.-]+|[A-Za-z]:\\", svg)) == set()


def test_no_measured_model_and_cpu_only():
    t = texts(share.render(share.facts({"date": "2026-10-15", "gpu": None, "vram_gb": 0.0, "ram_gb": 16.0,
                                        "os": "Windows 11", "language": "th", "pick": None, "tune": {}, "results": {}})))
    assert "CPU only" in t and "no measured model fits this GPU yet" in t and t.count("-") == 2


def test_cli_writes_the_card_next_to_the_report(tmp_path, monkeypatch):
    from localllm import report
    monkeypatch.setattr(report, "collect", lambda: copy.deepcopy(REP))
    monkeypatch.setattr(report, "save", lambda rep: tmp_path / "report-2026-10-15.json")
    opened = []
    monkeypatch.setattr(cli.webbrowser, "open", opened.append)
    cli.build_parser().parse_args(["report", "--share"]).fn(cli.build_parser().parse_args(["report", "--share"]))
    card = tmp_path / "share-2026-10-15.svg"
    assert card.exists() and "RTX 4090" in card.read_text(encoding="utf-8") and opened == [card.as_uri()]
