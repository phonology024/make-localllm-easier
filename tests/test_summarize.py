"""Summarization suite (#43): LR-Sum (CC BY 4.0), chrF++ and ROUGE-L, and the summarize task in routing."""
import json

from localllm import bench, catalog, router


def test_rouge_l_words_and_characters():
    assert bench.rouge_l("the cat sat", "the cat sat") == 100.0
    assert round(bench.rouge_l("the cat sat", "the cat sat on the mat"), 1) == 66.7
    assert bench.rouge_l("", "x") == 0.0 and bench.rouge_l("abc", "xyz") == 0.0
    assert round(bench.rouge_l("แมวนั่ง", "แมวนั่งบนเสื่อ", chars=True), 1) == 66.7   # Thai: no word spaces


def test_summarize_suite_is_available_where_lr_sum_has_the_language():
    assert bench.available("th", ("summarize",)) == ["summarize"]
    assert bench.available("en", ("summarize",)) == ["summarize"]
    assert bench.available("ja", ("summarize",)) == []          # not in LR-Sum
    assert "summarize" in bench.SUITES and catalog.TASK_SUITES["summarize"] == ("summarize",)


def test_run_scores_summaries(monkeypatch, tmp_path):
    monkeypatch.setattr(bench, "HOME", tmp_path)
    monkeypatch.setattr(bench, "load", lambda suite, lang, limit=0: [{"text": "t", "ref": "a b c d", "lang": "en"}] * 2)

    class R:
        def __init__(self, text):
            self.text = text

        def read(self):
            return json.dumps({"choices": [{"message": {"content": self.text}}]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass
    monkeypatch.setattr(bench.urllib.request, "urlopen", lambda req, timeout=0: R("a b c d"))
    res = bench.run("http://x", "m", ["en"], suites=("summarize",))
    assert res["en/summarize"]["rougeL"] == 100.0 and res["en/summarize"]["chrf"] == 100.0


def test_summarize_requests_route_as_summarize():
    keys = ["gemma4-26b-a4b-qat"]
    for text in ("Summarize the plot of Hamlet", "สรุปข่าวเศรษฐกิจวันนี้ให้หน่อย", "请总结这篇文章", "tl;dr this thread"):
        assert " summarize by keyword" in router.pick_local(text, keys, None, 16, 20)[1], text
    for text in ("resume the download", "What is 2+2 and 3*4?"):
        assert "summarize" not in router.pick_local(text, keys, None, 16, 20)[1], text
    assert router.detect_task("Summarize the plot of Hamlet") == "general"     # the classifier's labels are unchanged
