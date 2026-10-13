"""#93: the long-context needle suite builds deterministic prompts in each language and scores replies strictly."""
from localllm import bench

FILLER = [f"Sentence number {i} about nothing in particular." for i in range(50)]


def test_items_cover_every_length_and_depth():
    items = bench.longctx_items("th", FILLER)
    assert len(items) == 4 * len(bench.LONGCTX_LENGTHS)
    assert {it["tokens"] for it in items} == set(bench.LONGCTX_LENGTHS)
    assert items == bench.longctx_items("th", FILLER)                     # same seed, same test
    assert all(len(it["nums"]) == len(it["depths"]) == len(it["names"]) for it in items)


def test_needles_sit_at_their_depth_and_the_question_comes_last():
    it = {**bench.longctx_items("en", FILLER)[0], "depths": [0.5], "names": ["Orchid"], "nums": [12345]}
    text = bench.needle_prompt(it, 100)
    doc, q = text.rsplit("\n\n", 1)
    pos = doc.index("The secret number of Orchid is 12345.")
    assert 0.4 < pos / len(doc) < 0.6
    assert q == "What is the secret number of Orchid? Reply with the number only."
    assert doc.count("Sentence number") == 100


def test_every_language_has_a_needle_and_both_questions():
    for lang, (needle, one, many) in bench.NEEDLES.items():
        assert "{num}" in needle and "{name}" in needle and "{name}" in one and "{name}" in many, lang
        assert lang in bench.FLORES or lang == "en"
        assert bench.available(lang, ("longctx",)) == ["longctx"]


def test_reply_must_hold_every_number():
    assert bench.needles_found("The numbers are 12,345, 67890 and 11111.", [12345, 67890, 11111])
    assert not bench.needles_found("12345 and 67890", [12345, 67890, 11111])
    assert bench.needles_found("الرقم هو ١٢٣٤٥", [12345])               # Arabic-Indic digits count
    assert not bench.needles_found("1234", [12345])


def test_three_needle_questions_ask_for_numbers_not_digits():
    """'three digits' (Thai ตัวเลขสามตัว, Chinese 三个数字) made gemma-4 answer 3-digit prefixes of the needles."""
    for lang, (needle, one, many) in bench.NEEDLES.items():
        for word in ("ตัวเลข", "数字", "أرقام الثلاثة", "अंक"):
            assert word not in one + many, (lang, word)
    assert bench.NEEDLES["th"][0].endswith(" ")
