"""Chat with your files (#113): reading, chunking, the index under ~/.localllm/docs, search, and the sources line.
The embedder here is a fake (hashed character trigrams), so these tests need no model; tests/test_docs_live.py and
`localllm eval --suites docqa` use the real e5 server."""
import array
import hashlib
import io
import json
import sys
import zipfile

import pytest

from localllm import docs, runtime


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "HOME", tmp_path)
    return tmp_path


class FakeEmbed:
    """Unit vectors from hashed character trigrams: texts that share words are close. Counts its calls."""
    def __init__(self, too_long: int | None = None):
        self.calls, self.too_long = 0, too_long

    def __call__(self, texts):
        out = []
        for t in texts:
            self.calls += 1
            body = t.split(": ", 1)[1] if t.startswith(("query: ", "passage: ")) else t
            if self.too_long and len(body) > self.too_long:
                raise docs.TooLong("input is too large")
            v = [0.0] * 64
            s = f"  {body.lower()}  "
            for i in range(len(s) - 2):
                v[int(hashlib.md5(s[i:i + 3].encode()).hexdigest(), 16) % 64] += 1
            out.append(docs._unit(v))
        return out


def _lib(embed=None):
    return docs.Library(embed=embed or FakeEmbed(), model_id="fake@1")


def _docx(paragraphs):
    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = "".join(f'<w:p><w:r><w:t xml:space="preserve">{p}</w:t></w:r></w:p>' for p in paragraphs)
    xml = f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="{w}"><w:body>{body}</w:body></w:document>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", xml)
    return buf.getvalue()


def test_normalize_fixes_what_pdfs_do_to_arabic_and_cjk():
    assert docs.normalize("ﺷﻤﺲ") == "شمس"                       # presentation forms -> letters
    assert docs.normalize("バッテリーは 購入日 から 24 か 月間保証") == "バッテリーは購入日から 24 か月間保証"
    assert docs.normalize("a \t b\r\nc") == "a b\nc"
    assert docs.normalize("แบตเตอรี\x00รับ") == "แบตเตอรีรับ"                  # pypdf's NUL for a dropped tone mark


def test_text_files_keep_the_line_each_paragraph_starts_on():
    units = docs.read_units("notes.md", "# Title\n\nFirst line\nsame paragraph\n\n\nThird\n".encode())
    assert units == [("# Title", "line 1"), ("First line same paragraph", "line 3"), ("Third", "line 7")]
    assert docs.read_units("bom.txt", "﻿hello".encode("utf-8"))[0][0] == "hello"
    assert docs.read_units("u16.txt", "สวัสดี".encode("utf-16"))[0][0] == "สวัสดี"


def test_docx_is_read_with_the_standard_library():
    units = docs.read_units("r.docx", _docx(["Warranty", "", "The battery is covered for 24 months."]))
    assert units == [("Warranty", "paragraph 1"), ("The battery is covered for 24 months.", "paragraph 2")]
    with pytest.raises(docs.DocError, match="not a readable .docx"):
        docs.read_units("r.docx", b"not a zip")


def test_unsupported_and_empty_files_say_why():
    with pytest.raises(docs.DocError, match="isn't supported"):
        docs.read_units("song.mp3", b"ID3")
    with pytest.raises(docs.DocError, match="no text"):
        docs.read_units("empty.txt", b"\n\n  \n")


def test_chunks_stay_under_the_limit_and_lose_no_text():
    long_en = " ".join(f"Sentence number {i} says something about batteries." for i in range(60))
    thai = "ก" * 1900                                                         # no spaces, no punctuation
    units = [(long_en, "p.1"), ("short one", "p.2"), (thai, "p.3")]
    chunks = docs.chunk(units, limit=300)
    assert all(len(c["text"]) <= 300 for c in chunks)
    joined = " ".join(c["text"] for c in chunks)
    assert all(f"number {i} " in joined for i in range(60)) and joined.count("ก") >= 1900
    assert {c["loc"] for c in chunks} >= {"p.1", "p.3"}


def test_overlap_repeats_a_short_last_sentence():
    text = " ".join(f"Fact {i} is stated here." for i in range(40))
    chunks = docs.chunk([(text, "line 1")], limit=200)
    assert len(chunks) >= 3
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt["text"].startswith(docs._last_sentence(prev["text"]))          # an answer at the cut is in both


def test_add_search_reuse_and_remove(home):
    emb = FakeEmbed()
    lib = _lib(emb)
    a = lib.add("warranty.txt", b"The battery is covered for 24 months.\n\nWater damage is not covered.")
    b = lib.add("recipe.md", b"Boil the pasta for nine minutes.\n\nAdd salt to the water.")
    assert a["chunks"] >= 1 and (home / "docs" / a["id"] / "vectors.f32").stat().st_size == a["chunks"] * 64 * 4
    hits = lib.search("how long is the battery covered", [a["id"], b["id"]], k=2)
    assert hits[0]["name"] == "warranty.txt" and hits[0]["n"] == 1 and "24 months" in hits[0]["text"]
    calls = emb.calls
    assert lib.add("renamed.txt", b"The battery is covered for 24 months.\n\nWater damage is not covered.")["id"] == a["id"]
    assert emb.calls == calls                                           # same bytes: not embedded again
    fresh = _lib(FakeEmbed())                                           # a new process reads the index from disk
    assert fresh.search("pasta minutes", [b["id"]], k=1)[0]["name"] == "recipe.md"
    assert [m["name"] for m in fresh.list()] == ["warranty.txt", "recipe.md"]
    assert fresh.remove(a["id"]) and not (home / "docs" / a["id"]).exists() and not fresh.remove(a["id"])
    with pytest.raises(docs.DocError):
        fresh.remove("../../etc")


def test_another_embedding_model_rebuilds_the_index():
    a = _lib().add("w.txt", b"hello world")
    other = docs.Library(embed=FakeEmbed(), model_id="other@2")
    assert other.get(a["id"]) is None and other.list() == []
    assert other.add("w.txt", b"hello world")["embed_model"] == "other@2"


def test_a_passage_the_server_refuses_is_split_until_it_fits():
    text = " ".join(f"word{i}" for i in range(120))                    # ~800 chars, one paragraph
    meta = _lib(FakeEmbed(too_long=250)).add("long.txt", text.encode())
    chunks = [json.loads(l) for l in (runtime.HOME / "docs" / meta["id"] / "chunks.jsonl").read_text("utf-8").splitlines()]
    assert all(len(c["text"]) <= 250 for c in chunks) and " ".join(c["text"] for c in chunks).split() == text.split()


def test_folders_add_every_supported_file(tmp_path):
    folder = tmp_path / "f"
    folder.mkdir()
    (folder / "a.txt").write_text("alpha")
    (folder / "b.md").write_text("beta")
    (folder / "c.exe").write_bytes(b"MZ")
    assert [m["name"] for m in _lib().add_path(folder)] == ["a.txt", "b.md"]
    with pytest.raises(docs.DocError, match="no such file"):
        _lib().add_path(tmp_path / "missing.txt")


def test_the_message_and_the_sources_line():
    hits = [{"n": 1, "name": "w.pdf", "loc": "p.3", "text": "Covered 24 months."},
            {"n": 2, "name": "n.md", "loc": "line 40", "text": "Not water."}]
    msg = docs.ask_with("How long?", hits)
    assert msg.startswith("Passages from the user's files") and "[1] (w.pdf, p.3)\nCovered 24 months." in msg
    assert msg.endswith("Question: How long?") and "say so plainly" in msg
    assert docs.sources_line(hits) == "Sources: [1] w.pdf p.3  ·  [2] n.md line 40"
    assert docs.ask_with("Hi", []) == "Hi" and docs.sources_line([]) == ""


def test_pdf_support_is_asked_for_never_downloaded_silently(monkeypatch):
    monkeypatch.setitem(sys.modules, "pypdf", None)                    # as if pypdf isn't installed
    assert not docs.pdf_support_ready()
    with pytest.raises(docs.NeedsPdfSupport, match="PDF support needs pypdf 6.19.0"):
        docs.read_units("a.pdf", b"%PDF-1.4")


def test_a_wheel_that_does_not_match_its_hash_is_deleted(monkeypatch):
    monkeypatch.setitem(sys.modules, "pypdf", None)
    monkeypatch.setattr(docs, "_wheels", lambda: [docs.PYPDF])
    lib = runtime.HOME / "lib"
    lib.mkdir()
    (lib / docs.PYPDF[0]).write_bytes(b"tampered")
    with pytest.raises(docs.DocError, match="did not match its pinned SHA-256"):
        docs.load_pypdf(allow_download=False)
    assert not (lib / docs.PYPDF[0]).exists()


def test_pinned_wheels_are_exact_versions_with_hashes():
    for name, url, sha in (docs.PYPDF, docs.TYPING_EXTENSIONS):
        assert url.endswith("/" + name) and len(sha) == 64 and int(sha, 16) >= 0
    assert "6.19.0" in docs.PYPDF[0]


def test_a_pdf_chunk_never_spans_two_pages():
    units = [("Warranty terms.", "p.1"), ("The battery is covered for 24 months.", "p.1"), ("ภาษาไทย แบตเตอรี่", "p.2")]
    assert [(c["loc"], c["text"]) for c in docs.chunk(units)] == [
        ("p.1", "Warranty terms. The battery is covered for 24 months."), ("p.2", "ภาษาไทย แบตเตอรี่")]


def test_attach_list_detach_and_the_question_sent(tmp_path):
    (tmp_path / "w.txt").write_text("The battery is covered for 24 months.\n\nWater damage is not covered.")
    said = []
    att = docs.Attached(_lib(), say=said.append)
    att.attach(f'"{tmp_path / "w.txt"}"')                                     # quotes from drag-and-drop
    assert "attached w.txt" in said[-1] and "1. w.txt" in att.listing()
    att.attach(str(tmp_path / "w.txt"))
    assert len(att.files) == 1                                                  # same file twice: once
    sent, hits = att.question("how long is the battery covered")
    assert hits and sent.startswith("Passages from the user's files") and sent.endswith("battery covered")
    att.attach(str(tmp_path / "nope.txt"))
    assert said[-1].startswith("(not attached: no such file")
    att.detach("5")
    assert "/detach N" in said[-1]
    att.detach("1")
    assert said[-1] == "(detached w.txt)" and att.question("x") == ("x", [])


def test_a_pdf_is_attached_only_after_a_yes(tmp_path, monkeypatch):
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(docs, "load_pypdf", lambda allow: (_ for _ in ()).throw(
        docs.NeedsPdfSupport(docs.PDF_SUPPORT) if not allow else docs.DocError("pretend: downloaded, then unreadable")))
    said, asked = [], []
    docs.Attached(_lib(), ask=lambda q: asked.append(q) or False, say=said.append).attach(str(tmp_path / "a.pdf"))
    assert "pypdf 6.19.0" in asked[0] and said[-1] == "(not attached: PDF support was not downloaded)"
    docs.Attached(_lib(), ask=lambda q: True, say=said.append).attach(str(tmp_path / "a.pdf"))
    assert said[-1] == "(not attached: pretend: downloaded, then unreadable)"


def test_chat_sends_the_passages_but_keeps_only_the_question(tmp_path, monkeypatch, capsys):
    from localllm import chat
    (tmp_path / "w.txt").write_text("The battery is covered for 24 months.")
    typed = iter([f"/attach {tmp_path / 'w.txt'}", "how long is the battery covered", "and water?", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(typed))
    sent = []

    def fake_stream(url, messages, show_thinking, out=None, stay=False):
        sent.append([dict(m) for m in messages])
        return "24 months [1]", {}
    monkeypatch.setattr(chat, "stream", fake_stream)
    chat.repl("http://x", files=docs.Attached(_lib()))
    first, second = sent
    assert first[-1]["content"].startswith("Passages from the user's files")
    assert second[0] == {"role": "user", "content": "how long is the battery covered"}   # history: question only
    assert second[-1]["content"].endswith("Question: and water?")
    assert capsys.readouterr().out.count("Sources: [1] w.txt line 1") == 2


def _http(url, method="GET", data=None, page=True, ctype="application/json"):
    import urllib.request
    h = {"Content-Type": ctype, **({"X-Localllm-UI": "1"} if page else {})}
    body = json.dumps(data).encode() if isinstance(data, (dict, list)) else data
    return urllib.request.urlopen(urllib.request.Request(url, data=body, method=method, headers=h), timeout=30)


def test_the_page_uploads_searches_and_forgets_files():
    import urllib.error
    from localllm import gateway
    from test_gateway import _up
    s, up = _up()
    lib = _lib()
    g = gateway.serve(up, port=0, model_name="m", ui={"docs": lib})
    base = f"http://127.0.0.1:{g.server_address[1]}/localllm/docs"
    try:
        meta = json.load(_http(base + "?name=w%C3%A4rranty.txt", "POST", "The battery is covered for 24 months.".encode(),
                               ctype="application/octet-stream"))
        assert meta["name"] == "wärranty.txt" and json.load(_http(base))["files"][0]["id"] == meta["id"]
        r = json.load(_http(base + "/search", "POST", {"q": "battery months", "ids": [meta["id"]]}))
        assert r["hits"][0]["loc"] == "line 1" and r["sources"] == "Sources: [1] wärranty.txt line 1"
        assert r["message"].endswith("Question: battery months")
        for method, path, data in (("POST", "/search", {"q": "x", "ids": [meta["id"]]}), ("DELETE", "/" + meta["id"], None),
                                   ("POST", "?name=a.txt", b"x")):
            with pytest.raises(urllib.error.HTTPError) as e:                       # another website: refused
                _http(base + path, method, data, page=False)
            assert e.value.code == 403
        with pytest.raises(urllib.error.HTTPError) as e:
            _http(base + "?name=a.mp3", "POST", b"ID3", ctype="application/octet-stream")
        assert e.value.code == 400 and "isn't supported" in json.load(e.value)["error"]
        assert _http(base + "/" + meta["id"], "DELETE").status == 200 and json.load(_http(base))["files"] == []
    finally:
        g.shutdown(); s.shutdown()


def test_the_page_is_asked_before_the_pdf_download(monkeypatch):
    import urllib.error
    from localllm import gateway
    from test_gateway import _up
    monkeypatch.setitem(sys.modules, "pypdf", None)
    s, up = _up()
    g = gateway.serve(up, port=0, model_name="m", ui={"docs": _lib()})
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            _http(f"http://127.0.0.1:{g.server_address[1]}/localllm/docs?name=a.pdf", "POST", b"%PDF-1.4",
                  ctype="application/pdf")
        assert e.value.code == 409 and json.load(e.value)["need"] == "pdf"
        assert not (runtime.HOME / "lib").exists()                                # nothing downloaded
    finally:
        g.shutdown(); s.shutdown()


def test_the_pdf_extra_pins_the_same_pypdf_the_download_uses():
    import re
    from pathlib import Path
    tomllib = pytest.importorskip("tomllib")
    pp = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert pp["dependencies"] == []                                             # the core stays dependency-free
    pinned = re.search(r"pypdf-([\d.]+)-", docs.PYPDF[0])[1]
    assert pp["optional-dependencies"]["pdf"] == [f"pypdf=={pinned}"]
    assert "keywords" in pp and "classifiers" in pp                             # the extra didn't swallow [project] keys


def test_the_binary_bundles_every_stdlib_module_pypdf_imports():
    """Frozen builds only contain modules something imports; pypdf comes from its wheel later (#113: uuid was missing)."""
    import ast
    from pathlib import Path
    wheel = next((p for p in (runtime.HOME / "lib" / docs.PYPDF[0], Path.home() / ".localllm" / "lib" / docs.PYPDF[0])
                  if p.is_file()), None)
    if wheel is None or not hasattr(sys, "stdlib_module_names"):
        pytest.skip("the pinned pypdf wheel isn't on this machine (or Python < 3.10)")
    names = set()
    with zipfile.ZipFile(wheel) as z:
        for n in (n for n in z.namelist() if n.endswith(".py")):
            for node in ast.walk(ast.parse(z.read(n))):
                if isinstance(node, ast.Import):
                    names |= {a.name for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names.add(node.module)
    stdlib = {n for n in names if n.split(".")[0] in sys.stdlib_module_names} - {"__future__"}
    assert stdlib <= set(docs.PYPDF_STDLIB), stdlib - set(docs.PYPDF_STDLIB)
    assert "--hidden-import" in (Path(__file__).resolve().parents[1] / "packaging" / "build_exe.py").read_text()
