import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
gguf = pytest.importorskip("gguf")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import trim_vocab as tv  # noqa: E402

UNI = {b: c for c, b in tv.BYTE_DECODER.items()}   # byte -> GPT-2 printable character


def enc(b: bytes) -> str:
    return "".join(UNI[x] for x in b)


def tiny_vocab():
    """Byte-level BPE with English, Thai and Chinese words, built the way real merges build them."""
    tokens = [enc(bytes([b])) for b in range(256)]
    merges = []

    def merge(a: bytes, b: bytes) -> None:
        merges.append(f"{enc(a)} {enc(b)}")
        tokens.append(enc(a + b))

    merge(b"h", b"e"); merge(b"l", b"l"); merge(b"he", b"ll"); merge(b"hell", b"o")
    sa = "ส".encode()                                   # e0 b8 aa
    merge(sa[:1], sa[1:2]); merge(sa[:2], sa[2:])
    zh = "中".encode()                                  # e4 b8 ad
    merge(zh[:1], zh[1:2]); merge(zh[:2], zh[2:])
    types = [tv.NORMAL] * len(tokens) + [3, 4]
    tokens += ["<|endoftext|>", "<|im_start|>"]
    return tokens, types, merges


def write_model(path: Path) -> list[str]:
    tokens, types, merges = tiny_vocab()
    n = len(tokens)
    w = gguf.GGUFWriter(path, arch="qwen2")
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen2")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(merges)
    w.add_eos_token_id(n - 2)
    w.add_pad_token_id(n - 2)
    rows = np.repeat(np.arange(n, dtype=np.float32)[:, None], 32, axis=1) / 100
    w.add_tensor("token_embd.weight", rows)
    q = gguf.quants.quantize(rows, gguf.GGMLQuantizationType.Q8_0)
    w.add_tensor("output.weight", q, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    w.add_tensor("blk.0.attn_norm.weight", np.ones(32, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return tokens


def test_charset_allows_fragments_of_allowed_characters_only():
    thai = tv.Charset(tv.scripts_for(["th"]))
    assert thai.bytes_ok("สวัสดี hello!".encode())
    assert thai.bytes_ok(b"\xe0\xb8")          # start of a Thai character
    assert thai.bytes_ok(b"\xb8\xad")          # end of Thai อ (e0 b8 ad)
    assert not thai.bytes_ok("中".encode())
    assert not thai.bytes_ok(b"\xe4\xb8")      # start of a CJK character
    assert not thai.bytes_ok(b"a\xff")


def test_unknown_language_is_an_error():
    with pytest.raises(SystemExit):
        tv.scripts_for(["xx"])


def test_select_keeps_every_piece_a_kept_word_is_built_from():
    tokens, types, merges = tiny_vocab()
    hello = tokens.index(enc(b"hello"))
    sel = tv.select(tokens, types, merges, True, tv.scripts_for(["en"]), extra={hello}, max_vocab=256 + 6)
    kept = {tokens[i] for i in sel["keep"]}
    assert {enc(b"hello"), enc(b"hell"), enc(b"he"), enc(b"ll"), "<|endoftext|>", "<|im_start|>"} <= kept
    assert enc("ส".encode()) not in kept and len(kept) == 256 + 6
    with pytest.raises(SystemExit):                      # a cap below what the languages need is refused
        tv.select(tokens, types, merges, True, tv.scripts_for(["en"]), extra={hello}, max_vocab=100)


def test_capped_raw_utf8_vocab_keeps_base_characters_and_newline_runs():
    """Gemma 4 style (no byte-level encoding, <0xXX> fallback): single characters and newline runs are never capped."""
    tokens = [f"<0x{b:02X}>" for b in range(256)] + list("ab\n") + ["\n\n", "ab", "▁ab", "▁", "ส", "สส"]
    types = [tv.BYTE] * 256 + [tv.NORMAL] * 9
    merges = ["a b", "▁ ab", "ส ส"]
    sel = tv.select(tokens, types, merges, False, tv.scripts_for(["th"]), max_vocab=256 + 7)
    kept = {tokens[i] for i in sel["keep"]}
    assert {"a", "b", "\n", "\n\n", "▁", "ส", "สส"} <= kept and len(kept) == 256 + 7   # ab, ▁ab capped


def test_suppress_tokens_and_arch_token_ids_follow_their_tokens(tmp_path):
    src, dst = tmp_path / "m.gguf", tmp_path / "t.gguf"
    tokens, types, merges = tiny_vocab()
    zh, im = tokens.index(enc("中".encode())), tokens.index("<|im_start|>")
    w = gguf.GGUFWriter(src, arch="qwen2")
    w.add_tokenizer_model("gpt2"); w.add_tokenizer_pre("qwen2")
    w.add_token_list(tokens); w.add_token_types(types); w.add_token_merges(merges)
    w.add_array("tokenizer.ggml.suppress_tokens", [im, zh])
    w.add_uint32("qwen2.decoder_start_token_id", im)
    w.add_tensor("token_embd.weight", np.zeros((len(tokens), 32), np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    tv.trim(src, dst, ["th"])
    r = gguf.GGUFReader(dst)
    new = tv._field(r, "tokenizer.ggml.tokens")
    assert [new[i] for i in tv._field(r, "tokenizer.ggml.suppress_tokens")] == ["<|im_start|>"]   # 中 is gone
    assert new[tv._field(r, "qwen2.decoder_start_token_id")] == "<|im_start|>"


def test_trim_writes_a_consistent_smaller_model(tmp_path):
    src, dst = tmp_path / "m.gguf", tmp_path / "t.gguf"
    tokens = write_model(src)
    report = tv.trim(src, dst, ["th"])
    r = gguf.GGUFReader(dst)
    new = tv._field(r, "tokenizer.ggml.tokens")
    assert report["n_kept"] == len(new) == len(tokens) - 2              # 中 and its first two bytes are gone
    assert enc("中".encode()) not in new and enc("中".encode()[:2]) not in new
    assert enc("ส".encode()) in new and enc(b"hello") in new
    merges = tv._field(r, "tokenizer.ggml.merges")
    assert len(merges) == 6 and all(enc(b"\xe4") + " " not in m for m in merges)
    assert new[tv._field(r, "tokenizer.ggml.eos_token_id")] == "<|endoftext|>"
    assert new[tv._field(r, "tokenizer.ggml.padding_token_id")] == "<|endoftext|>"
    assert len(tv._field(r, "tokenizer.ggml.token_type")) == len(new)

    old = {t.name: t for t in gguf.GGUFReader(src).tensors}
    keep = [tokens.index(t) for t in new]
    for t in r.tensors:
        if t.name == "blk.0.attn_norm.weight":
            assert np.array_equal(t.data, old[t.name].data)
        else:
            assert [int(d) for d in t.shape][-1] == len(new)
            assert np.array_equal(t.data, old[t.name].data[keep])      # same rows, quantized bytes untouched
    assert '"n_vocab_orig": %d' % len(tokens) in tv._field(r, "localllm.vocab_trim")


def test_dry_run_reports_savings_without_writing(tmp_path):
    src = tmp_path / "m.gguf"
    write_model(src)
    report = tv.trim(src, tmp_path / "t.gguf", ["th"], dry_run=True)
    assert not (tmp_path / "t.gguf").exists()
    assert set(report["tensors"]) == {"token_embd.weight", "output.weight"} and report["saved_gb"] > 0


def test_detokenize_round_trips_kept_and_byte_fallback_text(tmp_path):
    src, dst = tmp_path / "m.gguf", tmp_path / "t.gguf"
    tokens = write_model(src)
    tv.trim(src, dst, ["th"])
    v = tv.Vocab(dst)
    ids = {t: i for i, t in enumerate(v.tokens)}
    text = "hello ส 中"
    pieces = [enc(b"hello"), enc(b" "), enc("ส".encode()), enc(b" ")] + [enc(bytes([b])) for b in "中".encode()]
    assert v.detokenize([ids[p] for p in pieces]).decode() == text      # 中 is spelled out in kept byte tokens
    assert tv.Vocab(src).to(v)[tokens.index(enc("中".encode()))] is None


def write_logprobs(path: Path, logits: np.ndarray, tokens: list[int]) -> None:
    """A llama-perplexity --kl-divergence-base file for one chunk whose scored rows are `logits`."""
    n_rows, n_vocab = logits.shape
    n_ctx = 2 * n_rows + 2                                   # rows = n_ctx - 1 - n_ctx // 2
    nv = 2 * ((n_vocab + 1) // 2) + 4
    out = np.zeros((n_rows, nv), np.uint16)
    for r, lg in enumerate(logits):
        mx = lg.max(); mn = max(lg.min(), mx - 16)
        lse = np.log(np.exp(lg - mx).sum())
        scale = (mx - mn) / 65535
        out[r, :4] = np.array([scale, mn - mx - lse], np.float32).view(np.uint16)
        out[r, 4:4 + n_vocab] = np.where(lg > mn, np.rint((lg - mn) / scale), 0)
    with open(path, "wb") as f:
        f.write(b"_logits_")
        np.array([n_ctx, n_vocab, 1], np.int32).tofile(f)
        np.array((tokens * n_ctx)[:n_ctx], np.int32).tofile(f)
        out.tofile(f)


def test_compare_logprobs_counts_top1_agreement_and_dropped_mass(tmp_path):
    keep = [0, 2, 3]                                          # old ids kept; old id 1 is dropped
    old_to_new = [0, None, 1, 2]
    orig = np.array([[5.0, 1.0, 0.0, 0.0],                   # top-1 kept (0)
                     [0.0, 6.0, 5.0, 0.0],                   # top-1 dropped (1): trimmed picks 2 instead
                     [0.0, 0.0, 0.0, 3.0]], np.float32)
    trimmed = orig[:, keep]
    write_logprobs(tmp_path / "o.bin", orig, [0, 2, 3])
    write_logprobs(tmp_path / "t.bin", trimmed, [0, 1, 2])
    r = tv.compare_logprobs(tmp_path / "o.bin", tmp_path / "t.bin", old_to_new)
    assert r["comparable"] and r["positions"] == 3
    assert round(r["same_top1_pct"], 1) == 66.7 and round(r["orig_top1_kept_pct"], 1) == 66.7
    p = np.exp(orig) / np.exp(orig).sum(1, keepdims=True)
    assert abs(r["dropped_mass_mean"] - p[:, 1].mean()) < 1e-3

    write_logprobs(tmp_path / "t2.bin", trimmed, [0, 2, 2])  # different tokens: not comparable
    assert not tv.compare_logprobs(tmp_path / "o.bin", tmp_path / "t2.bin", old_to_new)["comparable"]


def test_compare_logprobs_aligns_differently_tokenized_runs(tmp_path):
    """Original splits "hello" as one token, the trimmed vocab as "he" + "llo": positions are compared where both runs
    have a token boundary, by the text of the most likely next token."""
    pieces_o = [b"a", b" ", b"hello", b"he", b"llo"]
    pieces_t = [b"a", b" ", b"he", b"llo"]
    old_to_new = [0, 1, None, 2, 3]
    # text "a a a hello a a" ; orig: a,_,a,_,a,_,hello,_,a,_,a ; trimmed: ..., he, llo, ...
    tok_o = [0, 1, 0, 1, 0, 1, 2, 1, 0, 1, 0, 1]
    tok_t = [0, 1, 0, 1, 0, 1, 2, 3, 1, 0, 1, 0]
    n_rows_o = 5                                          # n_ctx 12 -> rows score tokens 7..11
    # rows predict the token starting at byte 11, 12, 13, 14, 15 (orig) and 8, 11, 12, 13, 14 (trimmed)
    lo = np.zeros((n_rows_o, 5), np.float32); lo[np.arange(5), [2, 0, 1, 0, 1]] = 9    # hello, a, " ", a, " "
    lt = np.zeros((5, 4), np.float32); lt[np.arange(5), [1, 2, 0, 1, 3]] = 9          # " ", he, a, " ", llo
    write_logprobs(tmp_path / "o.bin", lo, tok_o)
    write_logprobs(tmp_path / "t.bin", lt, tok_t)
    r = tv.compare_logprobs(tmp_path / "o.bin", tmp_path / "t.bin", old_to_new, pieces_o, pieces_t)
    assert r["comparable"] and r["mode"] == "aligned"
    assert r["positions"] == 4 and r["coverage_pct"] == 80.0                 # bytes 11-14 are boundaries in both
    assert r["same_top1_pct"] == 50.0 and r["compatible_pct"] == 75.0       # hello~he, a=a, " "=" ", a!=llo


def test_raw_utf8_vocab_trim_drops_placeholders_and_decodes_byte_fallback(tmp_path):
    """Gemma 4 style GGUF: space marker, <0xXX> byte tokens, <unusedN> placeholders."""
    tokens = [f"<0x{b:02X}>" for b in range(256)] + ["<bos>", "<unused0>", "▁", "a", "b", "ab", "▁ab", "ส", "中"]
    types = [tv.BYTE] * 256 + [3] + [tv.NORMAL] * 8
    src, dst = tmp_path / "g.gguf", tmp_path / "t.gguf"
    w = gguf.GGUFWriter(src, arch="gemma4")
    w.add_tokenizer_model("gemma4")
    w.add_token_list(tokens); w.add_token_types(types); w.add_token_merges(["a b", "▁ ab"])
    w.add_bos_token_id(256)
    w.add_tensor("token_embd.weight", np.arange(len(tokens) * 32, dtype=np.float32).reshape(len(tokens), 32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    tv.trim(src, dst, ["th"])
    v = tv.Vocab(dst)
    assert "<unused0>" not in v.tokens and "中" not in v.tokens and "ส" in v.tokens
    ids = {t: i for i, t in enumerate(v.tokens)}
    pieces = ["▁ab", "ส", "▁"] + [f"<0x{b:02X}>" for b in "中".encode()]
    assert v.detokenize([ids[x] for x in pieces]).decode() == " abส 中"
    assert v.tokens[tv._field(gguf.GGUFReader(dst), "tokenizer.ggml.bos_token_id")] == "<bos>"


def test_trim_refuses_to_overwrite_its_input(tmp_path):
    src = tmp_path / "m.gguf"
    write_model(src)
    before = src.read_bytes()
    with pytest.raises(SystemExit):
        tv.trim(src, src, ["th"])
    assert src.read_bytes() == before
