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
    sel = tv.select(tokens, types, merges, True, tv.scripts_for(["en"]), extra={hello}, max_vocab=0)
    kept = {tokens[i] for i in sel["keep"]}
    assert {enc(b"hello"), enc(b"hell"), enc(b"he"), enc(b"ll"), "<|endoftext|>", "<|im_start|>"} <= kept
    assert enc("ส".encode()) not in kept and len(kept) == 256 + 6


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
