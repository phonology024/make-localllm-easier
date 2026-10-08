"""Trim a GGUF's vocabulary to the scripts you use (issue #27).

Most of a 150-260k-token vocabulary is other languages' text: a Thai + English user never needs the ~25k Chinese tokens.
Dropping them shrinks the token embedding and output matrices (and any other per-token tensor), so the file is
smaller, less of it stays CPU-mapped, and the output layer runs faster.

  python tools/trim_vocab.py trim  MODEL.gguf OUT.gguf --langs th,en --dry-run   # what would be kept, sizes saved
  python tools/trim_vocab.py trim  MODEL.gguf OUT.gguf --langs th,en             # keep every Thai/Latin/symbol token
  python tools/trim_vocab.py check MODEL.gguf OUT.gguf --texts DIR               # same tokens as before, per text file

Text written only in the kept scripts tokenizes exactly as before (`check` verifies it on DIR/*.txt), so the model sees
the same input and its logits for kept tokens are unchanged. Text in other scripts still works: the merges that built
dropped tokens are removed too, so it falls back to smaller kept pieces (bytes at worst) instead of becoming
un-encodable. `--max-vocab N` also drops the rarest ASCII/symbol tokens (latest BPE merges first) and `--keep-text FILE`
protects every token FILE uses. Tokens holding the chosen languages' letters are never capped, only ASCII/symbol ones,
but rare English words and code identifiers then split differently, so measure before using it.

Supports BPE vocabularies with merges (Qwen, Llama 3, Gemma 4, ...). Needs `pip install gguf numpy`; `check` and
`--keep-text` run llama-tokenize (next to the llama-server localllm downloaded, or --tokenizer).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:  # pragma: no cover
    sys.exit("needs the gguf package: pip install gguf numpy")

# Unicode blocks per writing system, inclusive. "common" (ASCII, punctuation, symbols, math, emoji) is always kept,
# so code, math and emoji survive any language choice.
SCRIPTS = {
    "common": [(0x00, 0x7F), (0xA0, 0xBF), (0xD7, 0xD7), (0xF7, 0xF7), (0x2000, 0x2BFF), (0xFE0E, 0xFE0F),
               (0xFFFD, 0xFFFD), (0x1D400, 0x1D7FF), (0x1F000, 0x1FAFF)],
    "latin": [(0xC0, 0x36F), (0x1E00, 0x1EFF), (0x2C60, 0x2C7F), (0xA720, 0xA7FF)],
    "thai": [(0x0E00, 0x0E7F)],
    "lao": [(0x0E80, 0x0EFF)],
    "khmer": [(0x1780, 0x17FF), (0x19E0, 0x19FF)],
    "myanmar": [(0x1000, 0x109F), (0xA9E0, 0xA9FF), (0xAA60, 0xAA7F)],
    "devanagari": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "bengali": [(0x0980, 0x09FF)],
    "gurmukhi": [(0x0A00, 0x0A7F)],
    "gujarati": [(0x0A80, 0x0AFF)],
    "tamil": [(0x0B80, 0x0BFF)],
    "telugu": [(0x0C00, 0x0C7F)],
    "kannada": [(0x0C80, 0x0CFF)],
    "malayalam": [(0x0D00, 0x0D7F)],
    "sinhala": [(0x0D80, 0x0DFF)],
    "arabic": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x0870, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "hebrew": [(0x0590, 0x05FF), (0xFB1D, 0xFB4F)],
    "cyrillic": [(0x0400, 0x052F), (0x1C80, 0x1C8F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)],
    "greek": [(0x0370, 0x03FF), (0x1F00, 0x1FFF)],
    "georgian": [(0x10A0, 0x10FF), (0x1C90, 0x1CBF), (0x2D00, 0x2D2F)],
    "armenian": [(0x0530, 0x058F)],
    "ethiopic": [(0x1200, 0x139F), (0x2D80, 0x2DDF)],
    "hangul": [(0x1100, 0x11FF), (0x3130, 0x318F), (0xA960, 0xA97F), (0xAC00, 0xD7FF)],
    "kana": [(0x3040, 0x30FF), (0x31F0, 0x31FF)],
    "han": [(0x2E80, 0x2FDF), (0x31C0, 0x31EF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF),
            (0x20000, 0x2FFFF)],
    "cjk-punct": [(0x3000, 0x303F), (0x3200, 0x33FF), (0xFF00, 0xFFEF)],
}
LATIN_LANGS = ("en es fr de it pt nl id ms vi tr pl sv da no nb fi cs sk ro hu hr sl et lv lt sw tl fil ca eu gl ga cy "
               "is mt sq az uz so ha yo ig zu xh af jv su").split()
LANG_SCRIPTS = {**{c: ["latin"] for c in LATIN_LANGS},
                "th": ["thai"], "lo": ["lao"], "km": ["khmer"], "my": ["myanmar"],
                "hi": ["devanagari"], "mr": ["devanagari"], "ne": ["devanagari"], "bn": ["bengali"],
                "pa": ["gurmukhi"], "gu": ["gujarati"], "ta": ["tamil"], "te": ["telugu"], "kn": ["kannada"],
                "ml": ["malayalam"], "si": ["sinhala"], "ar": ["arabic"], "fa": ["arabic"], "ur": ["arabic"],
                "he": ["hebrew"], "yi": ["hebrew"], "ru": ["cyrillic"], "uk": ["cyrillic"], "bg": ["cyrillic"],
                "sr": ["cyrillic", "latin"], "mk": ["cyrillic"], "be": ["cyrillic"], "kk": ["cyrillic"],
                "ky": ["cyrillic"], "mn": ["cyrillic"], "el": ["greek"], "ka": ["georgian"], "hy": ["armenian"],
                "am": ["ethiopic"], "ko": ["hangul", "cjk-punct"], "zh": ["han", "cjk-punct"],
                "ja": ["han", "kana", "cjk-punct"]}

NORMAL, UNUSED = 1, 5   # llama_token_type; every other type (control, user-defined, unknown, byte) is always kept
SPACE = "\u2581"        # SentencePiece-style space marker used by Gemma 4's BPE
PLACEHOLDER = re.compile(r"<unused\d+>")


def scripts_for(langs: list[str]) -> list[str]:
    out = ["common"]
    for code in langs:
        if code in SCRIPTS:
            out.append(code)
        elif code in LANG_SCRIPTS:
            out += LANG_SCRIPTS[code]
        else:
            known = ", ".join(sorted({*LANG_SCRIPTS, *SCRIPTS}))
            raise SystemExit(f"unknown language or script '{code}'; known: {known}")
    return sorted(set(out))


class Charset:
    """Which byte strings can occur inside text written only in the allowed scripts. Byte-level BPE tokens may hold
    part of a multi-byte character, so a token is allowed if its leading and trailing fragments can belong to an
    allowed character too."""

    def __init__(self, scripts: list[str]):
        self.ranges = [r for s in scripts for r in SCRIPTS[s]]
        self.heads: set[bytes] = set()   # proper prefixes of allowed characters' UTF-8
        self.tails: set[bytes] = set()   # proper suffixes
        self.inner: set[bytes] = set()   # any proper substring made only of continuation bytes
        for lo, hi in self.ranges:
            for cp in range(max(lo, 0x80), hi + 1):
                if 0xD800 <= cp <= 0xDFFF:
                    continue
                b = chr(cp).encode()
                for i in range(1, len(b)):
                    self.heads.add(b[:i])
                    self.tails.add(b[i:])
                    for j in range(i + 1, len(b) + 1):
                        self.inner.add(b[i:j])

    def char_ok(self, ch: str) -> bool:
        cp = ord(ch)
        return any(lo <= cp <= hi for lo, hi in self.ranges)

    def bytes_ok(self, b: bytes) -> bool:
        lead = 0
        while lead < len(b) and 0x80 <= b[lead] <= 0xBF:
            lead += 1
        if lead == len(b):
            return b in self.inner
        if lead and b[:lead] not in self.tails:
            return False
        rest = b[lead:]
        for cut in range(len(rest), max(len(rest) - 4, -1), -1):   # longest valid UTF-8 prefix + incomplete tail
            try:
                text = rest[:cut].decode("utf-8")
            except UnicodeDecodeError:
                continue
            tail = rest[cut:]
            return (not tail or tail in self.heads) and all(self.char_ok(c) for c in text)
        return False


def _byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's bytes_to_unicode: byte-level BPE stores each byte as one printable character."""
    bs = [*range(ord("!"), ord("~") + 1), *range(ord("\xa1"), ord("\xac") + 1), *range(ord("\xae"), ord("\xff") + 1)]
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BYTE_DECODER = _byte_decoder()


def byte_level(model: str, pre: str) -> bool:
    if model == "gpt2" and pre not in ("sarvam-moe", "mmbert", "granite-embed-multi-311m"):
        return True
    if model == "gemma4" or pre in ("sarvam-moe", "mmbert", "granite-embed-multi-311m"):
        return False
    raise SystemExit(f"tokenizer '{model}' (pre '{pre}') is not supported yet: only BPE vocabularies with merges")


def token_bytes(text: str, is_byte_level: bool) -> bytes:
    if is_byte_level:
        if all(c in BYTE_DECODER for c in text):
            return bytes(BYTE_DECODER[c] for c in text)
        return text.encode()
    return text.replace(SPACE, " ").encode()


def split_merge(m: str) -> tuple[str, str]:
    pos = m.find(" ", 1)   # same split as llama.cpp's merge loader
    return (m[:pos], m[pos + 1:]) if pos > 0 else (m, "")


def select(tokens: list[str], types: list[int], merges: list[str], is_byte_level: bool, scripts: list[str],
           extra: set[int] = frozenset(), forced: set[int] = frozenset(), max_vocab: int | None = None) -> dict:
    """Token ids to keep. Always: special/byte tokens, the 256 single-byte tokens of byte-level BPE, `forced` and
    `extra` ids, and every token holding a letter of the chosen languages (Thai, accented Latin, ...). Then the
    ASCII/symbol tokens, most frequent first (earliest BPE merge), until `max_vocab`. Merge rank can't rank the
    languages' own letters: smaller languages' merges come late in the list, so a rank cut would drop them first.
    Never: placeholders like Gemma's 6k `<unusedN>`, which no merge builds and the model was never trained to emit.
    Closed under merges: a kept token keeps the pieces every merge that builds it starts from, so text that used only
    kept tokens before goes through exactly the same merges after trimming."""
    n = len(tokens)
    to_id = {t: i for i, t in enumerate(tokens)}
    built_by: dict[str, list[tuple[str, str]]] = {}
    rank = [-1] * n                                     # earliest merge that builds the token; -1 = base piece
    for r, m in enumerate(merges):
        a, b = split_merge(m)
        built_by.setdefault(a + b, []).append((a, b))
        i = to_id.get(a + b)
        if i is not None and rank[i] < 0:
            rank[i] = r
    keep: set[int] = set()

    def add(i: int) -> None:
        stack = [i]
        while stack:
            j = stack.pop()
            if j in keep:
                continue
            keep.add(j)
            for a, b in built_by.get(tokens[j], ()):
                stack += [to_id[p] for p in (a, b) if p in to_id and to_id[p] not in keep]

    charset = Charset(scripts)
    core = {i for i in range(n) if types[i] not in (NORMAL, UNUSED)} | set(forced) | set(extra)
    if is_byte_level:
        core |= {i for i in range(n) if len(token_bytes(tokens[i], True)) == 1}
    for i in sorted(core):
        add(i)
    n_core = len(keep)
    letters = [r for s in scripts if s != "common" for r in SCRIPTS[s]]
    allowed = []
    for i in range(n):
        if types[i] != NORMAL or i in keep or PLACEHOLDER.fullmatch(tokens[i]):
            continue
        b = token_bytes(tokens[i], is_byte_level)
        if not charset.bytes_ok(b):
            continue
        if any(lo <= ord(c) <= hi for c in b.decode("utf-8", "ignore") for lo, hi in letters):
            add(i)
        else:
            allowed.append(i)
    n_letters = len(keep) - n_core

    def priority(i: int) -> tuple[int, int]:
        t = tokens[i]
        if rank[i] < 0 and len(t) > 1 and t.replace(SPACE, " ").strip():
            return len(merges), i                       # no merge builds it: only reachable as a whole word, last
        return rank[i], i                               # base characters and whitespace runs (-1) first

    for i in sorted(allowed, key=priority):
        if max_vocab is not None and len(keep) >= max_vocab:
            break
        add(i)
    return {"keep": sorted(keep), "n_core": n_core, "n_letters": n_letters, "n_ascii": len(allowed)}


def kept_merges(merges: list[str], tokens: list[str], keep: list[int]) -> list[str]:
    """Drop every merge that builds or starts from a dropped token, so its pieces stay as kept tokens instead of
    merging into something that is no longer in the vocabulary (which llama.cpp would spell out byte by byte)."""
    vocab = set(tokens)
    kept = {tokens[i] for i in keep}
    dropped = lambda s: s in vocab and s not in kept   # noqa: E731
    out = []
    for m in merges:
        a, b = split_merge(m)
        if not (dropped(a) or dropped(b) or dropped(a + b)):
            out.append(m)
    return out


def _field(reader: gguf.GGUFReader, key: str, default=None):
    f = reader.fields.get(key)
    return f.contents() if f is not None else default


def _token_id_fields(reader: gguf.GGUFReader) -> dict[str, object]:
    return {k: f.contents() for k, f in reader.fields.items()
            if k.startswith("tokenizer.ggml.") and (k.endswith("_token_id") or k.endswith("_token_ids"))}


def vocab_tensors(reader: gguf.GGUFReader, n_vocab: int) -> list:
    """Tensors with one row (or element) per token: token_embd, output, per-layer embeddings, MTP heads, biases."""
    out = []
    for t in reader.tensors:
        dims = [int(d) for d in t.shape]
        if len(dims) in (1, 2) and dims[-1] == n_vocab:
            out.append(t)
        elif n_vocab in dims:
            raise SystemExit(f"{t.name} {dims}: has a vocabulary-sized dimension this tool doesn't know how to slice")
    return out


def trim(src: Path, dst: Path | None, langs: list[str], max_vocab: int | None = None, extra: set[int] = frozenset(),
         dry_run: bool = False) -> dict:
    reader = gguf.GGUFReader(src)
    tokens = _field(reader, "tokenizer.ggml.tokens")
    types = _field(reader, "tokenizer.ggml.token_type")
    merges = _field(reader, "tokenizer.ggml.merges", [])
    if tokens is None or types is None or not merges:
        raise SystemExit("needs tokenizer.ggml.tokens, token_type and merges (BPE vocabulary)")
    is_bl = byte_level(_field(reader, "tokenizer.ggml.model", ""), _field(reader, "tokenizer.ggml.pre", ""))
    id_fields = _token_id_fields(reader)
    forced = {int(v) for vals in id_fields.values() for v in (vals if isinstance(vals, list) else [vals])
              if 0 <= int(v) < len(tokens)}
    scripts = scripts_for(langs)
    sel = select(tokens, types, merges, is_bl, scripts, extra, forced, max_vocab)
    keep = sel["keep"]
    new_id = {old: new for new, old in enumerate(keep)}
    new_merges = kept_merges(merges, tokens, keep)
    n, k = len(tokens), len(keep)
    vt = vocab_tensors(reader, n)
    saved = sum(t.n_bytes - t.n_bytes // n * k for t in vt)
    report = {"model": src.name, "langs": langs, "scripts": scripts, "max_vocab": max_vocab, "n_vocab": n,
              "n_kept": k, "n_core": sel["n_core"], "n_letters": sel["n_letters"], "n_ascii": sel["n_ascii"],
              "merges": len(merges), "merges_kept": len(new_merges), "file_gb": src.stat().st_size / 2**30,
              "saved_gb": saved / 2**30,
              "tensors": {t.name: [t.tensor_type.name, t.n_bytes / 2**20, t.n_bytes // n * k / 2**20] for t in vt}}
    if dry_run or dst is None:
        return report

    arch = _field(reader, "general.architecture")
    per_token = {key for key, f in reader.fields.items() if key.startswith("tokenizer.ggml.")
                 and key != "tokenizer.ggml.merges" and f.types and f.types[0] == gguf.GGUFValueType.ARRAY
                 and len(f.data) == n}
    w = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)
    for key, f in reader.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        if key == "general.alignment":
            w.add_custom_alignment(int(f.contents()))
            continue
        vtype = f.types[0]
        sub = f.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        val = f.contents()
        if key in per_token:
            val = [val[i] for i in keep]
        elif key == "tokenizer.ggml.merges":
            val = new_merges
        elif key in id_fields:
            remap = lambda v: new_id.get(int(v), int(v))   # noqa: E731
            val = [remap(v) for v in val] if isinstance(val, list) else remap(val)
        elif key == f"{arch}.vocab_size":
            val = k
        w.add_key_value(key, val, vtype, sub_type=sub)
    w.add_string("localllm.vocab_trim", json.dumps({"langs": langs, "max_vocab": max_vocab, "n_vocab_orig": n}))

    rows = np.asarray(keep)
    sliced = {t.name for t in vt}
    datas = []
    for t in reader.tensors:
        data = t.data[rows] if t.name in sliced else t.data
        datas.append(data)
        w.add_tensor_info(t.name, data.shape, data.dtype, data.nbytes, t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for data in datas:
        w.write_tensor_data(data)
    w.close()
    report["out_gb"] = dst.stat().st_size / 2**30
    return report


def find_tokenizer(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from localllm import runtime
    server = runtime.find_server()
    return server.with_name("llama-tokenize" + server.suffix)


def tokenize(tokenizer: Path, model: Path, text_file: Path) -> list[int]:
    out = subprocess.run([str(tokenizer), "-m", str(model), "-f", str(text_file), "--ids", "--log-disable",
                          "--no-bos"], capture_output=True, text=True, errors="replace")
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("[")]
    if out.returncode or not lines:
        raise SystemExit(f"llama-tokenize failed on {text_file}:\n{out.stderr[-800:]}")
    return json.loads(lines[-1])


def check(orig: Path, trimmed: Path, texts: Path, tokenizer: Path) -> list[dict]:
    """Per text file: does the trimmed model split it into exactly the same tokens? And how many more tokens text in
    dropped scripts now takes."""
    old = _field(gguf.GGUFReader(orig), "tokenizer.ggml.tokens")
    new = {t: i for i, t in enumerate(_field(gguf.GGUFReader(trimmed), "tokenizer.ggml.tokens"))}
    old_to_new = [new.get(t) for t in old]
    rows = []
    for txt in sorted(texts.glob("*.txt")):
        a, b = tokenize(tokenizer, orig, txt), tokenize(tokenizer, trimmed, txt)
        mapped = [old_to_new[i] for i in a]
        rows.append({"text": txt.stem, "tokens": len(a), "tokens_trimmed": len(b),
                     "kept_pct": 100 * sum(m is not None for m in mapped) / max(len(a), 1),
                     "identical": mapped == b, "growth_pct": 100 * (len(b) / max(len(a), 1) - 1)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trim", help="write a GGUF that keeps only the chosen languages' tokens")
    t.add_argument("model", type=Path)
    t.add_argument("out", type=Path, nargs="?")
    t.add_argument("--langs", required=True, help="language codes or script names, e.g. th,en or thai,latin")
    t.add_argument("--max-vocab", type=int, help="also drop the rarest ASCII/symbol tokens down to about this many")
    t.add_argument("--keep-text", type=Path, action="append", default=[], help="never drop tokens this text uses")
    t.add_argument("--tokenizer", help="llama-tokenize path (for --keep-text)")
    t.add_argument("--dry-run", action="store_true")
    c = sub.add_parser("check", help="compare tokenization of DIR/*.txt before and after trimming")
    c.add_argument("orig", type=Path)
    c.add_argument("trimmed", type=Path)
    c.add_argument("--texts", type=Path, required=True)
    c.add_argument("--tokenizer", help="llama-tokenize path")
    a = ap.parse_args()

    if a.cmd == "check":
        for r in check(a.orig, a.trimmed, a.texts, find_tokenizer(a.tokenizer)):
            print(f"{r['text']:>10}: {r['tokens']:>7} -> {r['tokens_trimmed']:>7} tokens ({r['growth_pct']:+.1f}%), "
                  f"{r['kept_pct']:.1f}% of its tokens kept, {'identical' if r['identical'] else 'CHANGED'}")
        return
    if a.out is None and not a.dry_run:
        ap.error("give OUT.gguf or --dry-run")
    extra: set[int] = set()
    for f in a.keep_text:
        extra |= set(tokenize(find_tokenizer(a.tokenizer), a.model, f))
    langs = [s.strip() for s in a.langs.split(",") if s.strip()]
    r = trim(a.model, a.out, langs, a.max_vocab, extra, a.dry_run)
    print(f"vocab {r['n_vocab']:,} -> {r['n_kept']:,} tokens ({100 * r['n_kept'] / r['n_vocab']:.1f}%), "
          f"scripts: {', '.join(r['scripts'])}")
    print(f"  {r['n_core']:,} special/byte/forced + {r['n_letters']:,} with the languages' letters + "
          + (f"the most frequent of {r['n_ascii']:,} ASCII/symbol tokens up to {r['max_vocab']:,}" if r["max_vocab"]
             else f"{r['n_ascii']:,} ASCII/symbol tokens"))
    if r["max_vocab"] and r["n_kept"] > r["max_vocab"]:
        print(f"  (special tokens + the languages' own tokens alone are more than --max-vocab {r['max_vocab']:,})")
    print(f"merges {r['merges']:,} -> {r['merges_kept']:,}")
    for name, (qt, before, after) in r["tensors"].items():
        print(f"  {name:<40} {qt:<6} {before:8.1f} -> {after:8.1f} MiB")
    print(f"file {r['file_gb']:.2f} GiB, {r['saved_gb']:.2f} GiB smaller"
          + (f" -> {r['out_gb']:.2f} GiB written to {a.out}" if "out_gb" in r else " (dry run)"))


if __name__ == "__main__":
    main()
