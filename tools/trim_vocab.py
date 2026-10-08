"""Trim a GGUF's vocabulary to the scripts you use (issue #27).

Most of a 150-260k-token vocabulary is other languages' text: a Thai + English user never needs the ~25k Chinese tokens.
Dropping them shrinks the token embedding and output matrices (and any other per-token tensor), so the file is
smaller, less of it stays CPU-mapped, and the output layer runs faster.

  python tools/trim_vocab.py trim  MODEL.gguf OUT.gguf --langs th,en --dry-run   # what would be kept, sizes saved
  python tools/trim_vocab.py trim  MODEL.gguf OUT.gguf --langs th,en             # keep every Thai/Latin/symbol token
  python tools/trim_vocab.py check MODEL.gguf OUT.gguf --texts DIR               # same tokens as before, per text file
  python tools/trim_vocab.py agree MODEL.gguf OUT.gguf --texts DIR               # same next-token pick, per text file

Text written only in the kept scripts tokenizes exactly as before (`check` verifies it on DIR/*.txt), so the model sees
the same input and its logits for kept tokens are unchanged. Text in other scripts still works: the merges that built
dropped tokens are removed too, so it falls back to smaller kept pieces (bytes at worst) instead of becoming
un-encodable. `--max-vocab N` also drops the rarest ASCII/symbol tokens (latest BPE merges first) and `--keep-text FILE`
protects every token FILE uses. Tokens holding the chosen languages' letters are never capped, only ASCII/symbol ones,
but rare English words and code identifiers then split differently, so measure before using it.

`agree` runs both models with llama-perplexity on each text and compares their predictions position by position: how
often the most likely next token is the same, and how much probability the original put on tokens that were dropped.

Supports BPE vocabularies with merges (Qwen, Llama 3, Gemma 4, ...). Needs `pip install gguf numpy`; `check`, `agree`
and `--keep-text` run llama-tokenize / llama-perplexity (next to the llama-server localllm downloaded, or --bin-dir).
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import tempfile
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

NORMAL, CONTROL, UNUSED, BYTE = 1, 3, 5, 6   # llama_token_type; all but normal/unused are always kept
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
    else:                                               # base characters and Gemma's newline-run words, never capped
        core |= {i for i in range(n) if types[i] == NORMAL and (len(tokens[i]) == 1 or not tokens[i].strip("\n"))
                 and charset.bytes_ok(token_bytes(tokens[i], False))}
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
    if max_vocab is not None and len(keep) > max_vocab:
        raise SystemExit(f"--keep-top {max_vocab:,} is below the {len(keep):,} tokens these languages need (special, "
                         f"byte, base-character and letter tokens): raise it or keep fewer languages")

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


SUPPRESS = "tokenizer.ggml.suppress_tokens"   # ids the sampler bans; dropped ones simply leave the list


def _token_id_fields(reader: gguf.GGUFReader) -> dict[str, object]:
    """Metadata holding token ids: tokenizer.ggml.*_token_id(s) and arch keys such as {arch}.decoder_start_token_id,
    {arch}.ple.eos_token_id or adapter token_ids_* lists. These tokens are always kept and renumbered."""
    return {k: f.contents() for k, f in reader.fields.items()
            if k.rsplit(".", 1)[-1].endswith("token_id") or "token_ids" in k}


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

    if dst.exists() and dst.resolve() == src.resolve():
        raise SystemExit("write the trimmed model to a new file: the original is still being read")
    arch = _field(reader, "general.architecture")
    tmp = dst.with_name(dst.name + ".part")
    per_token = {key for key, f in reader.fields.items() if key.startswith("tokenizer.ggml.")
                 and key != "tokenizer.ggml.merges" and f.types and f.types[0] == gguf.GGUFValueType.ARRAY
                 and len(f.data) == n}
    w = gguf.GGUFWriter(tmp, arch=arch, endianess=reader.endianess)
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
        elif key == SUPPRESS:
            val = [new_id[int(v)] for v in val if int(v) in new_id]
            if not val:
                continue
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
    tmp.replace(dst)                                    # never leave a half-written model under the final name
    report["out_gb"] = dst.stat().st_size / 2**30
    return report


def find_tool(name: str, bin_dir: str | None) -> Path:
    """A llama.cpp binary from --bin-dir, else from the llama.cpp build localllm downloaded."""
    if bin_dir:
        found = sorted(Path(bin_dir).rglob(name + (".exe" if sys.platform == "win32" else "")))
        if not found:
            raise SystemExit(f"{name} not found under {bin_dir}")
        return found[0]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from localllm import runtime
    server = runtime.find_server()
    return server.with_name(name + server.suffix)


def tokenize(tokenizer: Path, model: Path, text_file: Path) -> list[int]:
    out = subprocess.run([str(tokenizer), "-m", str(model), "-f", str(text_file), "--ids", "--log-disable",
                          "--no-bos", "--no-escape"], capture_output=True, text=True, errors="replace")
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("[")]
    if out.returncode or not lines:
        raise SystemExit(f"llama-tokenize failed on {text_file}:\n{out.stderr[-800:]}")
    return json.loads(lines[-1])


class Vocab:
    """A GGUF's tokenizer arrays, enough to map ids between two vocabularies and turn ids back into bytes."""

    def __init__(self, path: Path):
        r = gguf.GGUFReader(path)
        self.tokens = _field(r, "tokenizer.ggml.tokens")
        self.types = _field(r, "tokenizer.ggml.token_type")
        self.byte_level = byte_level(_field(r, "tokenizer.ggml.model", ""), _field(r, "tokenizer.ggml.pre", ""))

    def to(self, other: "Vocab") -> list[int | None]:
        """For each of our ids, the id of the same token in `other` (None if it was dropped)."""
        ids = {t: i for i, t in enumerate(other.tokens)}
        return [ids.get(t) for t in self.tokens]

    def pieces(self) -> list[bytes]:
        """Text bytes of every token; control tokens (<bos>, <|im_start|>, ...) add no text."""
        return [b"" if ty == CONTROL else self.detokenize([i]) for i, ty in enumerate(self.types)]

    def detokenize(self, ids: list[int]) -> bytes:
        """Bytes llama.cpp would print for these ids: byte-level text decoded, <0xXX> byte tokens, U+2581 -> space."""
        out = bytearray()
        for i in ids:
            t, ty = self.tokens[i], self.types[i]
            if ty == BYTE and not self.byte_level:
                out.append(int(t[3:5], 16))
            elif ty == NORMAL:
                out += token_bytes(t, self.byte_level)
            else:
                out += t.encode()
        return bytes(out)


def check(orig: Path, trimmed: Path, texts: Path, tokenizer: Path) -> list[dict]:
    """Per text file: does the trimmed model split it into exactly the same tokens, does it decode back to the same
    bytes, and how many more tokens does text in dropped scripts now take."""
    vo, vt = Vocab(orig), Vocab(trimmed)
    old_to_new = vo.to(vt)
    rows = []
    for txt in _texts(texts):
        a, b = tokenize(tokenizer, orig, txt), tokenize(tokenizer, trimmed, txt)
        mapped = [old_to_new[i] for i in a]
        raw = txt.read_bytes()
        rows.append({"text": txt.stem, "tokens": len(a), "tokens_trimmed": len(b),
                     "kept_pct": 100 * sum(m is not None for m in mapped) / max(len(a), 1),
                     "identical": mapped == b, "growth_pct": 100 * (len(b) / max(len(a), 1) - 1),
                     "roundtrip": vt.detokenize(b) == raw, "roundtrip_orig": vo.detokenize(a) == raw})
    return rows


def _texts(folder: Path) -> list[Path]:
    found = sorted(folder.glob("*.txt"))
    if not found:
        raise SystemExit(f"no .txt files in {folder}")
    return found


def read_logprobs(path: Path):
    """llama-perplexity --kl-divergence-base file: (n_ctx, n_vocab, tokens, rows). Each row is one scored position:
    2 floats (scale, min log-prob) + n_vocab uint16 q, log p = min + scale * q; q = 0 means 16+ nats below the top."""
    head = np.fromfile(path, dtype=np.uint8, count=20)
    if head[:8].tobytes() != b"_logits_":
        raise SystemExit(f"{path}: not a llama-perplexity logits file")
    n_ctx, n_vocab, n_chunk = (int(x) for x in head[8:20].view(np.int32))
    tokens = np.fromfile(path, dtype=np.int32, count=n_ctx * n_chunk, offset=20)
    nv = 2 * ((n_vocab + 1) // 2) + 4
    rows = np.memmap(path, dtype=np.uint16, mode="r", offset=20 + 4 * n_ctx * n_chunk)
    return n_ctx, n_vocab, tokens, rows[: rows.size // nv * nv].reshape(-1, nv)


def compare_logprobs(base_o: Path, base_t: Path, old_to_new: list[int | None], pieces_o: list[bytes] | None = None,
                     pieces_t: list[bytes] | None = None, block: int = 64) -> dict:
    """Same tokens on both sides: position by position, same most-likely token? How much probability did the original
    give dropped tokens (tokens 16+ nats below the top are stored without a value and count as 0, a lower bound)?
    Different tokens (a --keep-top trim, or a dropped script): compare at the token boundaries both runs share, with the
    same context start, whether the most likely next piece of text is the same (or one is a prefix of the other)."""
    n_ctx_o, nvo, tok_o, rows_o = read_logprobs(base_o)
    n_ctx_t, nvt, tok_t, rows_t = read_logprobs(base_t)
    mapped = np.array([-1 if old_to_new[i] is None else old_to_new[i] for i in tok_o])
    if len(tok_o) != len(tok_t) or not np.array_equal(mapped, tok_t):
        if pieces_o is None or pieces_t is None:
            return {"comparable": False}
        return _compare_aligned(n_ctx_o, tok_o, rows_o, pieces_o, n_ctx_t, tok_t, rows_t, pieces_t, block)
    new_to_old = np.full(nvt, -1)
    new_to_old[[n for n in old_to_new if n is not None]] = [o for o, n in enumerate(old_to_new) if n is not None]
    kept = np.zeros(nvo, bool)
    kept[new_to_old] = True
    same = top_kept = 0
    dropped_mass = []
    for s in range(0, len(rows_o), block):
        qo, qt = rows_o[s:s + block], rows_t[s:s + block]
        top_o = qo[:, 4:4 + nvo].argmax(1)
        top_t = new_to_old[qt[:, 4:4 + nvt].argmax(1)]
        same += int((top_o == top_t).sum())
        top_kept += int(kept[top_o].sum())
        hdr = np.ascontiguousarray(qo[:, :4]).view(np.float32)            # (scale, min log-prob) per row
        q = qo[:, 4:4 + nvo].astype(np.float32)
        p = np.exp(hdr[:, 1:2] + hdr[:, 0:1] * q)
        clipped = hdr[:, 0:1] * 65535 > 15.999                            # range hit 16 nats: q = 0 is "below that"
        p = np.where((q > 0) | ~clipped, p, 0.0)                          # count those as 0 (a lower bound)
        dropped_mass += list(p[:, ~kept].sum(1) / p.sum(1))
    n = len(rows_o)
    return {"comparable": True, "mode": "same tokens", "positions": n, "same_top1_pct": 100 * same / n,
            "orig_top1_kept_pct": 100 * top_kept / n, "dropped_mass_mean": float(np.mean(dropped_mass)),
            "dropped_mass_p99": float(np.percentile(dropped_mass, 99))}


def _boundaries(n_ctx: int, tokens: np.ndarray, pieces: list[bytes], n_rows: int) -> dict[tuple[int, int], int]:
    """(byte where the chunk's context starts, byte where the predicted token starts) -> row of the logits file.
    llama-perplexity scores the second half of each n_ctx chunk: n_ctx - 1 - n_ctx // 2 rows per chunk."""
    off = np.concatenate([[0], np.cumsum([len(pieces[i]) for i in tokens])])
    first, per = n_ctx // 2, n_ctx - 1 - n_ctx // 2
    out = {}
    for r in range(n_rows):
        c, j = divmod(r, per)
        out[(int(off[c * n_ctx]), int(off[c * n_ctx + first + j + 1]))] = r
    return out


def _compare_aligned(n_ctx_o, tok_o, rows_o, pieces_o, n_ctx_t, tok_t, rows_t, pieces_t, block: int) -> dict:
    bo = _boundaries(n_ctx_o, tok_o, pieces_o, len(rows_o))
    bt = _boundaries(n_ctx_t, tok_t, pieces_t, len(rows_t))
    pairs = sorted((r, bt[k]) for k, r in bo.items() if k in bt)
    if not pairs:
        return {"comparable": False}
    same = compatible = 0
    for s in range(0, len(pairs), block):
        ro, rt = (np.array(x) for x in zip(*pairs[s:s + block]))
        top_o = rows_o[ro][:, 4:4 + len(pieces_o)].argmax(1)
        top_t = rows_t[rt][:, 4:4 + len(pieces_t)].argmax(1)
        for a, b in zip(top_o, top_t):
            po, pt = pieces_o[a], pieces_t[b]
            same += po == pt
            compatible += bool(po and pt and (po.startswith(pt) or pt.startswith(po)))
    n = len(pairs)
    return {"comparable": True, "mode": "aligned", "positions": n, "coverage_pct": 100 * n / len(rows_o),
            "same_top1_pct": 100 * same / n, "compatible_pct": 100 * compatible / n}


def agree(orig: Path, trimmed: Path, texts: Path, perplexity: Path, ctx: int = 512, chunks: int = 4,
          extra: list[str] = ()) -> list[dict]:
    """Run both models over each text with llama-perplexity and compare their next-token predictions."""
    vo, vt = Vocab(orig), Vocab(trimmed)
    old_to_new, pieces_o, pieces_t = vo.to(vt), vo.pieces(), vt.pieces()
    rows = []
    with tempfile.TemporaryDirectory(prefix="trim-agree-") as d:
        for txt in _texts(texts):
            bases = []
            for tag, model in (("orig", orig), ("trim", trimmed)):
                base = Path(d, f"{txt.stem}-{tag}.bin")
                out = subprocess.run([str(perplexity), "-m", str(model), "-f", str(txt), "-c", str(ctx), "--chunks",
                                      str(chunks), "--kl-divergence-base", str(base), *extra],
                                     capture_output=True, text=True, errors="replace")
                if out.returncode or not base.exists():
                    raise SystemExit(f"llama-perplexity failed on {txt} ({tag}):\n{out.stderr[-1500:]}")
                bases.append(base)
            rows.append({"text": txt.stem, **compare_logprobs(*bases, old_to_new, pieces_o, pieces_t)})
            for b in bases:
                b.unlink()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trim", help="write a GGUF that keeps only the chosen languages' tokens")
    t.add_argument("model", type=Path)
    t.add_argument("out", type=Path, nargs="?")
    t.add_argument("--langs", required=True, help="language codes or script names, e.g. th,en or thai,latin")
    t.add_argument("--max-vocab", "--keep-top", dest="max_vocab", type=int,
                   help="also drop the rarest ASCII/symbol tokens down to about this many")
    t.add_argument("--keep-text", type=Path, action="append", default=[], help="never drop tokens this text uses")
    t.add_argument("--dry-run", action="store_true")
    c = sub.add_parser("check", help="compare tokenization of DIR/*.txt before and after trimming")
    g = sub.add_parser("agree", help="compare next-token predictions on DIR/*.txt (runs llama-perplexity)")
    for p in (c, g):
        p.add_argument("orig", type=Path)
        p.add_argument("trimmed", type=Path)
        p.add_argument("--texts", type=Path, required=True)
        p.add_argument("--require", default="", help="comma-separated text names (e.g. th,en) that must tokenize "
                       "identically (check) or agree on the top-1 token (agree); otherwise exit with an error")
    g.add_argument("--min-agree", type=float, default=99.0, help="percent same top-1 that --require texts need")
    g.add_argument("--ctx", type=int, default=512)
    g.add_argument("--chunks", type=int, default=4)
    g.add_argument("--ppl-args", default="", help='extra llama-perplexity arguments, e.g. "-ngl 99 -dev Vulkan0"')
    for p in (t, c, g):
        p.add_argument("--bin-dir", help="folder with llama.cpp binaries (default: the build localllm downloaded)")
    a = ap.parse_args()

    required = {x.strip() for x in getattr(a, "require", "").split(",") if x.strip()}
    if a.cmd == "check":
        bad = []
        rows = check(a.orig, a.trimmed, a.texts, find_tool("llama-tokenize", a.bin_dir))
        for r in rows:
            if (r["text"] in required and not r["identical"]) or (r["roundtrip_orig"] and not r["roundtrip"]):
                bad.append(r["text"])
            print(f"{r['text']:>10}: {r['tokens']:>7} -> {r['tokens_trimmed']:>7} tokens ({r['growth_pct']:+.1f}%), "
                  f"{r['kept_pct']:.1f}% of its tokens kept, {'identical' if r['identical'] else 'CHANGED'}, "
                  f"decodes back {'exactly' if r['roundtrip'] else 'DIFFERENTLY'}"
                  + ("" if r["roundtrip_orig"] else " (the original model doesn't round-trip it either)"))
        missing = required - {r["text"] for r in rows}
        if bad or missing:
            sys.exit(f"FAILED: {', '.join(bad)} changed or didn't decode back" if bad else f"no text for {missing}")
        return
    if a.cmd == "agree":
        rows = agree(a.orig, a.trimmed, a.texts, find_tool("llama-perplexity", a.bin_dir), a.ctx, a.chunks,
                     shlex.split(a.ppl_args))
        bad = [r["text"] for r in rows if r["text"] in required
               and not (r["comparable"] and r["same_top1_pct"] >= a.min_agree)]
        bad += sorted(required - {r["text"] for r in rows})
        for r in rows:
            if not r["comparable"]:
                print(f"{r['text']:>10}: tokenized too differently to compare (no shared scored positions)")
                continue
            if r["mode"] == "aligned":
                print(f"{r['text']:>10}: tokenized differently; at {r['positions']} shared token boundaries "
                      f"({r['coverage_pct']:.0f}% of positions) the most likely next text is the same "
                      f"{r['same_top1_pct']:.2f}%, compatible (one a prefix of the other) {r['compatible_pct']:.2f}%")
                continue
            print(f"{r['text']:>10}: same top-1 {r['same_top1_pct']:.2f}% of {r['positions']} positions, "
                  f"original's top-1 kept {r['orig_top1_kept_pct']:.2f}%, probability on dropped tokens "
                  f"mean {r['dropped_mass_mean']:.5f} / p99 {r['dropped_mass_p99']:.5f}")
        if bad:
            sys.exit(f"FAILED: top-1 agreement below {a.min_agree}% (or not measurable) for {', '.join(bad)}")
        return
    if a.out is None and not a.dry_run:
        ap.error("give OUT.gguf or --dry-run")
    extra: set[int] = set()
    for f in a.keep_text:
        extra |= set(tokenize(find_tool("llama-tokenize", a.bin_dir), a.model, f))
    langs = [s.strip() for s in a.langs.split(",") if s.strip()]
    r = trim(a.model, a.out, langs, a.max_vocab, extra, a.dry_run)
    print(f"vocab {r['n_vocab']:,} -> {r['n_kept']:,} tokens ({100 * r['n_kept'] / r['n_vocab']:.1f}%), "
          f"scripts: {', '.join(r['scripts'])}")
    print(f"  {r['n_core']:,} special/byte/forced + {r['n_letters']:,} with the languages' letters + "
          + (f"the most frequent of {r['n_ascii']:,} ASCII/symbol tokens up to {r['max_vocab']:,}" if r["max_vocab"]
             else f"{r['n_ascii']:,} ASCII/symbol tokens"))
    print(f"merges {r['merges']:,} -> {r['merges_kept']:,}")
    for name, (qt, before, after) in r["tensors"].items():
        print(f"  {name:<40} {qt:<6} {before:8.1f} -> {after:8.1f} MiB")
    print(f"file {r['file_gb']:.2f} GiB, {r['saved_gb']:.2f} GiB smaller"
          + (f" -> {r['out_gb']:.2f} GiB written to {a.out}" if "out_gb" in r else " (dry run)"))


if __name__ == "__main__":
    main()
