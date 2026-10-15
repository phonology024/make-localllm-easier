"""Chat with your files (#113): attach a PDF, DOCX, TXT or MD file; each answer is grounded in the passages that match
the question, and localllm lists those passages as sources under the answer.

  localllm chat      /attach PATH (a file or a folder)   /docs   /detach N|all
  localllm --ui      drag a file onto the chat, or the paperclip button

How: the text is split into chunks of at most CHUNK_CHARS characters (paragraphs, then sentences, then spaces), each
chunk is embedded by the same CPU-only multilingual-e5-small server the task router uses (taskclf.py: no VRAM, so it
never competes with the chat model), and the index is kept in ~/.localllm/docs/<sha256 of the file>/: meta.json,
chunks.jsonl (text + where it came from) and vectors.f32 (L2-normalised float32). The same file attached again is
not indexed again. A question is embedded the same way and the TOP_K closest chunks (exact cosine) go into the message.

TXT, MD and DOCX are read with the standard library. PDF needs pypdf (BSD-3): an installed one is used
(pip install make-localllm-easier[pdf]); otherwise the pinned wheel is downloaded once, after asking, into
~/.localllm/lib/, checked against its SHA-256 and imported from the zip. Scanned PDFs (images, no text layer) are
refused with a reason: there is no OCR. Nothing leaves the PC.
"""
from __future__ import annotations

import array
import hashlib
import io
import json
import re
import shutil
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from . import runtime

CHUNKER = 1                    # bump when chunking changes: indexes made by another version are rebuilt
CHUNK_CHARS = 700              # e5-small reads 512 tokens; a chunk the server still refuses is split in two
OVERLAP_CHARS = 160            # the last sentence of a chunk (if this short) starts the next one too
TOP_K = 5
MAX_FILE_BYTES = 64 * 2**20
TEXT_KINDS = (".txt", ".md", ".markdown", ".text")
KINDS = (*TEXT_KINDS, ".docx", ".pdf")
EMBED_TIMEOUT_S = 30

# pypdf for PDFs, pinned and hash-checked (#113); typing_extensions only on Python < 3.11 (pypdf's one requirement)
_PYPI = "https://files.pythonhosted.org/packages/"
PYPDF = ("pypdf-6.19.0-py3-none-any.whl",
         _PYPI + "3c/2c/c43c03eaf630435f023f1dc61ec4a4a78951ad5530a62c71cc89bde307b7/pypdf-6.19.0-py3-none-any.whl",
         "7e5d6e730e7dae87d560a2cee218b852f6498c8be61966f3cd02ead971e48d14")
TYPING_EXTENSIONS = ("typing_extensions-4.16.0-py3-none-any.whl",
                     _PYPI + "49/d3/b8441a820a491ddfc024b0b0cf0393375b75ea13866d9c66727e54c2fc80/"
                             "typing_extensions-4.16.0-py3-none-any.whl",
                     "481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8")
# Standard-library modules pypdf imports (dotted: a hidden import of "xml" doesn't bring xml.dom). The single-file binary (packaging/build_exe.py) bundles them as hidden imports,
# since nothing else in localllm imports e.g. uuid; tests/test_docs.py re-derives the list from the wheel when it can.
PYPDF_STDLIB = ("abc", "base64", "binascii", "bisect", "codecs", "collections", "collections.abc", "contextlib",
                "contextvars", "copy", "dataclasses", "datetime", "decimal", "enum", "functools", "hashlib", "io",
                "itertools", "json", "logging", "math", "operator", "os", "pathlib", "re", "secrets", "shutil", "string",
                "stringprep", "struct", "subprocess", "sys", "tempfile", "types", "typing", "unicodedata", "uuid",
                "warnings", "xml.dom.expatbuilder", "xml.dom.minidom", "xml.dom.xmlbuilder", "xml.parsers.expat", "zlib")
PDF_SUPPORT = "PDF support needs pypdf 6.19.0 (BSD-3, about 0.4 MB, SHA-256 checked) downloaded once"


class DocError(Exception):
    """A file that can't be used, with a reason a person can act on."""


class NeedsPdfSupport(DocError):
    """The pinned pypdf isn't here yet and the caller didn't agree to download it."""


# ---- reading -----------------------------------------------------------------------------------------------------

_CJK = r"぀-ヿ㐀-䶿一-鿿가-힯＀-￯"
_CJK_GAP = re.compile(rf"(?<=[{_CJK}])[ \t]+(?=[{_CJK}])")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize(text: str) -> str:
    """NFKC (PDF Arabic comes out as presentation forms, full-width digits), one space per run, and no spaces that a
    PDF put between CJK characters."""
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)                  # pypdf leaves NUL where it drops a Thai tone mark
    text = re.sub(r"[ \t ]+", " ", text)
    return _CJK_GAP.sub("", text).strip()


def _paragraphs(text: str, first_line: int = 1) -> list[tuple[str, str]]:
    """Blank-line separated paragraphs of a text file, each with the line it starts on."""
    out, start, buf = [], first_line, []
    for i, line in enumerate(text.split("\n"), first_line):
        if line.strip():
            if not buf:
                start = i
            buf.append(line.strip())
        elif buf:
            out.append((normalize(" ".join(buf)), f"line {start}"))
            buf = []
    if buf:
        out.append((normalize(" ".join(buf)), f"line {start}"))
    return [(t, loc) for t, loc in out if t]


def _decode(data: bytes) -> str:
    for bom, enc in ((b"\xef\xbb\xbf", "utf-8-sig"), (b"\xff\xfe", "utf-16"), (b"\xfe\xff", "utf-16")):
        if data.startswith(bom):
            return data.decode(enc, errors="replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _docx(data: bytes) -> list[tuple[str, str]]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            root = ElementTree.fromstring(z.read("word/document.xml"))
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError) as e:
        raise DocError(f"not a readable .docx file ({type(e).__name__})") from None
    out, n = [], 0
    for p in root.iter(_W + "p"):
        parts = []
        for el in p.iter():
            if el.tag == _W + "t" and el.text:
                parts.append(el.text)
            elif el.tag in (_W + "tab",):
                parts.append(" ")
            elif el.tag in (_W + "br", _W + "cr"):
                parts.append("\n")
        text = normalize("".join(parts))
        if text:
            n += 1
            out.append((text, f"paragraph {n}"))
    return out


def _pdf(data: bytes, allow_download: bool) -> list[tuple[str, str]]:
    pypdf = load_pypdf(allow_download)
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise DocError("this PDF is password-protected; open it, save an unprotected copy and attach that")
        pages = [(p.extract_text() or "") for p in reader.pages]
    except DocError:
        raise
    except Exception as e:                      # pypdf raises many kinds on damaged files
        raise DocError(f"could not read this PDF ({type(e).__name__}: {e})") from None
    out = []
    for n, text in enumerate(pages, 1):
        out += [(t, f"p.{n}") for t, _ in _paragraphs(text)]
    if not out:
        raise DocError("no text found in this PDF: it is probably scanned images, and localllm has no OCR")
    return out


def read_units(name: str, data: bytes, allow_download: bool = False) -> list[tuple[str, str]]:
    """(paragraph text, where it is: 'p.3' / 'line 40' / 'paragraph 12') for a file's bytes; its kind from the name."""
    kind = Path(name).suffix.lower()
    if kind in TEXT_KINDS:
        units = _paragraphs(_decode(data))
    elif kind == ".docx":
        units = _docx(data)
    elif kind == ".pdf":
        units = _pdf(data, allow_download)
    else:
        raise DocError(f"{kind or 'this kind of file'} isn't supported; attach {', '.join(KINDS)}")
    if not units:
        raise DocError("no text found in this file")
    return units


# ---- pypdf, pinned -----------------------------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wheels() -> list[tuple[str, str, str]]:
    return [PYPDF] + ([TYPING_EXTENSIONS] if sys.version_info < (3, 11) else [])


def pdf_support_ready() -> bool:
    try:
        import pypdf  # noqa: F401
        return True
    except ImportError:
        return all((runtime.HOME / "lib" / name).is_file() for name, _, _ in _wheels())


def load_pypdf(allow_download: bool):
    """pypdf as installed, else the pinned wheels in ~/.localllm/lib (downloaded only when allow_download), each checked
    against its SHA-256 before it is imported."""
    try:
        import pypdf
        return pypdf
    except ImportError:
        pass
    lib = runtime.HOME / "lib"
    for name, url, sha in _wheels():
        path = lib / name
        if not path.is_file():
            if not allow_download:
                raise NeedsPdfSupport(PDF_SUPPORT)
            from .cli import _download
            _download(url, path, name, timeout=30, sha256=sha)
        if _sha256(path) != sha:
            path.unlink()
            raise DocError(f"{name} did not match its pinned SHA-256 and was deleted; attach the PDF again")
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        import pypdf
    except ImportError as e:                    # e.g. a stdlib module a frozen build left out: say so, don't crash
        raise DocError(f"PDF support could not load ({e}); attach a DOCX, TXT or MD copy instead") from None
    return pypdf


# ---- chunking ----------------------------------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+|(?<=[。！？।؟])\s*")


def _split(text: str, limit: int) -> list[str]:
    """Sentences, each cut to at most `limit` characters (at spaces, then anywhere); chunk() packs them."""
    if len(text) <= limit // 3:
        return [text]
    pieces = []
    for sent in (s.strip() for s in _SENTENCE_END.split(text) if s and s.strip()):
        if len(sent) > limit:
            words, sent_parts, cur = sent.split(" "), [], ""
            for w in words:
                while len(w) > limit:                       # Thai and CJK runs without spaces
                    if cur:
                        sent_parts.append(cur)
                        cur = ""
                    sent_parts.append(w[:limit])
                    w = w[limit:]
                if cur and len(cur) + 1 + len(w) > limit:
                    sent_parts.append(cur)
                    cur = w
                else:
                    cur = f"{cur} {w}" if cur else w
            if cur:
                sent_parts.append(cur)
        else:
            sent_parts = [sent]
        pieces += sent_parts
    return pieces


def _last_sentence(text: str) -> str:
    sents = [s for s in _SENTENCE_END.split(text) if s and s.strip()]
    return sents[-1] if len(sents) > 1 and len(sents[-1]) <= OVERLAP_CHARS else ""


def chunk(units: list[tuple[str, str]], limit: int = CHUNK_CHARS) -> list[dict]:
    """Paragraphs packed into chunks of at most `limit` characters; a chunk keeps the place it starts at. The last
    short sentence of a chunk is repeated at the start of the next, so an answer that spans the cut is still found."""
    out, buf, loc, carried = [], "", None, False     # carried: buf holds only the previous chunk's last sentence
    for text, where in units:
        if buf and where.startswith("p.") and where != loc:     # a PDF chunk never spans pages: its "p.N" is exact
            if not carried:
                out.append({"text": buf, "loc": loc})
            buf, carried = "", False
        for piece in _split(text, limit):
            if buf and len(buf) + 1 + len(piece) > limit:
                if not carried:
                    out.append({"text": buf, "loc": loc})
                carry = "" if carried else _last_sentence(buf)
                buf = carry if carry and len(carry) + 1 + len(piece) <= limit else ""   # no room: no overlap
                carried = bool(buf)
            if not buf:
                loc = where
            buf = f"{buf} {piece}" if buf else piece
            carried = False
    if buf and not carried:
        out.append({"text": buf, "loc": loc})
    return out


# ---- embeddings --------------------------------------------------------------------------------------------------

class TooLong(Exception):
    pass


def _unit(vec: list[float]) -> array.array:
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return array.array("f", (v / norm for v in vec))


def http_embedder(url_fn):
    """embed(texts) -> unit vectors via POST {url}/v1/embeddings, one text per request (the e5 server has one 512-token
    slot). A text the server refuses as too long raises TooLong; the caller splits it."""
    def embed(texts: list[str]) -> list[array.array]:
        url, out = url_fn().rstrip("/"), []
        for t in texts:
            req = urllib.request.Request(url + "/v1/embeddings", data=json.dumps({"input": t}).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                d = json.load(urllib.request.urlopen(req, timeout=EMBED_TIMEOUT_S))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")
                if e.code in (400, 500) and re.search(r"too large|too long|exceed|context", body, re.I):
                    raise TooLong(body[:200]) from None
                raise DocError(f"the embedding server refused a passage ({e.code}): {body[:200]}") from None
            out.append(_unit(d["data"][0]["embedding"]))
        return out
    return embed


_CLF = None


def default_embedder():
    """The CPU e5 server (taskclf.Classifier), started on first use and shared by everything in this process."""
    from . import taskclf

    def url():
        global _CLF
        if _CLF is None:
            _CLF = taskclf.Classifier()
        try:
            return _CLF.embeddings_url()
        except taskclf.Unavailable as e:
            raise DocError(f"the embedding model isn't available: {e}") from None
    return http_embedder(url)


def embed_model_id() -> str:
    from . import taskclf
    return f"{taskclf.GGUF_FILE}@{taskclf.SHA256[taskclf.GGUF_URL][:12]}"


# ---- the library -------------------------------------------------------------------------------------------------

def _halves(text: str) -> list[str]:
    mid = len(text) // 2
    cut = text.rfind(" ", 0, mid + 1)
    cut = cut if cut > len(text) // 4 else mid
    return [p for p in (text[:cut].strip(), text[cut:].strip()) if p]


class Library:
    """The indexed files under ~/.localllm/docs. embed(texts) -> unit vectors; tests pass a fake one."""

    def __init__(self, embed=None, root: Path | None = None, model_id: str | None = None):
        self.root = Path(root) if root else runtime.HOME / "docs"
        self._embed = embed
        self.model_id = model_id or embed_model_id()
        self._vectors: dict[str, tuple[list[dict], array.array, int]] = {}
        self._lock = threading.Lock()              # the page's server indexes on request threads

    @property
    def embed(self):
        if self._embed is None:
            self._embed = default_embedder()
        return self._embed

    def _passages(self, texts: list[str]) -> list[array.array]:
        return self.embed(["passage: " + t for t in texts])

    def _meta_path(self, doc_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{16}", doc_id or ""):
            raise DocError("no such document")
        return self.root / doc_id / "meta.json"

    def get(self, doc_id: str) -> dict | None:
        try:
            m = json.loads(self._meta_path(doc_id).read_text(encoding="utf-8"))
        except (OSError, ValueError, DocError):
            return None
        return m if m.get("chunker") == CHUNKER and m.get("embed_model") == self.model_id else None

    def list(self) -> list[dict]:
        if not self.root.is_dir():
            return []
        metas = [self.get(p.name) for p in self.root.iterdir() if p.is_dir()]
        return sorted((m for m in metas if m), key=lambda m: m["added"])

    def add(self, name: str, data: bytes, path: str | None = None, allow_download: bool = False,
            progress=None) -> dict:
        """Index a file's bytes (or return the index it already has). progress(done, total) while embedding."""
        if len(data) > MAX_FILE_BYTES:
            raise DocError(f"{name} is {len(data) / 2**20:.0f} MB; files up to {MAX_FILE_BYTES // 2**20} MB are indexed")
        with self._lock:
            return self._add(name, data, path, allow_download, progress)

    def _add(self, name, data, path, allow_download, progress) -> dict:
        doc_id = hashlib.sha256(data).hexdigest()[:16]
        known = self.get(doc_id)
        if known:
            return known
        chunks = chunk(read_units(name, data, allow_download))
        todo, done, vecs = list(chunks), [], []
        while todo:
            c = todo.pop(0)
            try:
                vecs += self._passages([c["text"]])
                done.append(c)
            except TooLong:
                parts = _halves(c["text"])
                if len(parts) < 2:
                    raise DocError("a passage could not be embedded even after splitting") from None
                todo[:0] = [{"text": p, "loc": c["loc"]} for p in parts]
            if progress:
                progress(len(done), len(done) + len(todo))
        meta = {"id": doc_id, "name": Path(name).name, "path": path, "bytes": len(data), "kind": Path(name).suffix.lower(),
                "chunks": len(done), "chars": sum(len(c["text"]) for c in done), "added": time.time(),
                "chunker": CHUNKER, "embed_model": self.model_id, "dims": len(vecs[0]) if vecs else 0}
        tmp = self.root / f".{doc_id}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        (tmp / "chunks.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in done),
                                          encoding="utf-8")
        flat = array.array("f")
        for v in vecs:
            flat.extend(v)
        if sys.byteorder != "little":
            flat.byteswap()
        (tmp / "vectors.f32").write_bytes(flat.tobytes())
        (tmp / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        final = self.root / doc_id
        shutil.rmtree(final, ignore_errors=True)        # an index from another chunker or embedding model
        tmp.rename(final)
        return meta

    def add_path(self, path: Path, allow_download: bool = False, progress=None) -> list[dict]:
        """A file, or every supported file in a folder (not recursive)."""
        path = Path(path).expanduser()
        if path.is_dir():
            files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in KINDS)
            if not files:
                raise DocError(f"no {', '.join(KINDS)} files in {path}")
        elif path.is_file():
            files = [path]
        else:
            raise DocError(f"no such file: {path}")
        return [self.add(p.name, p.read_bytes(), str(p.resolve()), allow_download, progress) for p in files]

    def remove(self, doc_id: str) -> bool:
        """Forget a file's index (the file itself is never touched)."""
        folder = self._meta_path(doc_id).parent
        self._vectors.pop(doc_id, None)
        if not folder.is_dir():
            return False
        shutil.rmtree(folder)
        return True

    def _load(self, doc_id: str):
        if doc_id not in self._vectors:
            meta = self.get(doc_id)
            if not meta:
                raise DocError("no such document (or it was indexed by another version): attach it again")
            folder = self.root / doc_id
            chunks = [json.loads(l) for l in (folder / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l]
            vecs = array.array("f")
            vecs.frombytes((folder / "vectors.f32").read_bytes())
            if sys.byteorder != "little":
                vecs.byteswap()
            self._vectors[doc_id] = (chunks, vecs, meta["dims"])
        return self._vectors[doc_id]

    def search(self, query: str, doc_ids: list[str], k: int = TOP_K) -> list[dict]:
        """The k chunks closest to the question across these files, best first."""
        if not doc_ids or not query.strip():
            return []
        q = self.embed(["query: " + query.strip()])[0]
        scored = []
        for doc_id in doc_ids:
            chunks, vecs, dims = self._load(doc_id)
            name = self.get(doc_id)["name"]
            for i, c in enumerate(chunks):
                row = vecs[i * dims:(i + 1) * dims]
                scored.append((sum(a * b for a, b in zip(q, row)), doc_id, name, c))
        scored.sort(key=lambda s: -s[0])
        return [{"n": n, "score": round(s, 4), "doc": d, "name": name, "loc": c["loc"], "text": c["text"]}
                for n, (s, d, name, c) in enumerate(scored[:k], 1)]


class Attached:
    """The files attached to one chat (`localllm chat`): /attach, /docs, /detach, and the question with its passages.
    ask(question) -> bool is how a person agrees to the one-time PDF download."""

    def __init__(self, library: Library | None = None, ask=None, say=print):
        self.lib, self.ask, self.say = library, ask or (lambda q: False), say
        self.files: list[dict] = []

    @property
    def library(self) -> Library:
        if self.lib is None:
            self.lib = Library()
        return self.lib

    def attach(self, arg: str) -> None:
        path = Path(arg.strip().strip('"').strip("'"))
        if not arg.strip():
            return self.say(f"(give a file or a folder: /attach PATH; {', '.join(KINDS)})")

        def progress(done, total):
            if total > 20 and (done % 20 == 0 or done == total):
                self.say(f"(indexing {path.name}: {done}/{total} passages)")
        try:
            try:
                metas = self.library.add_path(path, progress=progress)
            except NeedsPdfSupport:
                if not self.ask(PDF_SUPPORT + " - download it now?"):
                    return self.say("(not attached: PDF support was not downloaded)")
                metas = self.library.add_path(path, allow_download=True, progress=progress)
        except DocError as e:
            return self.say(f"(not attached: {e})")
        for m in metas:
            if all(f["id"] != m["id"] for f in self.files):
                self.files.append(m)
            self.say(f"(attached {m['name']}: {m['chunks']} passages; answers now quote it - /docs lists, /detach removes)")

    def detach(self, arg: str) -> None:
        arg = arg.strip()
        if arg in ("", "all"):
            self.files = []
            return self.say("(no files attached)")
        if not arg.isdigit() or not 1 <= int(arg) <= len(self.files):
            return self.say("(/detach N, with N from /docs, or /detach all)")
        self.say(f"(detached {self.files.pop(int(arg) - 1)['name']})")

    def listing(self) -> str:
        if not self.files:
            return "(no files attached: /attach PATH)"
        return "\n".join(f"  {i}. {m['name']}  ({m['chunks']} passages)" for i, m in enumerate(self.files, 1))

    def question(self, text: str) -> tuple[str, list[dict]]:
        """(what to send for this turn, the passages it used)."""
        if not self.files:
            return text, []
        try:
            hits = self.library.search(text, [m["id"] for m in self.files])
        except DocError as e:
            self.say(f"(files not searched: {e})")
            return text, []
        return ask_with(text, hits), hits


# ---- what the model and the person see ---------------------------------------------------------------------------

def ask_with(question: str, hits: list[dict]) -> str:
    """The user message sent to the model: the passages, numbered, then the question. Only this turn carries them; the
    saved history keeps the question alone."""
    if not hits:
        return question
    passages = "\n\n".join(f"[{h['n']}] ({h['name']}, {h['loc']})\n{h['text']}" for h in hits)
    return ("Passages from the user's files:\n\n" + passages + "\n\n"
            "Answer the question below using these passages, and cite them as [1], [2]. If they don't contain the "
            "answer, say so plainly instead of guessing. Answer in the language of the question.\n\n"
            "Question: " + question)


def sources_line(hits: list[dict]) -> str:
    """Printed under every answer, from the search itself (so it shows even when the model doesn't cite)."""
    return "Sources: " + "  ·  ".join(f"[{h['n']}] {h['name']} {h['loc']}" for h in hits) if hits else ""
