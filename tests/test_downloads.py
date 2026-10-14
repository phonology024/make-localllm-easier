"""Downloads are checked against pinned SHA-256s (#92)."""
import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from localllm import catalog, cli, runtime, taskclf

BODY = b"GGUF" + bytes(range(256)) * 64


def _serve():
    class Files(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            rng = self.headers.get("Range")
            start = int(rng.split("=")[1].rstrip("-")) if rng else 0
            self.send_response(206 if rng else 200)
            self.send_header("Content-Length", str(len(BODY) - start))
            self.end_headers()
            self.wfile.write(BODY[start:])

    s = ThreadingHTTPServer(("127.0.0.1", 0), Files)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, f"http://127.0.0.1:{s.server_port}/m.gguf"


def test_download_checks_sha256(tmp_path):
    s, url = _serve()
    good = hashlib.sha256(BODY).hexdigest()
    cli._download(url, tmp_path / "ok.gguf", "ok", sha256=good)
    assert (tmp_path / "ok.gguf").read_bytes() == BODY
    with pytest.raises(cli.ChecksumError, match="expected 00"):
        cli._download(url, tmp_path / "bad.gguf", "bad", sha256="00" * 32)
    assert not (tmp_path / "bad.gguf").exists() and not (tmp_path / "bad.gguf.part").exists()
    s.shutdown(); s.server_close()


def test_resumed_download_hashes_the_part_already_on_disk(tmp_path):
    s, url = _serve()
    (tmp_path / "r.gguf.part").write_bytes(BODY[:1000])
    cli._download(url, tmp_path / "r.gguf", "r", sha256=hashlib.sha256(BODY).hexdigest())
    assert (tmp_path / "r.gguf").read_bytes() == BODY
    (tmp_path / "x.gguf.part").write_bytes(b"X" * 1000)          # a corrupt partial file is caught too
    with pytest.raises(cli.ChecksumError):
        cli._download(url, tmp_path / "x.gguf", "x", sha256=hashlib.sha256(BODY).hexdigest())
    s.shutdown(); s.server_close()


def test_every_catalog_file_is_pinned():
    for key, m in catalog.MODELS.items():
        assert len(m["rev"]) == 40 and len(m["sha256"]) == 64, key
        assert m["rev"] in cli._hf_url(m, m["file"]) and "/main/" not in cli._hf_url(m, m["file"])
        if m.get("mmproj"):
            assert len(m["mmproj_sha256"]) == 64, key
    assert "/resolve/main/" not in taskclf.GGUF_URL
    assert set(taskclf.SHA256) == {taskclf.GGUF_URL, taskclf.HEAD_URL}


def test_llama_cpp_asset_digest():
    for name, digest in runtime.LLAMA_SHA256.items():
        assert runtime.LLAMA_BUILD in name and len(digest) == 64
    assert runtime.expected_sha256({"name": "llama-b11476-bin-win-vulkan-x64.zip", "digest": "sha256:" + "f" * 64}) \
        == runtime.LLAMA_SHA256["llama-b11476-bin-win-vulkan-x64.zip"]          # the pin wins over the listing
    assert runtime.expected_sha256({"name": "llama-b2-bin-win-vulkan-x64.zip", "digest": "sha256:" + "ab" * 32}) \
        == "ab" * 32
    assert runtime.expected_sha256({"name": "llama-b2-bin-win-vulkan-x64.zip"}) is None
    runtime.check_sha256(b"abc", "a", hashlib.sha256(b"abc").hexdigest())
    with pytest.raises(RuntimeError, match="Nothing was unpacked"):
        runtime.check_sha256(b"abc", "a", "0" * 64)
    with pytest.raises(RuntimeError, match="no SHA-256"):
        runtime.check_sha256(b"abc", "a", None)


def test_dropped_connection_resumes_instead_of_failing_the_checksum(tmp_path, monkeypatch):
    """The link closing early used to be hashed as if complete: wrong SHA-256, .part deleted, 13 GB lost."""
    calls = []

    class Flaky(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            rng = self.headers.get("Range")
            start = int(rng.split("=")[1].rstrip("-")) if rng else 0
            calls.append(start)
            self.send_response(206 if rng else 200)
            self.send_header("Content-Length", str(len(BODY) - start))
            self.end_headers()
            self.wfile.write(BODY[start:start + 5000] if not rng else BODY[start:])   # 1st reply stops short

    s = ThreadingHTTPServer(("127.0.0.1", 0), Flaky)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{s.server_port}/m.gguf"
    with pytest.raises(cli.IncompleteDownload):
        cli._download(url, tmp_path / "m.gguf", "m", sha256=hashlib.sha256(BODY).hexdigest())
    assert (tmp_path / "m.gguf.part").stat().st_size == 5000
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    cli._fetch(url, tmp_path / "m.gguf", "m", hashlib.sha256(BODY).hexdigest())
    assert (tmp_path / "m.gguf").read_bytes() == BODY and calls[-1] == 5000
    s.shutdown(); s.server_close()
