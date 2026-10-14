"""`localllm bugreport` (#119): what goes in the zip, and that keys, the LAN key and personal paths never do."""
import json
import zipfile

import pytest

from localllm import bugreport, cli, report, runtime

HOME_WIN = "C:\\Users\\somchai"
BITS = [HOME_WIN, "/home/somchai", "somchai"]


def test_api_keys_and_tokens_are_redacted():
    text = ("POST /v1/chat Authorization: Bearer abcdefgh12345678\n"
            "llama-server --api-key s3cretLANkey -m model.gguf\n"
            "openrouter sk-or-v1-0123456789abcdef  hf hf_ABCDEFGHIJKLMNOPQRS  gh ghp_ABCDEFGHIJKLMNOPQRSTUV\n"
            '{"api_key": "plain-value", "token": "t-1", "key_env": "OPENROUTER_API_KEY", "url": "https://x"}\n')
    out = bugreport.redact(text, secrets=[], bits=[])
    for gone in ("abcdefgh12345678", "s3cretLANkey", "sk-or-v1-0123456789abcdef", "hf_ABCDEFGHIJKLMNOPQRS",
                 "ghp_ABCDEFGHIJKLMNOPQRSTUV", "plain-value", '"t-1"'):
        assert gone not in out, gone
    assert '"key_env": "OPENROUTER_API_KEY"' in out and '"url": "https://x"' in out     # names of variables stay


def test_known_secrets_lan_key_and_route_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "HOME", tmp_path)
    (tmp_path / "route.json").write_text(json.dumps({"providers": {"or": {"key_env": "OPENROUTER_API_KEY"}}}))
    env = {"LOCALLLM_API_KEY": "lan-key-XYZ123", "OPENROUTER_API_KEY": "or-key-not-shaped-like-sk", "PATH": "/usr/bin"}
    secrets = bugreport.known_secrets(env)
    assert set(secrets) == {"lan-key-XYZ123", "or-key-not-shaped-like-sk"}
    out = bugreport.redact("key lan-key-XYZ123 and or-key-not-shaped-like-sk in a log", secrets, [])
    assert "lan-key" not in out and "or-key" not in out and out.count("<redacted>") == 2


def test_user_name_and_home_folder_in_paths():
    text = (f"model: {HOME_WIN}\\.localllm\\models\\x.gguf\n"
            "json: \"C:\\\\Users\\\\somchai\\\\.localllm\"\n"
            "posix: /home/somchai/.localllm/llama.cpp/b11476/llama-server\n"
            "fwd: C:/Users/somchai/AppData\n"
            "user somchai started it; somchai-pc and somchais stay\n")
    out = bugreport.redact(text, secrets=[], bits=BITS)
    assert "somchai\\" not in out and "somchai/" not in out and " somchai " not in out
    assert "~\\.localllm\\models\\x.gguf" in out and "~/.localllm/llama.cpp" in out and "user <user> started" in out
    assert "somchai-pc" in out and "somchais" in out                        # whole names only


def test_tail_reads_only_the_end(tmp_path):
    log = tmp_path / "llama-server.log"
    log.write_text("".join(f"line {i}\n" for i in range(5000)))
    t = bugreport.tail(log, 3)
    assert t == "line 4997\nline 4998\nline 4999\n" and bugreport.tail(tmp_path / "missing.log") is None


def test_collect_and_write(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "HOME", tmp_path)
    monkeypatch.setattr(report, "_user_bits", lambda: [str(tmp_path), "somchai"])
    monkeypatch.setattr(report, "installed_server", lambda: tmp_path / "llama.cpp" / "b11476" / "llama-server")
    monkeypatch.setattr(report, "driver", lambda: "32.0.21001")
    monkeypatch.setenv("LOCALLLM_API_KEY", "lan-key-XYZ123")
    (tmp_path / "llama-server.log").write_text(f"loading {tmp_path}/models/x.gguf\nkey lan-key-XYZ123\n")
    (tmp_path / "tune.json").write_text('{"gpu|x.gguf|b11476": {"tok_s": 50}}')

    def doctor():
        print(f"GPU AMD Radeon RX 9070 XT, models in {tmp_path}")
        raise RuntimeError("no llama-server")

    files = bugreport.collect(doctor)
    assert set(files) == {"doctor.txt", "versions.json", "llama-server.log", "tune.json"}
    assert "doctor failed: RuntimeError" in files["doctor.txt"] and json.loads(files["versions.json"])["llama_cpp"] == "b11476"
    everything = "".join(files.values())
    assert str(tmp_path) not in everything and "lan-key-XYZ123" not in everything
    out = bugreport.write(files, tmp_path)
    with zipfile.ZipFile(out) as z:
        assert sorted(z.namelist()) == sorted(files) and z.read("tune.json").decode() == files["tune.json"]


def test_cli_asks_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "HOME", tmp_path)
    monkeypatch.setattr(bugreport, "collect", lambda doctor: {"doctor.txt": "ok\n"})
    monkeypatch.setattr(cli, "confirm", lambda q: False)
    with pytest.raises(SystemExit, match="nothing written"):
        cli.cmd_bugreport(cli.build_parser().parse_args(["bugreport"]))
    assert not list(tmp_path.glob("bugreport-*.zip"))
    monkeypatch.setattr(cli, "confirm", lambda q: True)
    cli.cmd_bugreport(cli.build_parser().parse_args(["bugreport", "--show"]))
    assert len(list(tmp_path.glob("bugreport-*.zip"))) == 1
