"""`localllm connect` writes agent configs, keeps everything else, and undoes cleanly (#75)."""
import json

import pytest

from localllm import connect, runtime

BASE = "http://127.0.0.1:8080"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALLLM_CONNECT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(runtime, "HOME", tmp_path / "localllm")
    return tmp_path / "home"


def test_hermes_replaces_only_the_model_block(home):
    cfg = home / ".hermes" / "config.yaml"
    cfg.parent.mkdir(parents=True)
    old = "model:\n  default: claude-sonnet\n  provider: anthropic\n\nterminal:\n  backend: local\n"
    cfg.write_text(old)
    connect.connect("hermes", BASE, "gemma4-26b-a4b-qat")
    new = cfg.read_text()
    assert 'base_url: "http://127.0.0.1:8080/v1"' in new and 'default: "gemma4-26b-a4b-qat"' in new
    assert "anthropic" not in new and new.endswith("terminal:\n  backend: local\n")
    assert f"context_length: {connect.AGENT_CTX}" in new
    assert "restored" in connect.undo("hermes") and cfg.read_text() == old


def test_opencode_keeps_other_providers(home):
    cfg = home / ".config" / "opencode" / "opencode.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({"provider": {"x": {"npm": "y"}}, "theme": "dark"}))
    connect.connect("opencode", BASE, "m")
    d = json.loads(cfg.read_text())
    assert d["provider"]["x"] == {"npm": "y"} and d["theme"] == "dark" and d["model"] == "localllm/m"
    assert d["provider"]["localllm"]["options"]["baseURL"] == BASE + "/v1"


def test_claude_code_sets_env_only(home):
    connect.connect("claude-code", BASE, "m")
    cfg = home / ".claude" / "settings.json"
    env = json.loads(cfg.read_text())["env"]
    assert env["ANTHROPIC_BASE_URL"] == BASE and env["ANTHROPIC_MODEL"] == "m"
    assert "removed" in connect.undo("claude-code") and not cfg.exists()      # it didn't exist before


def test_codex_adds_a_profile_and_stays_valid_toml(home):
    tomllib = pytest.importorskip("tomllib")
    cfg = home / ".codex" / "config.toml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text('model = "gpt-5"\n\n[mcp_servers.x]\ncommand = "y"\n')
    connect.connect("codex", BASE, "m")
    connect.connect("codex", BASE, "m2")                  # again: the block is replaced, not added twice
    text = cfg.read_text()
    d = tomllib.loads(text)
    assert d["model"] == "gpt-5" and d["mcp_servers"]["x"]["command"] == "y"
    assert d["profiles"]["localllm"]["model"] == "m2" and d["model_providers"]["localllm"]["wire_api"] == "responses"
    assert text.count(connect.MARK_START) == 1
    connect.undo("codex")
    assert cfg.read_text() == 'model = "gpt-5"\n\n[mcp_servers.x]\ncommand = "y"\n'   # the first original


def test_aider_flat_keys(home):
    cfg = home / ".aider.conf.yml"
    home.mkdir()
    cfg.write_text("model: gpt-4o\ndark-mode: true\n")
    connect.connect("aider", BASE, "m")
    assert cfg.read_text() == ("model: openai/m\ndark-mode: true\nopenai-api-base: http://127.0.0.1:8080/v1\n"
                               "openai-api-key: local\n")


def test_dry_run_and_cline_write_nothing(home):
    out = connect.connect("hermes", BASE, "m", dry_run=True)
    assert out.startswith("would write") and not (home / ".hermes").exists()
    assert "OpenAI Compatible" in connect.connect("cline", BASE, "m")
    assert connect.undo("opencode").startswith("nothing to undo")


def test_model_id_falls_back_when_nothing_runs():
    assert connect.model_id("http://127.0.0.1:9") == "localllm"


def test_obsidian_sets_copilot_default_and_flags_cloud_left(home, tmp_path):
    vault = tmp_path / "vault"
    data = vault / ".obsidian" / "plugins" / "copilot" / "data.json"
    data.parent.mkdir(parents=True)
    old = {"activeModels": [{"name": "gpt-4o", "provider": "openai", "enabled": True, "isBuiltIn": True}],
           "defaultModelKey": "gpt-4o|openai", "embeddingModelKey": "text-embedding-3-small|openai", "temperature": 0.1}
    data.write_text(json.dumps(old))
    msg = connect.connect("obsidian", BASE, "gemma", vault)
    d = json.loads(data.read_text(encoding="utf-8"))
    assert d["defaultModelKey"] == f"gemma|{connect.COPILOT_PROVIDER}" and d["temperature"] == 0.1
    assert d["activeModels"][0]["baseUrl"] == BASE + "/v1" and d["activeModels"][1]["name"] == "gpt-4o"
    assert "embeddings" in msg and "gpt-4o (openai)" in msg           # what still reaches the cloud is said
    connect.connect("obsidian", BASE, "gemma", vault)                   # again: no duplicate entry
    assert len(json.loads(data.read_text(encoding="utf-8"))["activeModels"]) == 2
    assert "restored" in connect.undo("obsidian", vault) and json.loads(data.read_text()) == old


def test_obsidian_needs_a_vault_with_the_plugin(home, tmp_path):
    with pytest.raises(SystemExit, match="not an Obsidian vault"):
        connect.connect("obsidian", BASE, "m", tmp_path)
    (tmp_path / ".obsidian").mkdir()
    with pytest.raises(SystemExit, match="Copilot community plugin"):
        connect.connect("obsidian", BASE, "m", tmp_path)
    with pytest.raises(SystemExit, match="give the vault"):
        connect.connect("obsidian", BASE, "m")
