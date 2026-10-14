"""Fails when something docs/stability.md promises disappears (#98). Removing one on purpose: follow the deprecation
steps in that file, then update both."""
import argparse
from pathlib import Path

from localllm import catalog, cli, gateway

COMMANDS = {"doctor", "list", "tune", "chat", "serve", "route", "eval", "report", "connect"}
TOP_FLAGS = {"--model", "--port", "--ctx", "--no-browser", "--models", "--idle-unload", "--host", "--api-key",
             "--require-key-local", "--vision", "--profile", "--ui"}
SUB_FLAGS = {"eval": {"--url", "--name", "--limit", "--langs", "--suites"}, "chat": {"--url"},
             "report": {"--no-browser"}, "serve": {"--port", "--ctx", "--models", "--host", "--api-key", "--vision", "--profile"},
             "connect": {"--undo", "--dry-run", "--url", "--model-id", "--path"}}
ENDPOINTS = {"/v1/chat/completions", "/v1/completions", "/v1/models", "/v1/embeddings", "/v1/responses",
             "/v1/messages", "/v1/messages/count_tokens", "/health"}
HEADERS = {"X-Localllm-Route", "X-Localllm-Model", "X-Localllm-Stay"}
ENV = {"LOCALLLM_HOME", "LOCALLLM_MODELS", "LOCALLLM_LLAMA_SERVER", "LOCALLLM_LLAMA_BUILD", "LOCALLLM_API_KEY"}
MODEL_KEYS = {"qwen3.8-27b-q3", "gemma4-26b-a4b-qat", "qwen3.8-27b-iq2"}
SRC = "".join(p.read_text(encoding="utf-8") for p in (Path(cli.__file__).parent).glob("*.py"))


def _flags(p: argparse.ArgumentParser) -> set[str]:
    return {o for a in p._actions for o in a.option_strings}


def test_commands_and_flags_still_exist():
    ap = cli.build_parser()
    subs = next(a for a in ap._actions if isinstance(a, argparse._SubParsersAction)).choices
    assert COMMANDS <= set(subs)
    assert TOP_FLAGS <= _flags(ap)
    for cmd, flags in SUB_FLAGS.items():
        assert flags <= _flags(subs[cmd]), cmd


def test_endpoints_headers_env_and_model_keys_still_exist():
    assert ENDPOINTS <= set(gateway.PASSTHROUGH)
    for path in ("/api/chat", "/api/generate", "/api/tags", "/api/version"):
        assert f'"{path}"' in SRC
    assert gateway.GEMINI.match("/v1beta/models/x:generateContent")
    assert all(h in SRC for h in HEADERS) and all(e in SRC for e in ENV)
    assert MODEL_KEYS <= set(catalog.MODELS)
