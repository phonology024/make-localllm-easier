"""`localllm doctor --fix` (#111): each first-run failure is detected and explained, and only safe fixes are applied,
only on a typed yes. The expected texts are what the test PC printed (RX 9070 XT, Windows 11)."""
import json
import socket
from types import SimpleNamespace

import pytest

from localllm import catalog, cli, fix

DEV = {"id": "Vulkan0", "name": "AMD Radeon RX 9070 XT", "total_gb": 15.9, "free_gb": 15.1}


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(fix, "DEFAULTS", tmp_path / "defaults.json")
    return tmp_path


def test_no_gpu_without_vulkan_says_install_the_driver():
    p = fix.check_gpu([], loader=False)
    assert p.id == "no-gpu" and "Vulkan runtime is missing" in p.what and "graphics driver" in p.fix and not p.apply
    assert fix.check_gpu([DEV], loader=False) is None


def test_no_gpu_with_vulkan_offers_the_smallest_model():
    p = fix.check_gpu([], loader=True)
    small = min(catalog.MODELS, key=lambda k: catalog.MODELS[k]["gb"])
    assert "sees no GPU" in p.what and p.apply == {"model": small}


def test_small_bar_on_the_test_pc_suggests_bios_then_tune():
    p = fix.check_small_bar(DEV, small_bar=True, tuned=False)
    assert p.what == ("Resizable BAR looks off on the AMD Radeon RX 9070 XT: llama.cpp's Vulkan buffers then sit in "
                      "system RAM, which makes some models 1.6-2.7x slower.")
    assert p.run == ["tune"] and "Resizable BAR" in p.fix
    assert fix.check_small_bar(DEV, small_bar=True, tuned=True) is None          # tune already measured it
    assert fix.check_small_bar({**DEV, "name": "NVIDIA GeForce RTX 4090"}, True, False) is None
    assert fix.check_small_bar(DEV, small_bar=None, tuned=False) is None          # unknown: say nothing


def test_vram_taken_by_other_apps_names_them_and_offers_a_model_that_fits(monkeypatch):
    monkeypatch.setattr(fix.runtime, "ram_available_gb", lambda: 20.0)
    busy = {**DEV, "free_gb": 10.5}
    p = fix.check_vram(busy, "gemma4-26b-a4b-qat", [("QuantFleet", 2.2), ("dwm", 2.5)])
    assert p.id == "vram-busy" and "only 10.5 GB of 15.9 GB is free" in p.what
    assert "QuantFleet (2.2 GB)" in p.fix and p.apply["model"] == "qwen3.8-27b-iq2"
    assert fix.check_vram(DEV, "gemma4-26b-a4b-qat", []) is None


def test_low_ram_keeps_the_lean_profile():
    p = fix.check_ram("gemma4-26b-a4b-qat", ram_free=1.2, need_gb=2.2)
    assert "Only 1.2 GB of system RAM is free" in p.what and p.apply == {"profile": "lean"}
    assert fix.check_ram("gemma4-26b-a4b-qat", ram_free=20, need_gb=2.2) is None


def test_port_in_use_is_reproduced_with_a_real_socket():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        assert not fix.port_free(port)
        p = fix.check_port(port, False, "python", is_localllm=False)
    assert p.id == "port-busy" and f"Port {port} is taken: python is already listening" in p.what
    assert p.apply["port"] > port and fix.check_port(port, False, "x", is_localllm=True) is None


def test_garbled_answers_are_recognised():
    assert fix.check_garbled("OK", None) is None and fix.check_garbled(None, None) is None
    p = fix.check_garbled("<start_of_turn>model\nOK<end_of_turn>", "gemma")
    assert p.id == "garbled" and "raw chat-template tokens" in p.what and "--jinja" in p.fix
    assert "the same word over and over" in fix.check_garbled("ok " * 12, None).what


def test_defaults_fill_only_flags_that_were_not_typed():
    fix.save_defaults({"port": 8085, "model": "qwen3.8-27b-iq2", "evil": 1})
    assert json.loads(fix.DEFAULTS.read_text()) == {"port": 8085, "model": "qwen3.8-27b-iq2"}
    a = SimpleNamespace(port=8080, model=None, profile="lean")
    fix.apply_defaults(a, given=set())
    assert (a.port, a.model) == (8085, "qwen3.8-27b-iq2")
    b = SimpleNamespace(port=9000, model=None, profile="lean")
    fix.apply_defaults(b, given={"port"})
    assert b.port == 9000                                                          # typed flags win


def test_report_text():
    assert fix.report([]) == "No first-run problems found."
    text = fix.report([fix.Problem("port-busy", "Port 8080 is taken.", "Use another port.")])
    assert text == "1. [port-busy] Port 8080 is taken.\n   fix: Use another port."


def test_a_fix_needs_a_typed_yes(monkeypatch):
    """`< NUL` counts as a terminal on Windows: an empty or missing answer must never apply a fix."""
    class TTY:
        def isatty(self):
            return True
    monkeypatch.setattr(cli.sys, "stdin", TTY())
    monkeypatch.setattr("builtins.input", lambda q: (_ for _ in ()).throw(EOFError()))
    assert cli._ask_yes("x?") is False
    monkeypatch.setattr("builtins.input", lambda q: "")
    assert cli._ask_yes("x?") is False
    monkeypatch.setattr("builtins.input", lambda q: "y")
    assert cli._ask_yes("x?") is True


def test_doctor_fix_applies_nothing_without_consent(monkeypatch, capsys):
    monkeypatch.setattr(cli, "ASSUME_YES", False)
    monkeypatch.setattr(cli, "_machine", lambda: (None, [], None, 32.0))
    monkeypatch.setattr(cli.runtime, "ram_available_gb", lambda: 20.0)
    monkeypatch.setattr(fix, "vulkan_loader_present", lambda: True)
    monkeypatch.setattr(fix, "port_free", lambda port: True)
    monkeypatch.setattr(cli, "_ask_yes", lambda q: False)
    import localllm.chat as chat
    monkeypatch.setattr(chat, "_get", lambda url, timeout=10: None)
    probs = cli._doctor_fix(SimpleNamespace(port=8080))
    out = capsys.readouterr().out
    assert [p.id for p in probs] == ["no-gpu"] and "not applied ([no-gpu])" in out and not fix.DEFAULTS.exists()


def test_vram_taken_with_nothing_smaller_that_fits(monkeypatch):
    monkeypatch.setattr(fix.runtime, "ram_available_gb", lambda: 20.0)
    p = fix.check_vram({**DEV, "free_gb": 6.0}, "gemma4-26b-a4b-qat", [])
    assert p.apply == {} and "Close what holds VRAM - other apps" in p.fix
