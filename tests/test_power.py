"""GPU power readers, J/token and tune's energy option (#42). The readers are mocked: no GPU needed."""
import subprocess
import sys
from pathlib import Path

from localllm import power, tune


def test_nvidia_reader_parses_nvidia_smi(monkeypatch):
    monkeypatch.setattr(power.shutil, "which", lambda exe: "nvidia-smi")
    monkeypatch.setattr(power.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="231.45\n", stderr=""))
    assert power.nvidia_reader()() == 231.45
    monkeypatch.setattr(power.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="[N/A]\n", stderr=""))
    assert power.nvidia_reader()() is None


def test_amd_sysfs_reader_reads_microwatts(tmp_path):
    hw = tmp_path / "card1" / "device" / "hwmon" / "hwmon3"
    hw.mkdir(parents=True)
    (hw / "power1_average").write_text("187000000\n")
    assert power.amd_sysfs_reader(tmp_path)() == 187.0
    assert power.amd_sysfs_reader(tmp_path / "none") is None


def test_reader_picks_the_vendor(monkeypatch):
    monkeypatch.setattr(power, "nvidia_reader", lambda: "nv")
    monkeypatch.setattr(power, "amd_adl_reader", lambda: "adl")
    monkeypatch.setattr(power, "amd_sysfs_reader", lambda: "sysfs")
    assert power.reader("NVIDIA GeForce RTX 4090") == "nv"
    assert power.reader("AMD Radeon RX 9070 XT") == ("adl" if sys.platform == "win32" else "sysfs")
    assert power.reader("Apple M3 Max") is None


def test_sampler_and_joules_per_token():
    with power.Sampler(lambda: 200.0, interval=0.01) as s:
        import time
        time.sleep(0.05)
    assert s.watts == 200.0 and s.seconds > 0
    assert power.joules_per_token(200.0, 2.0, 100) == 4.0 and power.joules_per_token(None, 2.0, 100) is None


def test_prefer_energy_rule():
    base = {"tok_s": 100.0, "j_per_token": 2.0}
    assert power.prefer_energy(base, {"tok_s": 96.0, "j_per_token": 1.6})          # -20% energy, -4% speed
    assert not power.prefer_energy(base, {"tok_s": 90.0, "j_per_token": 1.6})      # 10% slower
    assert not power.prefer_energy(base, {"tok_s": 100.0, "j_per_token": 1.8})     # only -10% energy
    assert not power.prefer_energy({"tok_s": 100.0}, {"tok_s": 100.0, "j_per_token": 1.0})


def test_tune_keeps_energy_and_prefers_a_saver(monkeypatch, tmp_path):
    monkeypatch.setattr(tune, "CACHE", tmp_path / "tune.json")
    monkeypatch.setattr(tune, "detect_small_bar", lambda name: True)
    monkeypatch.setattr(power, "reader", lambda name: (lambda: 250.0))
    small_bar = {"tok_s": 97.0, "watts": 200.0, "j_per_token": 1.5}

    def stats(server, args, env, port=18234, power=None):
        if env.get("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM"):
            return small_bar
        return {"tok_s": 100.0, "watts": 250.0, "j_per_token": 2.0}
    monkeypatch.setattr(tune, "server_stats", stats)
    plain = tune.run(Path("s"), Path("m.gguf"), "AMD Radeon RX 9070 XT", [], False, log=lambda s: None)
    assert plain["env"] == {} and plain["j_per_token"] == 2.0 and plain["watts"] == 250.0
    saver = tune.run(Path("s"), Path("m.gguf"), "AMD Radeon RX 9070 XT", [], False, log=lambda s: None,
                     prefer_energy=True)
    assert saver["env"] == {"GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM": "1"} and saver["j_per_token"] == 1.5
