from localllm import catalog, runtime


def test_pick_by_vram():
    assert catalog.pick(8) is None
    assert catalog.pick(12) == "qwen3.8-27b-iq2"
    assert catalog.pick(16, "th") == "gemma4-26b-a4b-qat"


def test_server_args_mtp_and_device():
    a = runtime.server_args(runtime.Path("m.gguf"), "Vulkan0", 8080, 4096, mtp=True)
    assert a[a.index("-dev") + 1] == "Vulkan0" and "draft-mtp" in a and a[a.index("--spec-draft-n-max") + 1] == "2"
    assert "draft-mtp" not in runtime.server_args(runtime.Path("m.gguf"), None, 8080, 4096, mtp=False)


def test_env_small_bar_fix_per_model_and_tune(monkeypatch):
    monkeypatch.delenv("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM", raising=False)
    assert runtime.server_env()["GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM"] == "1"            # unknown model: on
    assert "GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM" not in runtime.server_env(vk_fix=False)  # gemma-4: unset = off
    assert "GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM" not in runtime.server_env({"env": {}})   # tuned: exactly as measured
    assert runtime.server_env({"env": {"GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM": "1"}})["GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM"] == "1"
    monkeypatch.setenv("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM", "1")
    assert runtime.server_env(vk_fix=False)["GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM"] == "1"  # the user's choice wins


def test_best_device_skips_igpu():
    devs = [{"id": "Vulkan0", "name": "AMD Radeon RX 9070 XT", "total_gb": 15.9},
            {"id": "Vulkan1", "name": "Intel(R) UHD Graphics 770", "total_gb": 15.9}]
    assert runtime.best_device(devs)["id"] == "Vulkan0"


def test_bench_coverage():
    from localllm import bench
    assert bench.available("en") == ["global"]
    assert bench.available("th") == ["regional"]
    assert bench.available("ja") == ["global", "regional"]
    assert bench.available("xx") == []


def test_sizing_tiers_16gb():
    from localllm import sizing
    rows = {r["shape"]: r for r in sizing.tiers(15.9, 32, "AMD Radeon RX 9070 XT")}
    assert rows["8B"]["status"] == "fits" and rows["24-32B"]["quant"] == "Q3"
    assert rows["30B MoE (3B active)"]["status"] == "offload-moe"
    assert rows["70B"]["status"] == "too-big"
    assert sizing.bandwidth("NVIDIA GeForce RTX 4060 Ti") == 288 and sizing.bandwidth("Mystery GPU") is None


def test_pick_prefers_accuracy_then_speed_on_ties():
    assert catalog.pick(15.9, "zh") == "qwen3.8-27b-q3"      # 5.5 points better in Chinese
    assert catalog.pick(15.9, "en") == "gemma4-26b-a4b-qat"  # tie on accuracy, faster


def test_ram_estimate_components():
    from localllm import sizing
    est = sizing.ram_estimate_gb({"gb": 13.3}, [])
    assert est == {"embed_gb": 0.3, "host_gb": 0.5, "prompt_cache_gb": 8.0,
                   "checkpoints_gb": 0.0, "total_gb": 8.8}
    assert est["embed_gb"] < est["total_gb"]


def test_ram_estimate_uses_per_model_cpu_mapped():
    from localllm import sizing
    est = sizing.ram_estimate_gb({"gb": 12.2, "cpu_mapped_gb": 0.51}, [])
    assert est["embed_gb"] == 0.5  # measured field wins over the 2% heuristic (0.2)


def test_ram_estimate_reads_server_args():
    from localllm import sizing
    m = {"gb": 12.2, "cpu_mapped_gb": 0.51, "kv_kb_per_token": 34.8}
    # a 0.2-style low-RAM profile: small prompt cache, few checkpoints
    est = sizing.ram_estimate_gb(m, ["--cache-ram", "512", "--ctx-checkpoints", "2", "-c", "4096"])
    assert est["prompt_cache_gb"] == 0.5
    assert est["checkpoints_gb"] == round(2 * 4096 * 34.8 / 2**20, 1)
    assert est["total_gb"] == round(0.51 + 0.5 + 0.5 + 2 * 4096 * 34.8 / 2**20, 1)
    # --flag=value spelling and -np slots
    est2 = sizing.ram_estimate_gb(m, ["--cache-ram=1024", "-np", "2", "--ctx-checkpoints=1", "--ctx-size=8192"])
    assert est2["prompt_cache_gb"] == 1.0
    assert est2["checkpoints_gb"] == round(2 * 1 * 8192 * 34.8 / 2**20, 1)


def test_ram_estimate_counts_llama_defaults():
    # on big-RAM PCs localllm keeps llama.cpp's own defaults (8192 MiB cache, 32 checkpoints/slot);
    # the estimate must count them rather than under-report
    from localllm import catalog, runtime, sizing
    from localllm.bench import system_language
    key = catalog.pick(15.9, system_language())
    m = catalog.MODELS[key]
    args = runtime.server_args(runtime.Path(m["file"]), "Vulkan0", 8080, 8192, m["mtp"], ram_total_gb=128)
    est = sizing.ram_estimate_gb(m, args)
    assert est["prompt_cache_gb"] == 8.0
    assert est["checkpoints_gb"] == round(32 * m["checkpoint_gb"], 1)   # hybrid/SWA: state per checkpoint
    assert est["total_gb"] > 8.0  # no longer under-reports the big RAM users


def test_ram_estimate_matches_measured_low_ram_profile():
    # measured on RX 9070 XT / 32 GB, 30-turn chat: Qwen3.8 working set 2.37 GB with --cache-ram 1024 --ctx-checkpoints 4
    from localllm import catalog, runtime, sizing
    m = catalog.MODELS["qwen3.8-27b-q3"]
    est = sizing.ram_estimate_gb(m, runtime.server_args(runtime.Path("m"), None, 8080, 8192, True, ram_total_gb=31.8))
    assert abs(est["total_gb"] - 2.37) < 0.5


def test_ram_available_gb_is_positive():
    assert runtime.ram_available_gb() > 0


def test_doctor_shows_ram_line(monkeypatch, capsys):
    from localllm import catalog, cli, runtime, sizing
    from localllm.bench import system_language
    dev = {"id": "Vulkan0", "name": "AMD Radeon RX 9070 XT", "total_gb": 15.9}
    monkeypatch.setattr(cli, "_machine", lambda: (cli.Path("llama-server"), [dev], dev, 32.0))
    monkeypatch.setattr(cli.runtime, "ram_available_gb", lambda: 28.0)
    cli.cmd_doctor(None)
    out = capsys.readouterr().out
    key = catalog.pick(15.9, system_language())
    m = catalog.MODELS[key]
    ctx = sizing.context_tokens(15.9, m)
    launch = runtime.server_args(cli.Path(m["file"]), dev["id"], 8080, ctx, m["mtp"])
    est = sizing.ram_estimate_gb(m, launch)
    assert (f"uses ~{est['total_gb']:.1f} GB of system RAM: ~{est['embed_gb']:.1f} GB "
            f"embeddings/CPU-mapped + ~{est['prompt_cache_gb']:.1f} GB prompt cache + "
            f"~{est['checkpoints_gb']:.1f} GB ctx checkpoints + ~{est['host_gb']:.1f} GB host (est.)") in out
    assert f"leaves ~{max(0.0, 28.0 - est['total_gb']):.0f} GB of RAM free for other apps (est.)" in out


def test_low_ram_profile_sized_from_installed_ram():
    assert runtime.ram_profile(16) == (512, 2)
    assert runtime.ram_profile(31.8) == (1024, 4)
    assert runtime.ram_profile(64) == (2048, 8)
    assert runtime.ram_profile(128) is None
    a = runtime.server_args(runtime.Path("m.gguf"), None, 8080, 8192, mtp=False, ram_total_gb=31.8)
    assert a[a.index("--cache-ram") + 1] == "1024" and a[a.index("--ctx-checkpoints") + 1] == "4"
    assert "--cache-ram" not in runtime.server_args(runtime.Path("m.gguf"), None, 8080, 8192, mtp=False, ram_total_gb=128)


def test_ram_estimate_drops_with_low_ram_profile():
    from localllm import catalog, sizing
    m = catalog.MODELS["qwen3.8-27b-q3"]
    big = sizing.ram_estimate_gb(m, runtime.server_args(runtime.Path("m"), None, 8080, 8192, m["mtp"], ram_total_gb=128))
    small = sizing.ram_estimate_gb(m, runtime.server_args(runtime.Path("m"), None, 8080, 8192, m["mtp"], ram_total_gb=16))
    assert small["total_gb"] < big["total_gb"]


def test_gpu_spill_probe_is_safe_for_unknown_pid():
    v = runtime.gpu_spill_gb(999999)
    assert v is None or v == 0.0


def test_moe_offload_on_a_12gb_card():
    assert catalog.cpu_moe_layers("gemma4-26b-a4b-qat", 15.9, 20) == 0          # fits whole
    n = catalog.cpu_moe_layers("gemma4-26b-a4b-qat", 12.0, 20)
    assert n and 0 < n <= 30                                                     # some experts to RAM
    assert catalog.cpu_moe_layers("gemma4-26b-a4b-qat", 12.0, 1) is None        # not enough RAM
    assert catalog.cpu_moe_layers("qwen3.8-27b-q3", 12.0, 20) is None           # dense: no offload path
    a = runtime.server_args(runtime.Path("m"), None, 8080, 8192, False, ram_total_gb=32, cpu_moe=n)
    assert a[a.index("--n-cpu-moe") + 1] == str(n) and a[a.index("--load-mode") + 1] == "none"
    assert catalog.speed("gemma4-26b-a4b-qat", 12.0, 20) == 45                  # measured, not the placeholder


def test_math_answer_parsing():
    from localllm import bench
    assert bench.final_number("16 - 3 - 4 = 9 eggs, 9 * $2 = $18.\nAnswer: 18") == 18
    assert bench.final_number("So the total is **1,250** dollars.\nAnswer: **1,250**") == 1250
    assert bench.final_number("She has 3.5 kg left.") == 3.5
    assert bench.final_number("no idea") is None
    assert bench.available("th", ("math",)) == ["math"] and bench.available("ar", ("math",)) == []
    assert bench.available("th") == ["regional"]          # knowledge suites stay the default


def test_quality_floor_and_qat():
    assert catalog.below_floor("qwen3.8-27b-iq2") and not catalog.below_floor("qwen3.8-27b-q3")
    assert not catalog.below_floor("gemma4-26b-a4b-qat")


def test_chrf_matches_sacrebleu():
    from localllm import bench
    # reference values from sacrebleu 2.x CHRF(word_order=2).sentence_score
    assert bench.chrf("The cat is on the mat.", "The cat sat on the mat.") == 67.49
    assert bench.chrf("แมวนั่งบนเสื่อ", "แมวนั่งอยู่บนพรม") == 33.1
    assert bench.chrf("我今天很高兴见到你。", "今天见到你我很高兴。") == 25.79
    assert bench.chrf('(hello) "world", ok!', "hello world ok") == 38.43
    assert bench.chrf("same", "same") == 100.0 and bench.chrf("", "x") == 0.0
    assert bench.available("th", ("translate",)) == ["translate"] and bench.available("en", ("translate",)) == []


def test_code_answers_are_compared_by_value_without_running_them():
    from localllm import bench
    assert bench.code_answer("so it returns 4\n[ANSWER] [(4, 1), (2, 3)] [/ANSWER]") == "[(4, 1), (2, 3)]"
    assert bench.code_answer("[ANSWER]\n```python\n{1: None}\n```\n[/ANSWER]") == "{1: None}"
    assert bench.code_answer("no tag") is None
    assert bench.same_value("[(4,1),(2,3)]", "[(4, 1), (2, 3)]") and bench.same_value("'abc'", '"abc"')
    assert not bench.same_value("[1, 2]", "[2, 1]")
    assert not bench.same_value("__import__('os').system('echo hi')", "0")     # parsed as text, never run
    assert bench.available("en", ("code",)) == ["code"] and bench.available("th", ("code",)) == []


def test_codegen_program_assembly_and_sandbox_flags():
    from localllm import bench, sandbox
    reply = "Sure:\n```python\ndef add(a, b):\n    return a + b\n```\nDone."
    item = {"head": "from typing import List\n\ndef add(a, b):\n    ...", "test": "def check(c):\n    assert c(1, 2) == 3\n\ncheck(add)\n"}
    prog = bench.codegen_program(item, reply)
    assert prog.startswith("from typing import List\n") and "return a + b" in prog and prog.endswith("check(add)\n")
    cmd = sandbox.command("docker", "/w", 10)
    for flag in ("--network", "none", "--read-only", "ALL", "no-new-privileges", "--pids-limit", "/w:/work:ro"):
        assert flag in cmd
    assert cmd[cmd.index("--user") + 1] != "0"


def test_codegen_program_keeps_prompt_helpers_and_future_imports():
    import pytest
    from localllm import bench
    poly = "    return sum([c * math.pow(x, i) for i, c in enumerate(xs)])\n"
    head = f"import math\n\n\ndef poly(xs: list, x: float):\n{poly}\n\ndef find_zero(xs: list):\n    '''A zero.'''\n"
    test = (f"import math\n\ndef _poly(xs: list, x: float):\n{poly}\n"                 # laid out as in the HF row
            "def check(candidate):\n    inputs = [[[1, 2]], [[-6, 11, -6, 1]]]\n    results = [-0.5, 1.0]\n"
            "    for i, (inp, exp) in enumerate(zip(inputs, results)):\n"
            "        assert _poly(*candidate(*inp), inp) <= 0.0001\n\ncheck(find_zero)\n")
    reply = ("```python\nfrom __future__ import annotations\n\ndef find_zero(xs: list) -> float:\n"
             "    lo, hi = -100.0, 100.0\n    while hi - lo > 1e-10:\n        mid = (lo + hi) / 2\n"
             "        lo, hi = (lo, mid) if poly(xs, lo) * poly(xs, mid) <= 0 else (mid, hi)\n    return lo\n```")
    item = {"id": "HumanEval/32", "head": head, "test": test}
    prog = bench.codegen_program(item, reply)
    assert prog.startswith("from __future__ import annotations\nimport math\n") and prog.count("__future__") == 1
    exec(compile(prog, "p.py", "exec"), {"__name__": "p"})   # our own code, not a model's: it needs the prompt's poly()
    wrong = bench.codegen_program(item, "```python\ndef find_zero(xs):\n    return 0.25\n```")
    with pytest.raises(AssertionError):
        exec(compile(wrong, "p.py", "exec"), {"__name__": "p"})
    assert "assert _poly(*candidate(*inp), inp)" in bench.codegen_program({**item, "id": "HumanEval/31"}, reply)


def test_codegen_drops_only_the_mbpp_255_input_no_sandbox_can_hold():
    from localllm import bench
    test = ("inputs = [[['Red', 'Green', 'Blue'], 1], [['Dog', 'Cat', 'CatBird', 'Bird', 'Fish'], 77], [[84, 15], 2]]\n"
            "for i, inp in enumerate(inputs):\n    assertion(combinations_colors(*inp), ref_func(*inp), 0)\n")
    prog = bench.codegen_program({"id": "Mbpp/255", "head": "", "test": test}, "def combinations_colors(l, n): ...")
    assert "inputs = [[['Red', 'Green', 'Blue'], 1], [[84, 15], 2]]\n" in prog
    assert "77" in bench.codegen_program({"id": "Mbpp/256", "head": "", "test": test}, "")


def test_codegen_passes_only_when_its_tests_ran_to_the_end(monkeypatch, tmp_path):
    """sandbox.BOOT starts every program: as a module (not __main__, like EvalPlus's exec()), and a byte on the
    runner's pipe after its last line. Run here in-process on our own programs, not a model's."""
    import os
    import sys
    from localllm import bench, sandbox
    item = {"id": "Mbpp/1", "head": "", "test": "assert add(2, 3) == 5\n"}
    good = "def add(a, b):\n    return a + b\n"

    def reached_the_end(reply: str) -> bool:
        (tmp_path / "p.py").write_text(bench.codegen_program(item, f"```python\n{reply}```"), encoding="utf-8")
        r, w = os.pipe()
        monkeypatch.setattr(sys, "argv", ["-c", str(tmp_path / "p.py"), str(w)])
        try:
            exec(sandbox.BOOT, {"__name__": "__main__"})
        except (AssertionError, SystemExit):
            pass
        finally:
            os.close(w)
        with os.fdopen(r, "rb") as f:
            return f.read() == b"1"
    assert reached_the_end(good)
    assert reached_the_end(good + "if __name__ == '__main__':\n    raise SystemExit('the main block ran')\n")
    assert not reached_the_end("def add(a, b):\n    return a - b\n")
    assert not reached_the_end("def add(a, b):\n    return a - b\n\nif __name__ == '__main__':\n"
                               "    import unittest\n    unittest.main()\n")      # it no longer skips the tests
    assert not reached_the_end("def add(a, b):\n    return a - b\n\nimport sys\nsys.exit(0)\n")
