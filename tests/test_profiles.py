"""#90: two profiles. lean keeps today's footprint; max fills the GPU with context and gives the prompt cache more RAM."""
from pathlib import Path

from localllm import catalog, cli, runtime, sizing


def _flag(args, name):
    return args[args.index(name) + 1]


def test_lean_is_the_default_everywhere():
    ap = cli.build_parser()
    assert ap.parse_args([]).profile == "lean"
    assert ap.parse_args(["serve"]).profile == "lean" and ap.parse_args(["chat"]).profile == "lean"
    assert ap.parse_args(["serve", "--profile", "max"]).profile == "max"
    assert ap.parse_args([]).ctx is None                       # resolved per profile at launch


def test_lean_context_and_caches_are_unchanged():
    assert catalog.context_for("gemma4-26b-a4b-qat", 15.9, 20, "lean") == catalog.MIN_CTX
    args = runtime.server_args(Path("m.gguf"), "Vulkan0", 1, 8192, False, ram_total_gb=32)
    assert (_flag(args, "--cache-ram"), _flag(args, "--ctx-checkpoints")) == ("1024", "4")


def test_max_fills_the_gpu_and_grows_the_prompt_cache():
    ctx = catalog.context_for("gemma4-26b-a4b-qat", 15.9, 20, "max")
    assert ctx % 1024 == 0 and 64 * 1024 <= ctx <= catalog.MAX_PROFILE_CTX
    assert ctx <= sizing.context_tokens(15.9, catalog.MODELS["gemma4-26b-a4b-qat"])
    args = runtime.server_args(Path("m.gguf"), "Vulkan0", 1, ctx, False, ram_total_gb=32, profile="max")
    assert (_flag(args, "--cache-ram"), _flag(args, "--ctx-checkpoints")) == ("8192", "8")
    small = runtime.server_args(Path("m.gguf"), None, 1, ctx, False, ram_total_gb=16, profile="max")
    assert _flag(small, "--cache-ram") == "4096"               # a quarter of the RAM, never llama.cpp's full 8 GiB


def test_max_never_goes_below_lean():
    # a card that only just fits the model still gets the 8k minimum, not less
    assert catalog.context_for("qwen3.8-27b-q3", 10.0, 8, "max") == catalog.MIN_CTX
