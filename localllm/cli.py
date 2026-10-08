"""make-localllm-easier: one command, the best local AI your computer can run.

  localllm                  check this PC, pick the best model, download, start, open the chat page
  localllm doctor           what GPU/RAM you have and which model fits
  localllm list             every model we have measured
  localllm tune             measure the fastest llama.cpp settings for this PC once (kept only if >= 1.1x faster)
  localllm chat             chat in this terminal (starts the model if it isn't running)
  localllm route [--test P] routing config; compare one prompt local vs your cloud keys
  localllm serve [MODEL]    start an OpenAI-compatible server only (http://127.0.0.1:8080/v1)
  localllm eval             score a running server in English + your language (global and local exams)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

from . import __version__, catalog, runtime


def _say(msg: str) -> None:
    print(f"[localllm] {msg}", flush=True)


def _download(url: str, dest: Path, label: str) -> None:
    """Resumable download with a one-line progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    done = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    with urllib.request.urlopen(req) as r, open(tmp, "ab") as f:
        total = done + int(r.headers.get("Content-Length") or 0)
        t0, got = time.time(), 0
        while chunk := r.read(1 << 22):
            f.write(chunk); got += len(chunk)
            if total:
                pct = 100 * (done + got) / total
                speed = got / max(time.time() - t0, 1e-3) / 2**20
                print(f"\r  {label}: {pct:5.1f}% of {total / 2**30:.1f} GB  ({speed:.0f} MB/s)  ", end="", flush=True)
    print()
    tmp.replace(dest)


def _model_path(key: str) -> Path:
    m = catalog.MODELS[key]
    for d in filter(None, os.environ.get("LOCALLLM_MODELS", "").split(os.pathsep)):
        if (Path(d) / m["file"]).exists():
            return Path(d) / m["file"]
    dest = runtime.HOME / "models" / m["file"]
    if not dest.exists():
        _say(f"downloading {key} ({m['gb']} GB, one time) ...")
        _download(f"https://huggingface.co/{m['repo']}/resolve/main/{m['file']}", dest, m["file"])
    return dest


def _machine():
    server = runtime.find_server()
    devs = runtime.devices(server)
    return server, devs, runtime.best_device(devs), runtime.ram_gb()


def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _PoolProc:
    """Looks like a Popen to the commands: wait()/terminate() for a smart pool of lazy-loaded models."""
    def __init__(self, p, gw):
        self.pool, self.gateway = p, gw

    def wait(self):
        while True:
            time.sleep(3600)

    def terminate(self):
        self.pool.close()
        self.gateway.shutdown()


def _start(key: str | None, port: int, ctx: int, models: str | None = None, idle_min: float | None = None):
    server, _devs, dev, _ram = _machine()
    from .bench import system_language
    ram_free = runtime.ram_available_gb()
    if models:
        if not dev:
            sys.exit("[localllm] --models needs a GPU")
        keys = [k for k in catalog.MODELS if catalog.cpu_moe_layers(k, dev["total_gb"], ram_free) == 0
                and not catalog.below_floor(k)] if models == "auto" else models.split(",")
        bad = [k for k in keys if k not in catalog.MODELS]
        if bad or not keys:
            sys.exit(f"[localllm] unknown or unfitting models: {bad or models}. See `localllm list`.")
        from . import gateway, pool
        first = catalog.pick(dev["total_gb"], system_language(), ram_free, candidates=keys)
        _say(f"smart routing between {', '.join(keys)}")
        need = sum(catalog.MODELS[k]["gb"] + catalog.MODELS[k]["kv_kb_per_token"] * ctx / 2**20
                   + catalog.MODELS[k]["fixed_cache_gb"] for k in keys) + 1.0     # + ~1 GB driver/compute buffers
        resident = need <= dev["total_gb"]
        _say(f"all {len(keys)} models fit in VRAM together ({need:.1f} GB): no swaps" if resident else
             f"they need {need:.1f} GB together: one at a time, swapped only when another is clearly better")
        p = pool.Pool(keys, lambda k: _launch(k, server, dev, ctx, ram_free), dev["total_gb"], ram_free, first, resident,
                      files=None if resident else {k: _model_path(k) for k in keys},
                      idle_unload_s=pool.IDLE_UNLOAD_S if idle_min is None else idle_min * 60)
        return _PoolProc(p, gateway.serve(p, port=port, model_name="localllm-auto")), f"http://127.0.0.1:{port}"
    key = key or (catalog.pick(dev["total_gb"], system_language(), ram_free) if dev else None)
    if not key:
        sys.exit("[localllm] no measured model fits this GPU yet (need >= 10 GB VRAM). See `localllm list`.")
    proc, url = _launch(key, server, dev, ctx, ram_free)
    from . import gateway
    proc.gateway = gateway.serve(url, port=port, model_name=key)
    return proc, f"http://127.0.0.1:{port}"


def _warn_spill(pid: int) -> None:
    spill = runtime.gpu_spill_gb(pid)
    if spill and spill > runtime.SPILL_WARN_GB:
        _say(f"warning: {spill:.1f} GB of the model spilled from the GPU into system RAM - answers will be "
             "slower. Close other GPU-heavy apps or pick a smaller model (`localllm list`).")


def _launch(key: str, server, dev, ctx: int, ram_free: float) -> tuple[subprocess.Popen, str]:
    """Start llama-server for `key` on a private port; return once it answers /health."""
    cpu_moe = (catalog.cpu_moe_layers(key, dev["total_gb"], ram_free) or 0) if dev else 0
    if cpu_moe:
        _say(f"{key} doesn't fit the GPU whole: keeping the experts of {cpu_moe} layers in system RAM")
    model = _model_path(key)
    from . import tune
    tuned = tune.load().get(tune.machine_key(dev["name"] if dev else "cpu", model.name, server))
    mtp = (tuned["mtp"] if tuned else catalog.MODELS[key]["mtp"]) if catalog.MODELS[key]["mtp"] else False
    inner = _free_port()   # llama-server listens privately; the gateway on `port` speaks OpenAI/Anthropic/Ollama/Gemini
    args = runtime.server_args(model, dev["id"] if dev else None, inner, ctx, mtp, cpu_moe=cpu_moe)
    runtime.HOME.mkdir(parents=True, exist_ok=True)
    log = open(runtime.HOME / "llama-server.log", "ab")
    proc = subprocess.Popen([str(server), *args], env=runtime.server_env(tuned, catalog.MODELS[key].get("vk_fix", True)), stdout=log, stderr=subprocess.STDOUT)
    _say(f"loading {key} on {dev['name'] if dev else 'CPU'} ...")
    url = f"http://127.0.0.1:{inner}"
    for _ in range(3000):
        if proc.poll() is not None:
            sys.exit(f"[localllm] llama-server stopped (exit {proc.returncode}); log: {runtime.HOME / 'llama-server.log'}")
        try:
            if b'"ok"' in urllib.request.urlopen(url + "/health", timeout=2).read():
                threading.Thread(target=_warn_spill, args=(proc.pid,), daemon=True).start()   # ~1.5 s: off the path
                return proc, url
        except OSError:
            pass
        time.sleep(0.1)   # poll fast: every 100 ms counts in a model swap
    proc.kill()
    sys.exit("[localllm] model did not load within 5 minutes")


def cmd_run(a) -> None:
    proc, url = _start(a.model, a.port, a.ctx, getattr(a, "models", None), getattr(a, "idle_unload", None))
    _say(f"ready. chat: {url}   API: {url}/v1 (OpenAI), {url}/v1/messages (Anthropic), {url}/api (Ollama), "
         f"{url}/v1beta (Gemini)   Ctrl+C to stop")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


def cmd_chat(a) -> None:
    from . import chat
    url = a.url or f"http://127.0.0.1:{a.port}"
    if chat.server_alive(url):
        chat.repl(url)
        return
    if a.url:
        sys.exit(f"[localllm] nothing is answering at {a.url}")
    proc, url = _start(a.model, a.port, a.ctx, getattr(a, "models", None), getattr(a, "idle_unload", None))
    try:
        chat.repl(url, a.model or "")
    finally:
        proc.terminate()


def cmd_tune(a) -> None:
    from . import tune
    from .bench import system_language
    server, _devs, dev, _ram = _machine()
    if not dev:
        sys.exit("[localllm] no GPU found to tune")
    ram_free = runtime.ram_available_gb()
    key = a.model or catalog.pick(dev["total_gb"], system_language(), ram_free)
    model = _model_path(key)
    cpu_moe = catalog.cpu_moe_layers(key, dev["total_gb"], ram_free) or 0
    base = [a for a in runtime.server_args(model, dev["id"], 0, 4096, False, cpu_moe=cpu_moe) if True]
    i = base.index("--port"); del base[i:i + 2]
    _say(f"tuning {key} on {dev['name']} (a few minutes; each setting is kept only if it is >= 1.1x faster) ...")
    chosen = tune.run(server, model, dev["name"], base, bool(catalog.MODELS[key]["mtp"]), log=lambda s: _say("  " + s))
    _say(f"kept: env {chosen['env'] or 'none'}, MTP draft {chosen['mtp'] or 'off'} -> {chosen['tok_s']} tok/s "
         f"(saved to {tune.CACHE})")


def cmd_route(a) -> None:
    from . import chat, router
    cfg = router.load_config()
    if not a.test:
        print(f"config: {router.CONFIG}  (routing {'ON' if cfg.get('enabled') else 'OFF - everything stays local'})")
        print(f"cloud keys found: {', '.join(router.providers(cfg)) or 'none'}")
        print(router.__doc__.split("Example")[1] if "Example" in router.__doc__ else "")
        return
    url = f"http://127.0.0.1:{a.port}"
    proc = None
    if not chat.server_alive(url):
        proc, url = _start(None, a.port, 8192)
    try:
        for r in router.compare(a.test, url, cfg):
            if "error" in r:
                print(f"--- {r['target']}: error {r['error']}")
                continue
            cost = f", ${r['cost_usd']:.5f}" if r.get("cost_usd") is not None else (", free (local)" if r["target"] == "local" else "")
            print(f"--- {r['target']}: {r['seconds']} s, {r['tokens']} tokens{cost}\n{r['answer'][:600]}\n")
    finally:
        if proc:
            proc.terminate()


def cmd_serve(a) -> None:
    a.no_browser = True
    cmd_run(a)


ICON = {"fits": "OK  ", "low-bits": "WARN", "offload-moe": "SLOW", "offload-dense": "SLOW", "too-big": "NO  "}


def cmd_doctor(_a) -> None:
    from . import sizing
    from .bench import system_language
    server, devs, dev, ram = _machine()
    lang = system_language()
    gpu = dev["name"] if dev else "no GPU found"
    vram = dev["total_gb"] if dev else 0.0
    bw = sizing.bandwidth(gpu)
    print(f"GPU {gpu}  {vram:.1f} GB" + (f"  ({bw} GB/s)" if bw else "") + f"    RAM {ram:.0f} GB    language: {lang}")
    print("\nModel sizes for this PC (whole model on the GPU = fast):")
    for r in sizing.tiers(vram, ram, gpu):
        speed = f"~{r['tok_s']} tok/s (est.)" if r["tok_s"] else ""
        what = {"fits": f"{r['quant']} {r['gb']} GB  {speed}",
                "low-bits": f"only at {r['quant']} ({r['gb']} GB) - fits, but quality drops sharply below 3 bits",
                "offload-moe": f"{r['quant']} {r['gb']} GB with experts in RAM - works, ~10-25 tok/s",
                "offload-dense": f"{r['quant']} {r['gb']} GB with layers in RAM - very slow (< 5 tok/s)",
                "too-big": f"needs ~{r['gb']} GB - too big for this PC"}[r["status"]]
        print(f"  [{ICON[r['status']]}] {r['shape']:24} {what}")
    ram_avail = runtime.ram_available_gb()
    key = catalog.pick(vram, lang, ram_avail) if dev else None
    if not key:
        print("\nNo measured model fits this GPU yet. Run `localllm list`, or help by measuring one (`localllm eval`).")
        return
    m = catalog.MODELS[key]
    cpu_moe = catalog.cpu_moe_layers(key, vram, ram_avail) or 0
    in_ram = cpu_moe * m["moe"]["expert_gb_per_layer"] if cpu_moe else 0.0
    ctx = sizing.context_tokens(vram, {**m, "gb": m["gb"] - in_ram})
    print(f"\nBest measured model for you: {key}  ({m['note']})")
    if catalog.below_floor(key):
        print(f"  note: {m['bpw']} bits per weight is below the quality floor ({catalog.QUALITY_FLOOR_BPW}) - it fits, but"
              " expect noticeably weaker answers, especially outside English")
    if cpu_moe:
        print(f"  doesn't fit the GPU whole: experts of {cpu_moe} of {m['moe']['layers']} layers ({in_ram:.1f} GB) stay in"
              " system RAM - works, but answers are slower than on a bigger card")
    print("What it can do here:")
    shown = [t for t in sorted(m["scores"]) if t.split("/")[0] in (lang, "en")]
    others = sorted({t.split("/")[0] for t in m["scores"]} - {lang, "en"})
    for test in shown:
        acc = m["scores"][test]
        tl, suite = test.split("/")
        kind = {"global": "translated world-knowledge exam", "math": "grade-school math word problems",
                "translate": "translation to/from English",
                "code": "predicting what Python code returns",
                "codegen": "writing code that passes tests"}.get(
            suite, "real local school/licence exams")
        mark = "  <- your language" if tl == lang else ""
        unit = "chrF++ (0-100)" if suite == "translate" else "% correct"
        print(f"  {tl.upper():3} {kind:32} {acc:5.1f}{'' if unit[0] == '%' else ' '}{unit}{mark}")
    if others:
        print(f"  also measured in {', '.join(others)} (`localllm list`)")
    print(f"  holds ~{ctx // 1000}k tokens at once (~{ctx // sizing.TOKENS_PER_PAGE} pages of text) next to the model")
    launch = runtime.server_args(Path(m["file"]), dev["id"] if dev else None, 8080, ctx, m["mtp"], cpu_moe=cpu_moe)
    ram_est = sizing.ram_estimate_gb(m, launch)
    ram_est["total_gb"] = round(ram_est["total_gb"] + in_ram, 1)
    ram_free = max(0.0, ram_avail - ram_est["total_gb"])
    print(f"  uses ~{ram_est['total_gb']:.1f} GB of system RAM: ~{ram_est['embed_gb']:.1f} GB embeddings/CPU-mapped"
          f" + ~{ram_est['prompt_cache_gb']:.1f} GB prompt cache + ~{ram_est['checkpoints_gb']:.1f} GB ctx checkpoints"
          f" + ~{ram_est['host_gb']:.1f} GB host (est.)")
    print(f"  leaves ~{ram_free:.0f} GB of RAM free for other apps (est.)")
    same = bw == sizing.BANDWIDTH["rx 9070 xt"]
    est = m["tok_s_9070xt"] if same else (int(m["tok_s_9070xt"] * bw / 640) if bw else None)
    from . import tune
    tuned = tune.load().get(tune.machine_key(gpu, m["file"], server))
    if tuned:
        print(f"  answers at ~{tuned['tok_s']:.0f} tok/s (measured here by `localllm tune`)")
    else:
        if est and not cpu_moe:
            print(f"  answers at ~{est} tok/s" + ("" if same else " (estimated from memory bandwidth)"))
        print("  speed settings not tuned for this PC yet: `localllm tune` measures them once (a few minutes)")
    print(f"\nRun it: localllm        (llama.cpp: {server})")


def cmd_list(_a) -> None:
    print(f"{'model':22} {'weights':>8} {'tok/s*':>7}  scores")
    for k, m in sorted(catalog.MODELS.items(), key=lambda kv: -catalog.score(kv[0])):
        sc = "  ".join(f"{t} {v:.1f}" for t, v in m["scores"].items())
        print(f"{k:22} {m['gb']:6.1f}GB {m['tok_s_9070xt']:7}  {sc}")
    print("* decode speed on an RX 9070 XT 16 GB.  scores: accuracy % from `localllm eval` (lang/suite)")


def cmd_eval(a) -> None:
    from . import bench
    langs = a.langs.split(",") if a.langs else sorted({"en", bench.system_language()})
    _say(f"benchmarking {a.url} in: {', '.join(langs)}  (pick others with --langs ja,de,...)")
    bench.run(a.url, a.name, langs, a.limit, tuple(a.suites.split(",")))


def main() -> None:
    ap = argparse.ArgumentParser(prog="localllm", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=__version__)
    ap.add_argument("--model", choices=list(catalog.MODELS), help="override the automatic pick")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--ctx", type=int, default=8192, help="context length in tokens")
    ap.add_argument("--no-browser", action="store_true")
    idle = {"type": float, "metavar": "MIN", "help": "with --models: stop the models after MIN minutes without a "
            "request (default 15, 0 = never); the next message loads them again"}
    ap.add_argument("--idle-unload", **idle)
    ap.add_argument("--models", metavar="auto|A,B", help="smart routing: pick the best of these models per message, "
                    "lazy-loading one at a time (auto = every model that fits this GPU)")
    ap.set_defaults(fn=cmd_run)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    t = sub.add_parser("tune", help="measure the fastest settings for this PC once and remember them")
    t.add_argument("model", nargs="?", choices=list(catalog.MODELS)); t.set_defaults(fn=cmd_tune)
    c = sub.add_parser("chat"); c.add_argument("model", nargs="?", choices=list(catalog.MODELS))
    c.add_argument("--url", help="chat with an already running OpenAI-compatible server instead")
    c.add_argument("--port", type=int, default=8080); c.add_argument("--ctx", type=int, default=8192)
    c.add_argument("--models", metavar="auto|A,B"); c.add_argument("--idle-unload", **idle)
    c.set_defaults(fn=cmd_chat)
    r = sub.add_parser("route", help="show routing config, or --test a prompt local vs cloud")
    r.add_argument("--test", metavar="PROMPT"); r.add_argument("--port", type=int, default=8080)
    r.set_defaults(fn=cmd_route)
    s = sub.add_parser("serve"); s.add_argument("model", nargs="?", choices=list(catalog.MODELS))
    s.add_argument("--port", type=int, default=8080); s.add_argument("--ctx", type=int, default=8192)
    s.add_argument("--models", metavar="auto|A,B"); s.add_argument("--idle-unload", **idle)
    s.set_defaults(fn=cmd_serve)
    e = sub.add_parser("eval"); e.add_argument("--url", default="http://127.0.0.1:8080")
    e.add_argument("--name", default="model"); e.add_argument("--limit", type=int, default=0)
    e.add_argument("--langs", help="comma-separated ISO codes, default: en + this PC's language")
    e.add_argument("--suites", default="global,regional",
                   help="global,regional (knowledge), math (MGSM), translate (FLORES, chrF++), code (CRUXEval), codegen (EvalPlus, needs Docker)")
    e.set_defaults(fn=cmd_eval)
    a = ap.parse_args()
    if a.fn is cmd_serve:
        a.model = a.model or None
    a.fn(a)


if __name__ == "__main__":
    main()
