"""`localllm report`: one JSON with this PC's hardware, tuned settings and eval scores, ready to share (#39).

No usernames, paths or keys: home folders and the user name are scrubbed from every string, and paths keep only their
last part. Nothing is uploaded; the report is printed, saved under ~/.localllm/, and offered as a prefilled GitHub
issue (the "Share a result" form, .github/ISSUE_TEMPLATE/results.yml) the user submits or edits themselves.
`privacy_problems` is the same rule run on a report someone shared, before it is published (#101).
"""
from __future__ import annotations

import getpass
import json
import os
import platform
import re
import time
import urllib.parse
from pathlib import Path

from . import __version__, catalog, runtime

ISSUES = "https://github.com/phonology024/make-localllm-easier/issues/new"
FORM = "results.yml"    # .github/ISSUE_TEMPLATE/results.yml; its fields are prefilled by id (summary, report)
MAX_URL = 7000          # browsers and GitHub start cutting prefilled URLs somewhere past 8k
SCHEMA = 1
SECRET = re.compile(r"(key|token|secret|password|authorization)", re.I)
ABS_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|/|\\\\)")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
HOME_DIR = re.compile(r"(?:[A-Za-z]:[\\/]+Users[\\/]+|/home/|/Users/)[^\\/\s]+", re.I)   # a home folder inside text


def _user_bits() -> list[str]:
    bits = {str(Path.home())}
    try:
        bits.add(getpass.getuser())
    except Exception:  # noqa: BLE001 - no user name is fine
        pass
    return sorted((b for b in bits if b and len(b) >= 3), key=len, reverse=True)


def scrub(obj, bits: list[str] | None = None):
    """Copy of `obj` without anything personal: secret-looking keys dropped, home/user name replaced, absolute paths
    cut to their file or folder name."""
    bits = _user_bits() if bits is None else bits
    if isinstance(obj, dict):
        return {scrub(k, bits): scrub(v, bits) for k, v in obj.items() if not SECRET.search(str(k))}
    if isinstance(obj, list):
        return [scrub(v, bits) for v in obj]
    if isinstance(obj, str):
        if ABS_PATH.match(obj):
            obj = re.split(r"[\\/]", obj.rstrip("\\/"))[-1]
        for b in bits:
            if "/" in b or "\\" in b:                   # a home folder: anywhere in the string
                obj = obj.replace(b, "~")
            else:                                       # the user name: whole tokens only, so a name that is also
                obj = re.sub(rf"(?<![\w.-]){re.escape(b)}(?![\w.-])", "<user>", obj)   # a word ("user") spares
        return obj                                      # "username", "userland", "users"
    return obj


def _strings(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)
    elif isinstance(obj, str):
        yield obj


def privacy_problems(report: dict) -> list[str]:
    """Why a shared report can't be published as it is (empty when it can). Same rule as `scrub`, which `collect`
    already applies, plus e-mail addresses: a report that was edited by hand, or made by an old version, is checked
    again before it reaches the public leaderboard."""
    why = []
    if scrub(report, bits=[]) != report:
        why.append("has a secret-looking field (key, token, password ...) or an absolute path")
    texts = list(_strings(report))
    if any(HOME_DIR.search(t) for t in texts):
        why.append("names a home folder")
    if any(EMAIL.search(t) for t in texts):
        why.append("has an e-mail address")
    return why


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def driver() -> str | None:
    """GPU driver version, best effort: Windows (CIM), NVIDIA (nvidia-smi), AMD on Linux (amdgpu module)."""
    import subprocess
    cmds = [["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]]
    if os.name == "nt":
        cmds.insert(0, ["powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_VideoController | Sort-Object AdapterRAM -Descending | "
                        "Select-Object -First 1).DriverVersion"])
    for cmd in cmds:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            continue
        if out:
            return out.splitlines()[0].strip()
    amd = Path("/sys/module/amdgpu/version")
    return f"amdgpu {amd.read_text().strip()}" if amd.exists() else None


def vendor(gpu: str | None) -> str | None:
    """AMD / NVIDIA / Intel / Apple from a GPU name, None when unknown."""
    if not gpu:
        return None
    g = gpu.lower()
    for name, pat in (("NVIDIA", r"nvidia|geforce|rtx|gtx|quadro|tesla"), ("AMD", r"amd|radeon|instinct|rx"),
                      ("Intel", r"intel|arc|iris|uhd"), ("Apple", r"apple|m[1-9]")):
        if re.search(pat, g):
            return name
    return None


def os_family(os_name: str | None) -> str | None:
    o = (os_name or "").lower()
    return "Windows" if o.startswith("windows") else "macOS" if o.startswith(("darwin", "macos")) else         "Linux" if o.startswith("linux") else None


PLATFORMS = Path(__file__).with_name("platforms.json")    # written by tools/build_leaderboard.py


def platform_label(gpu: str | None, os_name: str | None = None) -> str:
    """How well this OS + GPU vendor is tested: by us, by community reports, or not at all."""
    os_name = os_name or platform.system()
    cell = f"{os_family(os_name) or os_name}/{vendor(gpu) or 'unknown GPU'}"
    m = _read(PLATFORMS).get(cell)
    if not m:
        return f"{cell}: untested - please run `localllm report` and share it"
    bits = (["tested by maintainers"] if m.get("maintainers") else []) +         ([f"{m['community_reports']} community report(s)"] if m.get("community_reports") else [])
    return f"{cell}: " + (", ".join(bits) if bits else "untested - please run `localllm report` and share it")


def installed_server() -> Path | None:
    """The llama-server localllm already downloaded; a report never downloads one."""
    if os.environ.get("LOCALLLM_LLAMA_SERVER"):
        return Path(os.environ["LOCALLLM_LLAMA_SERVER"])
    found = sorted((runtime.HOME / "llama.cpp").rglob(runtime.EXE)) if (runtime.HOME / "llama.cpp").exists() else []
    return found[-1] if found else None


PER_ITEM = ("outcomes", "items_sha")   # bench.paired(): kept locally and in runner lines, too long for a shared link


def lean_results(results: dict) -> dict:
    return {name: {k: ({f: x for f, x in v.items() if f not in PER_ITEM} if isinstance(v, dict) else v)
                   for k, v in res.items()} if isinstance(res, dict) else res
            for name, res in results.items()}


def collect(server: Path | None = None, devs: list[dict] | None = None, ram_gb: float | None = None,
            lang: str | None = None) -> dict:
    from .bench import system_language
    server = server if server is not None else installed_server()
    devs = devs if devs is not None else (runtime.devices(server) if server else [])
    dev = runtime.best_device(devs)
    ram_gb = ram_gb if ram_gb is not None else runtime.ram_gb()
    lang = lang or system_language()
    pick = catalog.pick(dev["total_gb"], lang, runtime.ram_available_gb()) if dev else None
    tune = {}
    for key, v in _read(runtime.HOME / "tune.json").items():          # "gpu|model file|llama.cpp build"
        gpu, model, build = (key.split("|") + ["", ""])[:3]
        tune.setdefault(model, {})[build] = {"gpu": gpu, **v}
    report = {
        "schema": SCHEMA, "localllm": __version__, "date": time.strftime("%Y-%m-%d"),
        "os": f"{platform.system()} {platform.release()}", "arch": platform.machine(),
        "cpu": platform.processor() or platform.machine(), "ram_gb": round(ram_gb, 1), "language": lang,
        "llama_cpp": server.parent.name if server else None,
        "gpu": dev["name"] if dev else None, "vram_gb": round(dev["total_gb"], 1) if dev else 0.0,
        "driver": driver() if dev else None,
        "devices": [{"name": d["name"], "vram_gb": round(d["total_gb"], 1)} for d in devs],
        "pick": pick, "tune": tune, "results": lean_results(_read(runtime.HOME / "results.json")),
    }
    host = os.environ.get("LOCALLLM_REPORT_HOST", "")
    if re.fullmatch(r"cloud(:[a-z0-9-]{1,30})?", host):          # e.g. cloud:runpod (tools/cloud_lab.sh sets it)
        report["host"] = host
    return scrub(report)


def summary(report: dict) -> str:
    lines = [f"**GPU:** {report['gpu'] or 'none'} ({report['vram_gb']} GB, driver {report.get('driver') or '-'})  "
             f"**RAM:** {report['ram_gb']} GB  "
             f"**OS:** {report['os']}  **llama.cpp:** {report['llama_cpp'] or '-'}  **localllm:** {report['localllm']}"]
    for model, builds in report["tune"].items():
        for build, v in builds.items():
            lines.append(f"- tune `{model}` ({build}): {v.get('tok_s')} tok/s, env {v.get('env') or '-'}, "
                         f"MTP {v.get('mtp') or 'off'}")
    for name, res in report["results"].items():
        scores = ", ".join(f"{k} {v.get('acc')}" for k, v in sorted(res.items()) if not k.startswith("_"))
        lines.append(f"- eval `{name}`: {scores or '-'}")
    return "\n".join(lines)


def issue_url(report: dict, saved: Path | None = None) -> str:
    """Prefilled "Share a result" form; when the JSON makes the link too long, the form asks for the saved file."""
    title = f"Results: {report['gpu'] or 'CPU only'} ({report['vram_gb']} GB), {report['os']}"

    def url(fields: dict) -> str:
        return f"{ISSUES}?{urllib.parse.urlencode({'template': FORM, 'title': title, **fields})}"

    full = url({"summary": summary(report), "report": json.dumps(report, ensure_ascii=False, indent=1)})
    if len(full) <= MAX_URL:
        return full
    ask = f"The full report is too long for a link: please paste `{saved.name if saved else 'report.json'}` " \
          f"(saved by `localllm report`) here."
    lines = summary(report).splitlines()
    while True:                                         # keep as much of the summary as fits
        short = url({"summary": "\n".join(lines), "report": ask})
        if len(short) <= MAX_URL or len(lines) <= 1:
            return short
        lines = lines[:-1]


def save(report: dict) -> Path:
    out = runtime.HOME / f"report-{report['date']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return out
