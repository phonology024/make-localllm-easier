"""Record the README GIF from a real run (#81): `localllm doctor`, then a streamed answer with its real tok/s.

  python tools/record_demo.py                 # needs the picked model downloaded (`localllm` once) and the GPU free
  python tools/record_demo.py --prompt "Explain what a GPU does in two sentences."

Nothing is typed in by hand: the doctor text is the command's own output, the answer streams at the speed the server
produced it (timestamps recorded per chunk), and the speed shown is llama-server's own timing. The frames are drawn
with Pillow into docs/img/demo.gif (kept under 2 MB). Uses a running localllm on --url, or starts one.
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "img" / "demo.gif"
W, H, PAD, LINE = 960, 540, 18, 19
BG, FG, DIM, GREEN = (20, 22, 20), (225, 228, 222), (140, 146, 138), (61, 187, 132)
FONTS = ("consola.ttf", "DejaVuSansMono.ttf", "Menlo.ttc", "LiberationMono-Regular.ttf")


def font(size: int = 15):
    from PIL import ImageFont
    for f in FONTS:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def frame(lines: list[tuple[str, tuple]], title: str):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 28), fill=(36, 39, 36))
    for i, c in enumerate(((237, 106, 94), (245, 191, 79), (98, 197, 84))):
        d.ellipse((12 + i * 20, 9, 24 + i * 20, 21), fill=c)
    f = font()
    d.text((W // 2 - len(title) * 4, 7), title, fill=DIM, font=font(13))
    rows = (H - 40 - PAD) // LINE
    for i, (text, color) in enumerate(lines[-rows:]):
        d.text((PAD, 40 + i * LINE), text[:110], fill=color, font=f)
    return im


def wrap(text: str, width: int = 104) -> list[str]:
    out = []
    for para in text.split("\n"):
        while len(para) > width:
            cut = para.rfind(" ", 0, width)
            cut = cut if cut > 0 else width
            out.append(para[:cut])
            para = para[cut:].lstrip()
        out.append(para)
    return out


def doctor_text() -> str:
    from localllm import cli
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.cmd_doctor(None)
    return buf.getvalue().split("\nRun it:", 1)[0]


def stream(url: str, prompt: str) -> tuple[list[tuple[float, str]], dict, str]:
    body = {"messages": [{"role": "user", "content": prompt}], "stream": True, "max_tokens": 160,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, pieces, timings = time.time(), [], {}
    with urllib.request.urlopen(req, timeout=600) as r:
        model = r.headers.get("X-Localllm-Model") or ""
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            timings = d.get("timings", timings)
            piece = ((d.get("choices") or [{}])[0].get("delta") or {}).get("content") or ""
            if piece:
                pieces.append((time.time() - t0, piece))
    return pieces, timings, model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--prompt", default="In two sentences: why does a local LLM need fast GPU memory?")
    a = ap.parse_args()
    from localllm import chat
    proc = None
    if not chat.server_alive(a.url):
        proc = subprocess.Popen([sys.executable, "-m", "localllm", "serve", "--port", a.url.rsplit(":", 1)[1]])
        for _ in range(600):
            if chat.server_alive(a.url):
                break
            time.sleep(1)
    try:
        doc = doctor_text()
        pieces, timings, model = stream(a.url, a.prompt)
    finally:
        if proc:
            proc.terminate()
    frames, durations = [], []
    gpu = doc.splitlines()[0].split("  ")[0].removeprefix("GPU ").strip()
    title = f"localllm - recorded on {gpu}"

    def add(lines, ms):
        frames.append(frame(lines, title))
        durations.append(ms)

    shown: list[tuple[str, tuple]] = []
    cmd = "$ localllm doctor"
    for i in range(1, len(cmd) + 1, 2):
        add([(cmd[:i], GREEN)], 60)
    shown = [(cmd, GREEN)] + [(l, FG) for raw in doc.splitlines() for l in wrap(raw)]
    add(shown, 2600)
    cmd = "$ localllm chat"
    base = [(cmd, GREEN), ("", FG), ("you> " + a.prompt, FG), ("", FG)]
    add(base, 900)
    text, last_t, step = "", 0.0, 0.12                 # one frame per 120 ms of real streaming time
    for t, piece in pieces:
        text += piece
        if t - last_t >= step:
            add(base + [(l, FG) for l in wrap(text)], int((t - last_t) * 1000))
            last_t = t
    speed = timings.get("predicted_per_second")
    tail = f"[{model.split(' ')[0] or 'local model'}  {speed:.0f} tok/s]" if speed else ""
    add(base + [(l, FG) for l in wrap(text)] + [("", FG), (tail, DIM)], 3500)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    size = OUT.stat().st_size / 2**20
    print(f"{OUT} ({len(frames)} frames, {size:.1f} MB, {speed or 0:.0f} tok/s)")
    if size > 2:
        print("over 2 MB: shorten --prompt")


if __name__ == "__main__":
    main()
