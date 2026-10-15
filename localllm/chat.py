"""Terminal chat against the local server (or any OpenAI-compatible URL).

  localllm chat                 start the recommended model if nothing is running, then chat
  /clear  /think  /stay  /status  /save FILE  /exit   (Ctrl+C stops the current answer, Ctrl+C again exits)
  /attach PATH  /docs  /detach N|all                   answer from your files, with sources (#113, docs.py)
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

HELP = ("commands: /clear new chat  /think show or hide reasoning  /stay keep the current model (no swaps)  "
        "/status memory, spill, context  /attach FILE|FOLDER answer from your files  /docs  /detach N|all  "
        "/save FILE  /exit   (Ctrl+C stops an answer)")
CHECK_EVERY = 5   # answers between background spill checks (one check takes ~1-2 s on Windows)


def _utf8_console() -> None:
    for s in (sys.stdin, sys.stdout):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def server_alive(url: str) -> bool:
    try:
        return b'"ok"' in urllib.request.urlopen(url.rstrip("/") + "/health", timeout=2).read()
    except OSError:
        return False


def stream(url: str, messages: list[dict], show_thinking: bool, out=sys.stdout, stay: bool = False) -> tuple[str, dict]:
    """Send the conversation, print the answer as it streams. Returns (answer text, timings).
    stay: ask a multi-model server to keep the loaded model instead of switching for this message."""
    body = {"messages": messages, "stream": True, "timings_per_token": False}
    headers = {"Content-Type": "application/json", **({"X-Localllm-Stay": "1"} if stay else {})}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers=headers)
    answer, timings, thinking_shown = [], {}, False
    with urllib.request.urlopen(req, timeout=3600) as r:
        model = r.headers.get("X-Localllm-Model")      # smart routing: which model answered, and why
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            timings = d.get("timings", timings)
            if not d.get("choices"):
                continue
            delta = d["choices"][0].get("delta", {})
            if delta.get("reasoning_content"):
                if show_thinking:
                    out.write(f"\x1b[2m{delta['reasoning_content']}\x1b[0m")
                elif not thinking_shown:
                    out.write("\x1b[2m(thinking...)\x1b[0m ")
                thinking_shown = True
                out.flush()
            if delta.get("content"):
                answer.append(delta["content"])
                out.write(delta["content"])
                out.flush()
    out.write("\n")
    return "".join(answer), {**timings, "model": model} if model else timings


def _get(url: str, timeout: float = 30) -> dict | None:
    try:
        return json.load(urllib.request.urlopen(url, timeout=timeout))
    except (OSError, ValueError):
        return None


def status_lines(url: str, last: dict) -> list[str]:
    """What `/status` prints: model, VRAM, RAM, spill (from /localllm/status), context used and last speed."""
    base = url.rstrip("/")
    st = _get(base + "/localllm/status")
    props = _get(base + "/props", timeout=5) or {}
    n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx") or props.get("n_ctx")
    lines = []
    if st is None or "loaded" not in st:
        lines.append("memory: not available from this server (it isn't localllm's)")
    elif not st.get("loaded"):
        lines.append("model: unloaded (idle); the next message loads it")
    else:
        gb = lambda v: f"{v:.1f} GB" if isinstance(v, (int, float)) else "n/a"  # noqa: E731
        lines.append(f"model: {st.get('running') or st.get('model')}")
        lines.append(f"VRAM: {gb(st.get('vram_gb'))}   system RAM: {gb(st.get('ram_gb'))}")
        spill = st.get("spill_gb")
        if isinstance(spill, (int, float)):
            bad = spill > st.get("spill_warn_gb", 1.5)
            lines.append(f"spilled into system RAM: {spill:.1f} GB" + (" - answers will be slower" if bad else " (ok)"))
    if st and st.get("savings"):
        lines.append(st["savings"])
    if st and st.get("download"):
        lines.append(st["download"])
    used = sum(last.get(k) or 0 for k in ("cache_n", "prompt_n", "predicted_n"))
    if used:
        lines.append(f"context: {used:,} tokens" + (f" of {n_ctx:,} ({100 * used / n_ctx:.0f}%)" if n_ctx else ""))
    if last.get("predicted_per_second"):
        lines.append(f"last answer: {last['predicted_per_second']:.0f} tok/s")
    return lines


def spill_warning(url: str) -> str | None:
    st = _get(url.rstrip("/") + "/localllm/status")
    spill = (st or {}).get("spill_gb")
    if isinstance(spill, (int, float)) and spill > st.get("spill_warn_gb", 1.5):
        return (f"warning: {spill:.1f} GB of the model has spilled from the GPU into system RAM since it started "
                "(long context?). Answers will be slower; /clear starts a shorter chat.")
    return None


def _check_spill(url: str, pending: list) -> None:
    w = spill_warning(url)
    if w:
        pending.append(w)


def repl(url: str, model_label: str = "", files=None) -> None:
    _utf8_console()
    print(f"[localllm] chatting with {model_label or url}. {HELP}")
    if files is None:
        from . import cli, docs
        files = docs.Attached(ask=cli.confirm)
    messages: list[dict] = []
    show_thinking = stay = False
    last: dict = {}
    answers, warned, pending = 0, False, []          # pending: a spill warning found by the background check
    while True:
        try:
            text = input("\n\x1b[1m> \x1b[0m").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return
        if not text:
            continue
        if text in ("/exit", "/quit"):
            return
        if text == "/clear":
            messages = []
            print("(new chat)")
            continue
        if text == "/think":
            show_thinking = not show_thinking
            print(f"(reasoning {'shown' if show_thinking else 'hidden'})")
            continue
        if text == "/stay":
            stay = not stay
            print("(staying on the current model: no switches)" if stay else "(the best model per message again)")
            continue
        if text == "/status":
            print("\n".join(f"\x1b[2m{line}\x1b[0m" for line in status_lines(url, last)))
            continue
        if text.startswith("/save"):
            path = Path(text[5:].strip() or f"chat-{time.strftime('%Y%m%d-%H%M%S')}.md")
            try:
                path.write_text("\n\n".join(f"**{m['role']}**: {m['content']}" for m in messages), encoding="utf-8")
                print(f"(saved {path.resolve()})")
            except OSError as e:
                print(f"(could not save: {e})")
            continue
        if text.startswith("/attach"):
            files.attach(text[7:])
            continue
        if text == "/docs":
            print(files.listing())
            continue
        if text.startswith("/detach"):
            files.detach(text[7:])
            continue
        if text.startswith("/"):
            print(HELP)
            continue
        messages.append({"role": "user", "content": text})
        sent, hits = files.question(text)          # only this turn carries the passages; history keeps the question
        if pending and not warned:
            print(f"\x1b[33m{pending[0]}\x1b[0m")
            warned = True
        try:
            answer, t = stream(url, messages[:-1] + [{"role": "user", "content": sent}], show_thinking, stay=stay)
        except KeyboardInterrupt:
            print("\n(stopped)")
            messages.pop()
            continue
        except OSError as e:
            print(f"(server error: {e})")
            messages.pop()
            continue
        messages.append({"role": "assistant", "content": answer})
        if hits:
            from .docs import sources_line
            print(f"\x1b[2m{sources_line(hits)}\x1b[0m")
        last, answers = t, answers + 1
        if not warned and not pending and answers % CHECK_EVERY == 0:   # spill can appear later (long context)
            threading.Thread(target=_check_spill, args=(url, pending), daemon=True).start()
        if t.get("predicted_per_second"):
            who = f"  [{t['model']}]" if t.get("model") else ""
            print(f"\x1b[2m{t['predicted_n']} tokens, {t['predicted_per_second']:.0f} tok/s{who}\x1b[0m")
