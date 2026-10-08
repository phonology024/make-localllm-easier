"""Terminal chat against the local server (or any OpenAI-compatible URL).

  localllm chat                 start the recommended model if nothing is running, then chat
  /clear  /think  /stay  /save FILE  /exit   (Ctrl+C stops the current answer, Ctrl+C again at the prompt exits)
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

HELP = ("commands: /clear new chat  /think show or hide reasoning  /stay keep the current model (no swaps)  "
        "/save FILE  /exit   (Ctrl+C stops an answer)")


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


def repl(url: str, model_label: str = "") -> None:
    _utf8_console()
    print(f"[localllm] chatting with {model_label or url}. {HELP}")
    messages: list[dict] = []
    show_thinking = stay = False
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
        if text.startswith("/save"):
            path = Path(text[5:].strip() or f"chat-{time.strftime('%Y%m%d-%H%M%S')}.md")
            try:
                path.write_text("\n\n".join(f"**{m['role']}**: {m['content']}" for m in messages), encoding="utf-8")
                print(f"(saved {path.resolve()})")
            except OSError as e:
                print(f"(could not save: {e})")
            continue
        if text.startswith("/"):
            print(HELP)
            continue
        messages.append({"role": "user", "content": text})
        try:
            answer, t = stream(url, messages, show_thinking, stay=stay)
        except KeyboardInterrupt:
            print("\n(stopped)")
            messages.pop()
            continue
        except OSError as e:
            print(f"(server error: {e})")
            messages.pop()
            continue
        messages.append({"role": "assistant", "content": answer})
        if t.get("predicted_per_second"):
            who = f"  [{t['model']}]" if t.get("model") else ""
            print(f"\x1b[2m{t['predicted_n']} tokens, {t['predicted_per_second']:.0f} tok/s{who}\x1b[0m")
