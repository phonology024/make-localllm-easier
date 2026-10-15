"""Agent calls, cheap or hard (#97): the cost saver under Hermes / OpenCode / Aider sends the cheap ones to the local
model and, if the user turned it on, the hard ones to their own cloud key.

classify() looks only at the request (no model, microseconds):
  cheap  the agent's housekeeping - compacting or summarising the conversation, saving memory or a skill, a title,
         a retry after its own tool call was malformed, short lookups with a small answer budget
  hard   planning a task or editing several files: a tool-using call that starts a task, or offers edit/write tools
  chat   not an agent call (no tools, a short conversation): routed as before
These are starting rules from how the agents' requests look; logged sessions (#77, #96) will tune them.

failed_check() is the escalation test for a local answer to an agent (non-streaming only): an error status, an empty
answer, a tool call whose arguments are not JSON, or an answer cut off by the token limit before saying anything.
"""
from __future__ import annotations

import json
import re

HOUSEKEEPING = re.compile(
    r"\b(summari[sz]e|summary|compact|condense|compress)\b.{0,60}\b(conversation|history|context|session|above)\b|"
    r"\b(save|store|update|write)\b.{0,40}\b(memory|memories|skill|note to self)\b|"
    r"\b(title|name)\b.{0,30}\b(for|of) (this|the) (conversation|chat|session)\b", re.I | re.S)
TOOL_RETRY = re.compile(r"(invalid|malformed|could not parse|failed to parse|unparseable|not valid) "
                        r"(json|tool call|arguments|function call)|tool call (failed|error)", re.I)
EDIT_TOOLS = re.compile(r"(edit|write|patch|apply|replace|create)_?(file|files|patch|diff)?$|^(str_replace|bash)", re.I)
SHORT_ANSWER = 256          # max_tokens at or below this: a lookup, not a piece of work


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return ""


def _tool_names(body: dict) -> list[str]:
    names = []
    for t in body.get("tools") or []:
        if isinstance(t, dict):
            names.append(str((t.get("function") or {}).get("name") or t.get("name") or ""))
    return names


def classify(body: dict) -> tuple[str, str]:
    """("cheap" | "hard" | "chat", why) for an OpenAI chat or Anthropic messages request."""
    msgs = [m for m in body.get("messages") or [] if isinstance(m, dict)]
    system = _text(body.get("system")) + " ".join(_text(m.get("content")) for m in msgs if m.get("role") == "system")
    last = msgs[-1] if msgs else {}
    last_text = _text(last.get("content"))
    tools = _tool_names(body)
    budget = body.get("max_tokens") or body.get("max_completion_tokens")
    if HOUSEKEEPING.search(last_text) or HOUSEKEEPING.search(system[-2000:]):
        return "cheap", "housekeeping (summary, memory, title)"
    if TOOL_RETRY.search(last_text):
        return "cheap", "retry after a malformed tool call"
    if not tools and len(msgs) <= 2:
        return "chat", "not an agent call"
    if isinstance(budget, int) and budget <= SHORT_ANSWER and not tools:
        return "cheap", f"short lookup (max_tokens {budget})"
    users = [m for m in msgs if m.get("role") == "user"]
    if tools and len(users) <= 1 and last.get("role") == "user":
        return "hard", "starts a task (plan)"
    if any(EDIT_TOOLS.search(n) for n in tools) and last.get("role") == "user":
        return "hard", "new instruction with edit tools"
    if tools:
        return "cheap", "continues after a tool result"
    return "chat", "conversation"


def failed_check(api: str, status: int, payload: bytes) -> str | None:
    """Why a local answer to an agent should go to the cloud instead, or None if it is usable."""
    if status >= 500:
        return f"local server error {status}"
    if status != 200:
        return None                                    # 4xx: the request itself is wrong; the cloud won't fix it
    try:
        d = json.loads(payload or b"{}")
    except ValueError:
        return "local answer is not JSON"
    if api == "anthropic":
        blocks = d.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")
        uses = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
        if not text.strip() and not uses:
            return "empty answer" if d.get("stop_reason") != "max_tokens" else "cut off before answering"
        return None
    choice = (d.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    calls = msg.get("tool_calls") or []
    for c in calls:
        try:
            json.loads((c.get("function") or {}).get("arguments") or "{}")
        except ValueError:
            return "tool call arguments are not JSON"
    if not (msg.get("content") or "").strip() and not calls:
        return "cut off before answering" if choice.get("finish_reason") == "length" else "empty answer"
    return None
