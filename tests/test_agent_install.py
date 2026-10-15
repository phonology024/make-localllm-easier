"""The page agents install from and the skill they drive localllm with must name real commands and flags (#163)."""
import argparse
import re
from pathlib import Path

from localllm import cli

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "install-for-agents.md"
SKILL = ROOT / "skills" / "localllm" / "SKILL.md"
RAW = "https://raw.githubusercontent.com/phonology024/make-localllm-easier/main/"


def _parsers() -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    ap = cli.build_parser()
    sub = next(a for a in ap._actions if isinstance(a, argparse._SubParsersAction))
    return ap, dict(sub.choices)


def _commands(text: str) -> list[list[str]]:
    """Every `localllm ...` invocation in code spans and code blocks, split into words."""
    spans = re.findall(r"`([^`\n]*)`", text) + re.findall(r"^(localllm[^\n]*)$", text, re.M)
    found = []
    for s in spans:
        s = s.split("#", 1)[0]
        for m in re.finditer(r"\blocalllm((?: +[-\w.,<>=/:]+)*)", s):
            found.append(m[1].split())
    return found


def test_every_command_and_flag_on_the_page_and_in_the_skill_exists():
    top, subs = _parsers()
    top_flags = {o for a in top._actions for o in a.option_strings}
    checked = 0
    for f in (PAGE, SKILL):
        for words in _commands(f.read_text(encoding="utf-8")):
            p, flags = top, top_flags
            if words and not words[0].startswith("-"):
                assert words[0] in subs, f"{f.name}: `localllm {' '.join(words)}`: no command {words[0]!r}"
                p = subs[words[0]]
                flags = {o for a in p._actions for o in a.option_strings}
            for w in words:
                if w.startswith("--"):
                    assert w.split("=")[0] in flags, f"{f.name}: `localllm {' '.join(words)}`: no flag {w}"
            checked += 1
    assert checked >= 20


def test_skill_has_frontmatter_and_the_raw_urls_point_at_real_files():
    head = SKILL.read_text(encoding="utf-8").split("---", 2)[1]
    assert re.search(r"^name: localllm$", head, re.M) and re.search(r"^description: .{80,}$", head, re.M)
    for f in (PAGE, SKILL):
        for rel in re.findall(re.escape(RAW) + r"(\S+?\.md)", f.read_text(encoding="utf-8")):
            assert (ROOT / rel).is_file(), f"{f.name} links {RAW}{rel}, which is not in the repo"


def test_page_states_its_boundaries_and_readme_links_it():
    page = PAGE.read_text(encoding="utf-8")
    for rule in ("No `sudo`", "Look first, change nothing", "Leave the user's workspace alone", "--dry-run",
                 "Never pass `--yes`"):
        assert rule in page
    assert "docs/install-for-agents.md" in (ROOT / "README.md").read_text(encoding="utf-8")
