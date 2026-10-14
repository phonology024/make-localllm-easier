"""Build the static website into site/public/ (#74). Standard library only.

  python site/build.py                       # then open site/public/index.html, or serve the folder
  SITE_URL=https://example.pages.dev python site/build.py

Numbers come from the same places the README uses: localllm/catalog.py (scores, speeds), data/leaderboard.json
(tools/build_leaderboard.py) and tools/footprint_results.jsonl (tools/headline.py). Nothing is typed in twice.
Deployed to Cloudflare Pages (the default *.pages.dev address): see site/README.md.
"""
from __future__ import annotations

import html
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from localllm import __version__, catalog  # noqa: E402
import headline  # noqa: E402

SITE = Path(__file__).resolve().parent
OUT = SITE / "public"
URL = os.environ.get("SITE_URL", "https://make-localllm-easier.pages.dev").rstrip("/")
GH = "https://github.com/phonology024/make-localllm-easier"
PYPI = "https://pypi.org/project/make-localllm-easier/"
ISSUES = GH + "/issues"
LANG_NAMES = {"en": "English", "th": "Thai", "zh": "Chinese", "ja": "Japanese", "es": "Spanish", "hi": "Hindi",
              "ar": "Arabic", "de": "German", "fr": "French", "ko": "Korean", "id": "Indonesian", "vi": "Vietnamese",
              "pt": "Portuguese", "ru": "Russian", "it": "Italian", "tr": "Turkish", "sw": "Swahili", "bn": "Bengali"}

NAV = [("Features", "/features/"), ("How it works", "/how-it-works/"), ("Use cases", "/use-cases/"),
       ("Compare", "/compare/"), ("Leaderboard", "/leaderboard/"), ("Guides", "/guides/")]
NAV_TH = [("วิธีเริ่มต้น", "/th/get-started/"), ("ตารางคะแนน", "/leaderboard/#th"), ("คู่มือ (อังกฤษ)", "/guides/")]
LOGO = ('<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="2" y="3" width="20" height="18" rx="4" fill="none" '
        'stroke="currentColor" stroke-width="2"/><path d="M7 9l3 3-3 3M12 15h5" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>')

M = catalog.MODELS
Q3, G4, IQ2 = M["qwen3.8-27b-q3"], M["gemma4-26b-a4b-qat"], M["qwen3.8-27b-iq2"]
RAM = headline.numbers()
RAM_Q, RAM_G = RAM["Qwen3.8-27B Q3"], RAM["gemma-4-26B-A4B QAT"]
pages: dict[str, str] = {}
meta: dict[str, str] = {}


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def term(lines: str, title: str = "terminal", copy_id: str | None = None) -> str:
    btn = f'<button class="copy" data-copy="{copy_id}" type="button">copy</button>' if copy_id else ""
    ident = f' id="{copy_id}"' if copy_id else ""
    return f'<div class="term"><div class="bar"><span>{esc(title)}</span>{btn}</div><pre{ident}>{lines}</pre></div>'


INSTALL = term('<span class="p">$ </span>pip install make-localllm-easier\n<span class="p">$ </span>localllm',
               "install and run", "install")


def doctor_sample() -> str:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    block = readme.split("## What `localllm doctor` tells you", 1)[1].split("```", 2)[1].strip("\n")
    out = []
    for line in esc(block).splitlines():
        line = line.replace("[OK  ]", '<span class="ok">[OK  ]</span>').replace(
            "[SLOW]", '<span class="warn">[SLOW]</span>').replace("[NO  ]", '<span class="no">[NO  ]</span>')
        if line.startswith("Best measured model"):
            line = f'<span class="hl">{line}</span>'
        out.append(line)
    return term('<span class="p">$ </span>localllm doctor\n' + "\n".join(out), "localllm doctor, RX 9070 XT 16 GB")


def ram_bars() -> str:
    top = max(v["default_gb"] for v in RAM.values()) * 1.15
    rows = []
    for name, v in RAM.items():
        rows.append(
            f'<div class="row"><div class="label">{esc(name)}</div>'
            f'<div class="bar"><div class="fill" style="width:{v["default_gb"] / top * 100:.1f}%"></div>'
            f'{v["default_gb"]:.2f} GB</div>'
            f'<div class="bar"><div class="fill ours" style="width:{v["localllm_gb"] / top * 100:.1f}%"></div>'
            f'{v["localllm_gb"]:.2f} GB</div></div>')
    return (f'<div class="bars" role="img" aria-label="System RAM after a 30-turn chat: llama.cpp defaults about '
            f'{RAM_Q["default_gb"]:.1f} GB, localllm about {RAM_Q["localllm_gb"]:.1f} GB">{"".join(rows)}</div>'
            '<div class="legend"><span><i style="background:var(--bar-base)"></i>llama.cpp defaults</span>'
            '<span><i style="background:var(--accent)"></i>localllm</span></div>')


def layout(path: str, title: str, desc: str, body: str, lang: str = "en", schema: dict | None = None) -> None:
    nav = "".join(f'<a href="{href}"{" aria-current=page" if href == path else ""}>{name}</a>'
                  for name, href in (NAV if lang == "en" else NAV_TH))
    other = '<a class="lang" href="/th/" hreflang="th">ไทย</a>' if lang == "en" else \
        '<a class="lang" href="/" hreflang="en">English</a>'
    start = ('<a class="btn primary nav-cta" href="/get-started/">Get started</a>' if lang == "en" else
             '<a class="btn primary nav-cta" href="/th/get-started/">เริ่มใช้งาน</a>')
    full = title if title.startswith("localllm") else f"{title} | localllm"
    ld = f'<script type="application/ld+json">{json.dumps(schema, ensure_ascii=False)}</script>' if schema else ""
    alt = ('<link rel="alternate" hreflang="en" href="' + URL + '/">'
           '<link rel="alternate" hreflang="th" href="' + URL + '/th/">') if path in ("/", "/th/") else ""
    pages[path] = f"""<!doctype html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(full)}</title>
<meta name="description" content="{esc(desc)}">
<link rel="canonical" href="{URL}{path}">{alt}
<meta property="og:type" content="website">
<meta property="og:title" content="{esc(full)}">
<meta property="og:description" content="{esc(desc)}">
<meta property="og:url" content="{URL}{path}">
<meta property="og:image" content="{URL}/assets/og.png">
<meta name="twitter:card" content="summary_large_image">
<meta name="theme-color" content="#1a8a5a">
<link rel="icon" href="/assets/favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="/assets/style.css">
{ld}
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header class="top"><div class="wrap">
<a class="brand" href="{'/th/' if lang == 'th' else '/'}">{LOGO}localllm</a>
<button class="menu-btn" type="button" aria-expanded="false" aria-controls="nav">Menu</button>
<nav class="nav" id="nav">{nav}<a href="{GH}">GitHub</a>{other}{start}</nav>
</div></header>
<main id="main">
{body}
</main>
<footer><div class="wrap">
<div class="cols">
<div><h4>localllm</h4><p>make-localllm-easier {__version__}. MIT licensed. Models keep their own licenses.</p></div>
<div><h4>Use it</h4><a href="/get-started/">Get started</a><a href="/how-it-works/">How it works</a><a href="/leaderboard/">Leaderboard</a><a href="{GH}/blob/main/docs/apis.md">API docs</a></div>
<div><h4>Read</h4><a href="/features/">Features</a><a href="/use-cases/">Use cases</a><a href="/compare/">Compare</a><a href="/guides/">Guides</a></div>
<div><h4>Project</h4><a href="{GH}">GitHub</a><a href="{PYPI}">PyPI</a><a href="{GH}/blob/main/ROADMAP.md">Roadmap</a><a href="{GH}/blob/main/CONTRIBUTING.md">Contributing</a><a href="{GH}/releases">Releases</a></div>
<div><h4>Contact</h4><a href="{ISSUES}/new">Report a problem</a><a href="{ISSUES}/new?title=Report">Share a result</a><a href="{ISSUES}">Questions and ideas</a><a href="/th/">ภาษาไทย</a></div>
</div>
</div></footer>
<script src="/assets/site.js" defer></script>
</body>
</html>
"""
    meta[path] = desc


def cta(primary=("Get started", "/get-started/"), secondary=("View on GitHub", GH)) -> str:
    return (f'<div class="btns"><a class="btn primary" href="{primary[1]}">{primary[0]}</a>'
            f'<a class="btn" href="{secondary[1]}">{secondary[0]}</a></div>')


STAR = f'<a class="btn" href="{GH}/stargazers">&#9734; Star on GitHub</a>'


def end_cta(text: str, secondary=("Read the docs", GH + "#readme")) -> str:
    btns = cta(("Get started", "/get-started/"), secondary).replace("</div>", STAR + "</div>")
    return f'<section><div class="wrap narrow"><h2>{text}</h2>{INSTALL}{btns}</div></section>'


def page_head(kicker: str, h1: str, lead: str, crumbs: str = "") -> str:
    c = f'<div class="crumbs">{crumbs}</div>' if crumbs else ""
    return (f'<section class="hero"><div class="wrap narrow">{c}<div class="kicker">{kicker}</div><h1>{h1}</h1>'
            f'<p class="lead">{lead}</p></div></section>')


# ---- Home ------------------------------------------------------------------------------------------------------------

def home() -> None:
    langs = sorted({k.split("/")[0] for m in M.values() for k in m["scores"]})
    body = f"""
<section class="hero"><div class="wrap hero-grid">
<div>
<div class="kicker">Open source · Windows, Linux, macOS</div>
<h1>Run the best local LLM your PC can handle, in one command.</h1>
<p class="lead"><code>localllm</code> checks your GPU and RAM, picks the most accurate model we have measured for your
language that fits the card, and runs it on llama.cpp with settings we profiled. Free, offline after the first
download, nothing leaves your PC.</p>
{INSTALL}
{cta()}
<p class="small muted">Needs Python 3.9+. The models we ship today want a GPU with 10 GB of VRAM or more;
<code>localllm doctor</code> tells you what fits on yours.</p>
</div>
<div>{doctor_sample()}</div>
</div></section>

<div class="trust"><div class="wrap">
<span>AMD · NVIDIA · Intel · Apple</span><span>Picks from measured benchmarks</span><span>Offline and private</span>
<span>Open source (MIT)</span><span>Downloads checked (SHA-256)</span>
</div></div>

<section><div class="wrap">
<div class="facts">
<div><b>{RAM_G['localllm_gb']:.1f} GB</b><span>system RAM after a 30-turn chat (llama.cpp defaults: {RAM_G['default_gb']:.1f} GB)</span></div>
<div><b>{len(langs)} languages</b><span>scored for every model; <code>eval</code> covers 23 languages and exams from 44 countries</span></div>
<div><b>4 APIs</b><span>OpenAI, Anthropic, Ollama and Gemini formats on one local port</span></div>
<div><b>~{G4['tok_s_9070xt']} tok/s</b><span>gemma-4-26B-A4B on an RX 9070 XT 16 GB</span></div>
</div>
</div></section>

<section><div class="wrap">
<h2>What it does for you</h2>
<p class="muted narrow">Choosing a local model usually means reading benchmark threads, guessing a quantization and
finding out later that it is slow, too big, or weak in your language. localllm does that part from measurements.</p>
<div class="cards">
<div class="card"><div class="n">01</div><h3>Picks from measured scores</h3><p>Every model in the catalog was run
through real exams in several languages on our test PC. The pick is the most accurate one that fits your card; near
ties go to the faster model.</p></div>
<div class="card"><div class="n">02</div><h3>Your language, not only English</h3><p>Scores are kept per language.
A Thai user and a Spanish user can get different models. 2-bit quants are flagged: they cost Thai, Hindi and Arabic
about 13 points.</p></div>
<div class="card"><div class="n">03</div><h3>Less RAM, same speed</h3><p>llama.cpp's defaults keep up to 8 GB of
prompt cache in system RAM. localllm sizes the caches to your PC: about 4x less RAM in the same chat, same tok/s.</p></div>
<div class="card"><div class="n">04</div><h3>Works with the apps you have</h3><p>One endpoint at
<code>127.0.0.1:8080</code> speaks four APIs, with tool calling and images. <code>localllm connect</code> sets up
Hermes, OpenCode, Claude Code, Codex, Aider or Obsidian.</p></div>
</div>
<p><a href="/features/">All features</a></p>
</div></section>

<section><div class="wrap two">
<div>
<h2>4x less RAM in a long chat</h2>
<p class="muted">Same 30-turn chat, same model, same speed. The difference is cache sizing: llama-server's defaults
keep an 8 GiB prompt cache and 32 conversation checkpoints in system RAM, so memory keeps growing while you talk.
Measured on an RX 9070 XT with 32 GB RAM.</p>
<p class="small"><a href="/guides/how-much-ram-does-a-local-llm-need/">How much RAM does a local LLM need?</a></p>
</div>
<div>{ram_bars()}</div>
</div></section>

<section><div class="wrap">
<h2>Who it is for</h2>
<div class="cards">
<div class="card"><h3>You want AI on your own PC</h3><p>Private by default. After the first download it runs
offline.</p></div>
<div class="card"><h3>You don't work in English</h3><p>Thai, Japanese, Chinese, Hindi, Arabic, Spanish are measured
today. Others can be measured with one command.</p></div>
<div class="card"><h3>You have an AMD or Intel GPU</h3><p>Vulkan builds, and the Windows Resizable BAR fix applied
where it measured faster.</p></div>
<div class="card"><h3>You build things</h3><p>A local API that existing SDKs already speak, for prototypes, scripts
and coding agents.</p></div>
</div>
<p><a href="/use-cases/">Use cases in detail</a></p>
</div></section>

<section><div class="wrap two">
<div>
<h2>How it works</h2>
<ol class="steps">
<li><div><b>Checks your PC.</b> GPU, VRAM, memory bandwidth, system RAM, your language.</div></li>
<li><div><b>Picks a model.</b> The most accurate measured model that fits with room for an 8k-token
conversation.</div></li>
<li><div><b>Downloads and verifies.</b> llama.cpp and the model, each checked against a pinned SHA-256.</div></li>
<li><div><b>Starts it.</b> A chat page in your browser and an API for your apps.</div></li>
</ol>
<p><a href="/how-it-works/">The details</a></p>
</div>
<div>
<h2>Compared with what you may use now</h2>
<div class="tbl"><table>
<tr><th></th><th>localllm</th><th>Ollama / LM Studio</th><th>llama.cpp yourself</th></tr>
<tr><td>Which model?</td><td>Picked from measured scores for your language</td><td>You choose</td><td>You choose</td></tr>
<tr><td>Settings</td><td>Profiled per model, <code>tune</code> per PC</td><td>Their defaults</td><td>All yours</td></tr>
<tr><td>Interface</td><td>Browser chat with saved chats, terminal, API</td><td>Polished apps</td><td>CLI and server</td></tr>
</table></div>
<p><a href="/compare/">Full comparison</a></p>
</div>
</div></section>
{end_cta("Try it on your PC")}
"""
    layout("/", "localllm: run the best local LLM your GPU can handle, in one command",
           "localllm picks, downloads and runs the most accurate local AI model for your GPU and your language, from "
           "measured benchmark scores, on a tuned llama.cpp. AMD, NVIDIA, Intel and Apple. Free and open source.",
           body, schema={"@context": "https://schema.org", "@type": "SoftwareApplication",
                         "name": "make-localllm-easier (localllm)", "applicationCategory": "DeveloperApplication",
                         "operatingSystem": "Windows, Linux, macOS", "softwareVersion": __version__,
                         "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"},
                         "license": "https://opensource.org/licenses/MIT", "url": URL + "/",
                         "downloadUrl": PYPI, "codeRepository": GH})


# ---- Features --------------------------------------------------------------------------------------------------------

def features() -> None:
    body = page_head("Features", "What localllm does, and what it doesn't",
                     "It is a small Python tool around llama.cpp. Its job is to make the choices you would otherwise "
                     "make by trial and error, and to show the numbers behind them.") + f"""
<section><div class="wrap prose">
<h2>Model choice from measurements</h2>
<p>Each model in the catalog was run through the same tests on the same PC: Global-MMLU-Lite (400 questions translated
into many languages), INCLUDE (real exams from 44 countries), ThaiExam, MGSM math word problems and FLORES
translation. <code>localllm</code> takes the scores in your language, keeps the models that fit your card with an
8k-token context, and picks the best. Gaps under 2 points count as a tie, because the benchmark margin is about
&plusmn;4, and a tie goes to the faster model.</p>
<p>The catalog is small on purpose: three files today. A model is added only after it is measured.
<a href="/leaderboard/">See every score.</a></p>

<h2>Per-language results</h2>
<p>Quantization does not cost every language the same. On Qwen3.8-27B, going from ~3.5 bits to 2 bits cost English,
Chinese and Spanish 8-9 points, and Hindi, Arabic and Thai 13. localllm keeps the 2-bit file only for cards too small
for anything else and says so in <code>doctor</code>.</p>

<h2>Memory: less RAM, same speed</h2>
<p>With the model fully on the GPU, localllm uses about {RAM_G['localllm_gb']:.1f}-{RAM_Q['localllm_gb']:.1f} GB of
system RAM in a 30-turn chat, where llama-server's defaults reach {RAM_G['default_gb']:.1f}-{RAM_Q['default_gb']:.1f}
GB. Weights are read straight into VRAM instead of being memory-mapped, which also keeps the working set small when a
MoE model has to keep some experts in RAM.</p>
<p>When a model spills out of VRAM into shared memory on Windows, answers get several times slower with no message
from the driver. localllm checks for it after loading and tells you.</p>

<h2>Speed settings that were measured</h2>
<ul>
<li>Multi-token prediction for Qwen3.8 (drafting 2 tokens: about 40% faster; 3 was slower).</li>
<li>The Vulkan host-visible memory fix for AMD and Intel cards without Resizable BAR, applied per model: 1.64x faster
on Qwen3.8-27B, 7-9% slower on gemma-4, so it is not on for everything.</li>
<li><code>localllm tune</code> measures these on your PC in a few minutes and keeps a setting only if it is at least
1.1x faster.</li>
</ul>

<h2>One endpoint, four APIs</h2>
<p><code>http://127.0.0.1:8080</code> answers OpenAI (<code>/v1/chat/completions</code>, <code>/v1/responses</code>,
embeddings), Anthropic (<code>/v1/messages</code>), Ollama (<code>/api/chat</code>) and Gemini
(<code>generateContent</code>) requests, including tool calls, streaming and images with <code>--vision</code>.
Existing SDKs only need a new base URL. <code>localllm serve --host 0.0.0.0</code> shares it on your network behind an
API key.</p>

<h2>Two profiles: lean or max</h2>
<p><b>lean</b> (the default) keeps an 8k-token context and caches sized so system RAM stays around
{RAM_G['localllm_gb']:.1f} GB. <b>max</b> (<code>--profile max</code>) spends what the PC has: a context that fills the
free VRAM (about 78k tokens on a 16 GB card) and a larger prompt cache, for long documents and coding agents.
<code>localllm doctor</code> shows both for your PC. Same model, same answers; only memory changes.</p>

<h2>A chat page with saved chats</h2>
<p><code>localllm --ui</code> opens our own chat page instead of llama-server's: chats saved as plain files on your
disk, a model picker when you run several models, the speed of each answer, and live VRAM and RAM with a warning
when the model spills out of the GPU. The page loads nothing from the internet. Shared on your network, it asks for
the key printed in the terminal.</p>

<h2>Agents and notes: <code>localllm connect</code></h2>
<p>Writes the config for Hermes Agent, OpenCode, Claude Code, Codex CLI, Aider or Copilot for Obsidian so they use the
local model, backs up the old file, and <code>--undo</code> puts it back. For Obsidian it also tells you what would
still go to a cloud service.</p>

<h2>See what running locally saves</h2>
<p><code>localllm savings</code> counts the tokens this PC answered this month, how much of each prompt came from the
cache, and, at the cloud prices you enter, what that would have cost. No prices are built in: they change, and a
guess would be worse than none.</p>

<h2>Optional extras</h2>
<ul>
<li><b>Routing between local models</b> (<code>--models auto</code>): each message goes to the model with the best
score for its language and task. Off by default: on a 16 GB card a swap costs 4-9 seconds.</li>
<li><b>Your own cloud key</b> (<code>localllm route</code>): off by default. When you turn it on, a rule decides when
a request goes to the cloud, and every answer says where it came from.</li>
<li><b>Benchmarks</b> (<code>localllm eval</code>): score any OpenAI-compatible server in your language, including
a long-context test that hides numbers deep in 8k-64k-token documents (<code>--suites longctx</code>) and news
summaries in 24 languages (<code>--suites summarize</code>).</li>
<li><b>Energy per answer</b> (<code>localllm tune</code>): the GPU's own power reading while it answers, shown as
joules per token in <code>doctor</code>. AMD and NVIDIA; <code>--prefer-energy</code> keeps settings that save
more than 15% energy for less than 5% speed.</li>
<li><b>Safe upgrades</b>: llama.cpp is pinned to the build the numbers were measured on. A new build is smoke-tested
on our PC first, and <code>doctor</code> says when yours differs.</li>
</ul>

<h2>What it doesn't do (yet)</h2>
<ul>
<li>No desktop app. The interface is a browser page, the terminal and the API.</li>
<li>No numbers yet from the long-context test for the catalog models; they are queued.</li>
<li>No models for cards under ~10 GB in the catalog yet. Measuring them is
<a href="{GH}/issues/38">issue #38</a>.</li>
<li>Only Windows with an AMD card is tested by us. Everything else depends on
<a href="/leaderboard/#platforms">community reports</a>.</li>
</ul>
</div></section>
{end_cta("Try it", ("How it works", "/how-it-works/"))}
"""
    layout("/features/", "Features",
           "Model choice from measured per-language scores, about 4x less RAM than llama.cpp defaults, measured speed "
           "settings, one local endpoint for OpenAI, Anthropic, Ollama and Gemini apps.", body)


# ---- How it works ----------------------------------------------------------------------------------------------------

def how() -> None:
    body = page_head("How it works", "From <code>pip install</code> to a running model",
                     "What happens when you type <code>localllm</code>, step by step, and what you can change.") + f"""
<section><div class="wrap prose">
<h2>1. It looks at your PC</h2>
<p>llama.cpp lists the GPUs it can use (Vulkan on Windows and Linux, Metal on macOS). localllm takes the one with the
most memory, skipping integrated graphics, reads installed and free RAM, and looks up your card's memory bandwidth to
estimate speed. It also reads your system language. Run <code>localllm doctor</code> to see all of it without
downloading a model.</p>

<h2>2. It picks a model</h2>
<p>From the catalog it keeps the models that fit on the card with room for an 8k-token conversation. A MoE model that
doesn't fit whole can keep some experts in system RAM. Among those, it takes the best measured score in your language,
and on a near tie (under 2 points) the faster one. You can override it: <code>localllm --model KEY</code>.</p>

<h2>3. It downloads and checks the files</h2>
<p>llama.cpp comes from build b11476, the one the numbers were measured on. The model comes from a pinned commit on
Hugging Face. Each file is checked against a SHA-256 stored in the code. Downloads resume if they stop. Everything goes
to <code>~/.localllm/</code>. The first time, before the multi-GB model download, it shows what your PC can run and
asks.</p>

<h2>4. It starts the model</h2>
<p>llama-server starts on a private port with the settings from the catalog: flash attention, an 8-bit KV cache, one
slot, caches sized to your RAM, MTP drafting where it helps. In front of it, a small gateway serves the chat page and
the four APIs on port 8080. Your browser opens the chat page.</p>
{term('<span class="p">$ </span>localllm' + chr(10) + '[localllm] loading gemma4-26b-a4b-qat on AMD Radeon RX 9070 XT ...' + chr(10) + '[localllm] ready. chat: http://127.0.0.1:8080   API: http://127.0.0.1:8080/v1 (OpenAI), ...', 'first run, after the download')}

<h2>5. You use it</h2>
<ul>
<li>Chat in the browser, or <code>localllm chat</code> in the terminal (<code>/save</code>, <code>/think</code>,
Ctrl+C stops an answer).</li>
<li>Point an app at <code>http://127.0.0.1:8080/v1</code> with any API key.</li>
<li>Run <code>localllm tune</code> once to measure the speed settings on your PC.</li>
<li>Want saved chats? <code>localllm --ui</code>. Long documents or an agent? <code>--profile max</code>.</li>
</ul>

<h2>What you don't have to do</h2>
<ul>
<li>Pick a quantization, or find out a 2-bit file lost 13 points in your language.</li>
<li>Work out whether a model fits with your context length.</li>
<li>Find the llama.cpp flags that matter, or notice that the prompt cache is eating your RAM.</li>
</ul>
</div></section>
{end_cta("Ready for the first step?", ("Use cases", "/use-cases/"))}
"""
    layout("/how-it-works/", "How it works",
           "localllm checks your GPU and RAM, picks the most accurate measured model that fits, downloads llama.cpp "
           "and the model with SHA-256 checks, and starts a chat page and a local API.", body)


# ---- Use cases -------------------------------------------------------------------------------------------------------

def uses() -> None:
    items = [
        ("A private assistant on your own PC",
         "You don't want every question to go to a cloud service, and you don't want to spend an evening on setup.",
         "<code>pip install make-localllm-easier</code>, then <code>localllm</code>. The chat page opens when the "
         "model is ready. After the first download it works without internet.",
         "A model that answers at 50-90 tok/s on a 16 GB card, with your data staying on the disk."),
        ("Working in Thai, Japanese, Hindi, Arabic or another language",
         "Most model comparisons are in English. A model that is fine in English can be clearly weaker in your "
         "language, and quantization makes the gap larger.",
         "localllm keeps scores per language and picks for yours. <code>localllm eval --langs en,ja</code> measures any "
         "model on your hardware, in a few minutes per language.",
         "On our test PC, Chinese knowledge questions go to Qwen3.8 (+5.5 points), and Thai math to gemma-4 (89.6 vs "
         "87.2)."),
        ("A local API for development",
         "You want an endpoint for prototypes, tests or internal tools without per-token bills.",
         "<code>localllm serve</code> gives you OpenAI, Anthropic, Ollama and Gemini formats on one port, with tool "
         "calling. Set the base URL in the SDK you already use.",
         "Code written against the OpenAI or Anthropic SDK runs locally unchanged."),
        ("A coding agent on a local model",
         "Agent tools like Hermes, OpenCode or Aider can run up large API bills, and you would like to try a local "
         "model for part of the work.",
         "<code>localllm serve --profile max</code>, then <code>localllm connect hermes</code> (or opencode, "
         "claude-code, codex, aider). <code>--undo</code> restores your old config.",
         "The agent talks to your GPU. How well each agent works with these models is still being measured "
         "(<a href=\"" + GH + "/issues/96\">#96</a>)."),
        ("Your notes in Obsidian, kept on your PC",
         "AI plugins for notes usually send your text to a cloud service, and it is easy to miss which setting does.",
         "<code>localllm connect obsidian --path YOUR_VAULT</code> makes the local model the default in Copilot for "
         "Obsidian and lists anything that would still reach the cloud, such as a cloud embedding model for search.",
         "Chat about your notes with the model on your GPU; <code>--undo</code> puts the plugin's old settings back."),
        ("An AMD or Intel graphics card",
         "Guides and tools often assume NVIDIA and CUDA. On Windows without Resizable BAR, llama.cpp on Vulkan can be "
         "up to 1.7x slower.",
         "localllm uses the Vulkan builds of llama.cpp and applies the host-visible memory fix for models where it "
         "measured faster. <code>localllm tune</code> checks it on your PC.",
         "Our own test PC is an RX 9070 XT on Windows 11, so this is the setup we know best."),
        ("A PC that is also used for other things",
         "Local models can take most of your RAM and keep the GPU busy, which gets in the way of games and other "
         "apps.",
         "Caches are sized to your RAM (about 2.2 GB in a long chat). With routing on, models unload after 15 minutes "
         "without a message and load again on the next one.",
         "The rest of your RAM stays free, and the GPU is free when you are not chatting."),
    ]
    cards = "".join(
        f'<div class="card"><h3>{t}</h3><p><b>The problem.</b> {p}</p><p><b>With localllm.</b> {s}</p>'
        f'<p class="muted"><b>Result.</b> {r}</p></div>' for t, p, s, r in items)
    body = page_head("Use cases", "What people use it for",
                     "Seven situations where it helps, with the commands. If yours isn't here, "
                     f'<a href="{ISSUES}">open an issue</a>.') + f"""
<section><div class="wrap"><div class="cards">{cards}</div>
<div class="note warn" style="margin-top:24px"><b>Not a fit yet:</b> GPUs under ~10 GB and CPU-only PCs. The tool runs
and <code>doctor</code> tells you what would fit, but no small model has been measured for the catalog yet
(<a href="{GH}/issues/38">#38</a>).</div>
</div></section>
{end_cta("Found yours?", ("Compare with other tools", "/compare/"))}
"""
    layout("/use-cases/", "Use cases",
           "Private local AI, non-English languages, a local OpenAI-compatible API, coding agents on a local model, "
           "AMD and Intel GPUs, and keeping RAM free for other apps.", body)


# ---- Compare ---------------------------------------------------------------------------------------------------------

def compare() -> None:
    rows = [
        ("Choosing a model", "Picked for you from measured scores in your language, with the reason shown",
         "You pick from a large library", "You pick from a model browser", "You find and download GGUF files"),
        ("Model library", "Small: 3 measured models today", "Large", "Large (GGUF and MLX)", "Any GGUF"),
        ("Per-language scores", "Yes, for 7 languages; measure others with <code>eval</code>", "No", "No", "No"),
        ("Launch settings", "Profiled per model; <code>tune</code> measures them on your PC",
         "Ollama's defaults", "Settings in the app", "Every flag is yours"),
        ("System RAM in a long chat", f"~{RAM_G['localllm_gb']:.1f} GB (measured)", "Not measured by us",
         "Not measured by us", f"~{RAM_G['default_gb']:.1f} GB with defaults (measured)"),
        ("Interface", "Browser chat page with saved chats (<code>--ui</code>), terminal chat, API",
         "CLI, API, desktop app", "Desktop app with a GUI",
         "CLI, server, simple web page"),
        ("APIs", "OpenAI, Anthropic, Ollama, Gemini on one port", "Its own API plus OpenAI-compatible",
         "Its own API plus OpenAI-compatible", "OpenAI-compatible, Anthropic Messages"),
        ("GPUs", "Vulkan (AMD, NVIDIA, Intel), Metal", "CUDA, ROCm, Metal, Vulkan", "CUDA, Vulkan, Metal",
         "All backends"),
        ("Open source", "Yes, MIT", "Yes, MIT", "No (free app)", "Yes, MIT"),
    ]
    trs = "".join(f"<tr><td><b>{r[0]}</b></td>" + "".join(f"<td>{c}</td>" for c in r[1:]) + "</tr>" for r in rows)
    body = page_head("Compare", "localllm, Ollama, LM Studio, or llama.cpp yourself",
                     "They all run local models, and three of them run on llama.cpp. The difference is who chooses "
                     "the model and the settings, and how much you can see of why.") + f"""
<section><div class="wrap">
<div class="tbl"><table><thead><tr><th></th><th>localllm</th><th>Ollama</th><th>LM Studio</th><th>llama.cpp
yourself</th></tr></thead><tbody>{trs}</tbody></table></div>
<p class="small muted" style="margin-top:10px">Other tools' features change often; this table reflects October 2026.
RAM numbers are only given where we measured them. Corrections are welcome as an
<a href="{ISSUES}">issue</a>.</p>
</div></section>

<section><div class="wrap two">
<div class="prose">
<h2>vs Ollama</h2>
<p>Ollama is easy to start and has the biggest model library. You choose the model and the tag. localllm is narrower:
fewer models, but each one measured per language, and the choice made for your card. If you already like Ollama's
workflow, keep it; localllm's endpoint also speaks the Ollama API, so apps built for Ollama work against it.</p>
<h2>vs LM Studio</h2>
<p>LM Studio has a good desktop interface for browsing, downloading and chatting. localllm has no desktop app. It is
for people who would rather have the answer to "which model and which settings" than a list to choose from, and who
are fine with a browser page or a terminal.</p>
</div>
<div class="prose">
<h2>vs running llama.cpp yourself</h2>
<p>llama.cpp gives you every option, and localllm is built on it. What localllm adds is the choices we measured:
cache sizes, which speed flags help which model, the Vulkan fix where it helps and not where it hurts, and which
quantization holds up in your language. You can read every flag it passes in
<a href="{GH}/blob/main/localllm/runtime.py">runtime.py</a>.</p>
<h2>When to pick something else</h2>
<ul>
<li>You want a polished desktop app: LM Studio.</li>
<li>You want a model we don't have in the catalog: Ollama, LM Studio or llama.cpp.</li>
<li>You want full control of every setting: llama.cpp.</li>
<li>Your GPU has less than ~10 GB: any of the others, for now.</li>
</ul>
</div>
</div></section>
{end_cta("Want the measured choice?", ("See the features", "/features/"))}
"""
    layout("/compare/", "localllm vs Ollama vs LM Studio vs llama.cpp",
           "An honest comparison: who picks the model and the settings, per-language scores, RAM use, interfaces and "
           "APIs. Where to use localllm and where another tool is the better fit.", body)


# ---- Get started -----------------------------------------------------------------------------------------------------

def start() -> None:
    commands = ("localllm chat       <span class=\"p\"># chat in the terminal</span>\n"
                "localllm doctor     <span class=\"p\"># what this GPU can run, speed, RAM</span>\n"
                "localllm list       <span class=\"p\"># every measured model and its scores</span>\n"
                "localllm serve      <span class=\"p\"># API only, no browser</span>\n"
                "localllm tune       <span class=\"p\"># measure the fastest settings once</span>\n"
                "localllm eval       <span class=\"p\"># score a running server in your language</span>\n"
                "localllm report     <span class=\"p\"># share your results (anonymous)</span>\n"
                "localllm connect hermes   <span class=\"p\"># point an agent at it</span>\n"
                "localllm --ui       <span class=\"p\"># chat page with saved chats</span>\n"
                "localllm savings    <span class=\"p\"># tokens answered locally this month</span>")
    faq = [
        ("<code>pip</code> is not found", "Install Python 3.9 or newer from python.org (on Windows, tick \"Add python.exe "
         "to PATH\"), then use <code>python -m pip install make-localllm-easier</code>."),
        ("<code>localllm</code> is not found after installing", "pip put it in a folder that is not on your PATH. "
         "<code>python -m localllm</code> works the same."),
        ("\"no measured model fits this GPU yet\"", "The catalog's smallest model needs about 10 GB of VRAM. Run "
         "<code>localllm doctor</code> to see what sizes fit; smaller models are tracked in "
         f"<a href=\"{GH}/issues/38\">#38</a>."),
        ("It is slower than the numbers here", "Run <code>localllm tune</code>. On Windows, if <code>doctor</code> "
         "warns that the model spilled into system RAM, close other GPU-heavy apps. On AMD/Intel, check Resizable BAR "
         "in your BIOS; see <a href=\"/guides/llama-cpp-slow-on-amd-intel-windows/\">this guide</a>."),
        ("The first answer takes a long time", "The first run of a new llama.cpp build compiles GPU shaders once "
         "(about 15 seconds). Later starts are fast."),
        ("Where are the files?", "<code>~/.localllm/</code>: models, llama.cpp, tune results, saved chats. Delete the "
         "folder to remove everything. Use <code>LOCALLLM_MODELS</code> to point at a folder of GGUF files you already "
         "have."),
        ("Windows or macOS warns about the downloaded file", "The single-file builds are not code-signed yet. Windows: "
         "More info > Run anyway. macOS: right-click > Open once. Or install with <code>pip</code>, which has no "
         "warning."),
    ]
    qa = "".join(f"<details><summary>{q}</summary><p>{a}</p></details>" for q, a in faq)
    body = page_head("Get started", "Install and run it",
                     "Two commands. The first run downloads llama.cpp (~30 MB) and one model (8-14 GB), once.") + f"""
<section><div class="wrap two">
<div class="prose">
<h2>Install</h2>
<ol class="steps">
<li><div>Have Python 3.9 or newer (<code>python --version</code>).</div></li>
<li><div>Install: <code>pip install make-localllm-easier</code></div></li>
<li><div>Run <code>localllm</code>. The chat page opens when the model is loaded.</div></li>
</ol>
{INSTALL}
<p class="small muted">The first run shows what your PC can run and asks before the 8-14 GB model download.
<code>localllm doctor</code> shows the same without downloading a model.</p>
<h3>No Python?</h3>
<p>From version 0.7.1, every <a href="{GH}/releases">release</a> has a single <code>localllm</code> file for Windows,
macOS and Linux, and an AppImage. Download it and run it from a terminal. winget and Homebrew packages follow.</p>
</div>
<div class="prose">
<h2>Then</h2>
{term(commands, "more commands")}
<p class="small">Using it from an app: base URL <code>http://127.0.0.1:8080/v1</code>, any API key.
<a href="{GH}/blob/main/docs/apis.md">API docs and examples</a>.</p>
</div>
</div></section>

<section><div class="wrap narrow">
<h2>Common problems</h2>
{qa}
<p>Something else? <a href="{ISSUES}">Open an issue</a> with the output of <code>localllm doctor</code>.</p>
</div></section>

<section><div class="wrap narrow">
<h2>Help the next person</h2>
<p>If it ran on your PC, <code>localllm tune</code>, <code>localllm eval</code> and <code>localllm report</code> take
10-30 minutes and add your GPU to the <a href="/leaderboard/">leaderboard</a>. The report has no names, paths or keys,
and you read it before you submit it.</p>
</div></section>
"""
    layout("/get-started/", "Get started",
           "Install with pip install make-localllm-easier and run localllm. Requirements, first commands, the local "
           "API address and fixes for common problems.", body)


# ---- Leaderboard -----------------------------------------------------------------------------------------------------

def leaderboard() -> None:
    data = json.loads((ROOT / "data" / "leaderboard.json").read_text(encoding="utf-8"))
    langs = sorted({k.split("/")[0] for e in data["entries"] for k in e["scores"]},
                   key=lambda c: (c != "en", LANG_NAMES.get(c, c)))
    opts = "".join(f'<option value="{c}">{LANG_NAMES.get(c, c)}</option>' for c in langs)
    payload = {"entries": data["entries"], "languages": {c: LANG_NAMES.get(c, c) for c in langs}}
    plat = data["platforms"]
    vendors = ["AMD", "NVIDIA", "Intel", "Apple"]
    prow = ""
    for o in ("Windows", "Linux", "macOS"):
        cells = ""
        for v in vendors:
            c = plat.get(f"{o}/{v}")
            if c is None:
                cells += '<td class="muted">-</td>'
                continue
            bits = (['<span class="tag ours">tested by us</span>'] if c["maintainers"] else []) + \
                ([f'<span class="tag">{c["community_reports"]} report(s)</span>'] if c["community_reports"] else [])
            cells += f"<td>{' '.join(bits) or '<span class=muted>untested</span>'}</td>"
        prow += f"<tr><td><b>{o}</b></td>{cells}</tr>"
    body = page_head("Leaderboard", "Measured local LLM scores, per language and per GPU",
                     "Accuracy on real exams and decode speed, from our test PC and from people who shared a "
                     "<code>localllm report</code>. Pick a language.") + f"""
<section><div class="wrap">
<div class="controls">
<label>Language<select id="lang">{opts}</select></label>
<label>Measured by<select id="src"><option value="all">everyone</option><option value="maintainers">maintainers</option><option value="community">community</option></select></label>
</div>
<div class="tbl" id="board"><noscript><p style="padding:12px">The table needs JavaScript. The same data is in
<a href="{GH}/blob/main/data/leaderboard.json">data/leaderboard.json</a>.</p></noscript></div>
<script type="application/json" id="board-data">{json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")}</script>
<p class="small muted" style="margin-top:12px"><b>Knowledge (translated)</b>: Global-MMLU-Lite, the same 400 questions in
every language, so languages compare like for like. <b>Local exams</b>: INCLUDE, written in each country (ThaiExam for
Thai). <b>Math</b>: MGSM. <b>Translation</b>: FLORES-101, chrF++ (0-100). Zero-shot, thinking off. Margins are about
&plusmn;4 points (translated) and &plusmn;5 (local exams). Speeds are llama-server decode on the listed card. Raw data:
<a href="{GH}/blob/main/data/leaderboard.json">data/leaderboard.json</a>.</p>
</div></section>

<section id="platforms"><div class="wrap">
<h2>Where it is tested</h2>
<p class="muted narrow">One test PC is ours: Windows 11 with an RX 9070 XT. The rest of this table fills in from shared
reports.</p>
<div class="tbl"><table><thead><tr><th></th>{"".join(f"<th>{v}</th>" for v in vendors)}</tr></thead>
<tbody>{prow}</tbody></table></div>
</div></section>

<section><div class="wrap narrow">
<h2>Add your GPU</h2>
<p>Ten to thirty minutes, no coding:</p>
{term('<span class="p">$ </span>localllm tune' + chr(10) + '<span class="p">$ </span>localllm eval --langs en,de     <span class="p"># your language instead of de</span>' + chr(10) + '<span class="p">$ </span>localllm report', "share a result", "report")}
<p class="small muted"><code>report</code> opens a prefilled GitHub issue with your GPU, RAM, OS, llama.cpp build,
tuned settings and scores. No names, paths or keys; you read it before you submit it.
<a href="{GH}/blob/main/reports/README.md">How reports are checked.</a></p>
</div></section>
"""
    layout("/leaderboard/", "Local LLM leaderboard by language and GPU",
           "Measured accuracy (knowledge, local exams, math, translation) and decode speed of local LLMs per language "
           "and per GPU, from localllm's test PC and community reports.", body,
           schema={"@context": "https://schema.org", "@type": "Dataset",
                   "name": "localllm leaderboard: local LLM scores per language and GPU",
                   "description": "Accuracy and decode speed of quantized local LLMs per language and GPU.",
                   "url": URL + "/leaderboard/", "license": "https://opensource.org/licenses/MIT",
                   "distribution": {"@type": "DataDownload", "encodingFormat": "application/json",
                                    "contentUrl": GH.replace("github.com", "raw.githubusercontent.com")
                                    + "/main/data/leaderboard.json"}})


# ---- Guides (the README FAQ questions, as pages) ---------------------------------------------------------------------

GUIDES: list[tuple[str, str, str, str]] = []   # (slug, title, description, body html)


def guide(slug: str, title: str, desc: str, body: str, faq: list[tuple[str, str]] | None = None) -> None:
    GUIDES.append((slug, title, desc, body))
    crumbs = '<a href="/guides/">Guides</a>'
    schema = {"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": [
        {"@type": "Question", "name": q, "acceptedAnswer": {"@type": "Answer", "text": a}} for q, a in faq]} \
        if faq else {"@context": "https://schema.org", "@type": "TechArticle", "headline": title}
    html_body = (f'<section class="hero"><div class="wrap narrow"><div class="crumbs">{crumbs}</div>'
                 f'<h1>{title}</h1><p class="lead">{desc}</p></div></section>'
                 f'<section><div class="wrap prose">{body}</div></section>'
                 + end_cta("Check your own PC", ("All guides", "/guides/")))
    layout(f"/guides/{slug}/", title, desc, html_body, schema=schema)


def guides() -> None:
    guide("best-local-llm-for-16gb-gpu", "Which local LLM should I run on a 16 GB GPU?",
          "Short answer: a ~27B dense model at about 3.5 bits, or a 26B MoE at 4 bits. Measured on an RX 9070 XT, "
          "with scores per language.", f"""
<h2>The short answer</h2>
<p>On a 16 GB card, the two models we measured as best are <b>gemma-4-26B-A4B QAT</b> (4-bit, {G4['gb']} GB) and
<b>Qwen3.8-27B</b> at ~3.5 bits ({Q3['gb']} GB). Both fit with a long context: about 78k tokens for gemma-4.</p>
<div class="tbl"><table>
<tr><th></th><th>Qwen3.8-27B Q3</th><th>gemma-4-26B-A4B QAT</th></tr>
<tr><td>File size</td><td class="num">{Q3['gb']} GB</td><td class="num">{G4['gb']} GB</td></tr>
<tr><td>Decode speed (RX 9070 XT)</td><td class="num">~{Q3['tok_s_9070xt']} tok/s</td><td class="num">~{G4['tok_s_9070xt']} tok/s</td></tr>
<tr><td>English knowledge</td><td class="num">{Q3['scores']['en/global']}%</td><td class="num">{G4['scores']['en/global']}%</td></tr>
<tr><td>Chinese knowledge</td><td class="num">{Q3['scores']['zh/global']}%</td><td class="num">{G4['scores']['zh/global']}%</td></tr>
<tr><td>Math, English / Thai</td><td class="num">{Q3['scores']['en/math']} / {Q3['scores']['th/math']}</td><td class="num">{G4['scores']['en/math']} / {G4['scores']['th/math']}</td></tr>
<tr><td>Thai local exams</td><td class="num">{Q3['scores']['th/regional']}%</td><td class="num">{G4['scores']['th/regional']}%</td></tr>
</table></div>
<p>For most languages they are within 2 points of each other, which is inside the benchmark margin, so gemma-4 wins on
speed. Qwen3.8 is clearly ahead on Chinese, Spanish and Japanese knowledge questions.</p>

<h2>Why not a bigger model at 2 bits?</h2>
<p>Because it costs more than it gives. Qwen3.8-27B at 2 bits ({IQ2['gb']} GB) scored 8-13 points lower than at ~3.5
bits, and the lower-resource languages paid the most: Hindi, Arabic and Thai lost 13. A 177B MoE squeezed to 1.6 bits
scored below the 27B at 3 bits.</p>

<h2>What about 70B?</h2>
<p>A dense 70B model needs about 42 GB at 4-5 bits. On a 16 GB card it only fits at ~1.6 bits, where quality is gone.
MoE models with experts in system RAM are the realistic route to bigger models; we are measuring them
(<a href="{GH}/blob/main/docs/research-big-models.md">research notes</a>).</p>

<h2>Check your card</h2>
<p><code>localllm doctor</code> lists which model sizes fit your GPU, at which quantization, how fast they should run,
and the best measured model for your language.</p>
""", [("Which local LLM should I run on a 16 GB GPU?",
       "gemma-4-26B-A4B QAT at 4 bits (13.3 GB, ~90 tok/s on an RX 9070 XT) or Qwen3.8-27B at ~3.5 bits (12.2 GB, "
       "~50 tok/s). They score within 2 points in most languages; Qwen3.8 leads in Chinese, Spanish and Japanese."),
      ("Can a 16 GB GPU run a 27B model?", f"Yes. Qwen3.8-27B at ~3.5 bits is {Q3['gb']} GB and runs at about "
       f"{Q3['tok_s_9070xt']} tok/s on an RX 9070 XT with {Q3['scores']['en/global']}% on English Global-MMLU-Lite.")])

    guide("llama-cpp-slow-on-amd-intel-windows", "Why is llama.cpp slow on my AMD or Intel GPU on Windows?",
          "Usually Resizable BAR. Without it, llama.cpp's Vulkan backend can put buffers in a small host-visible heap, "
          "and decode gets up to 1.7x slower. Here is what to check and the fix.", f"""
<h2>The cause</h2>
<p>With Resizable BAR off, the GPU exposes only a 256 MB window of its memory to the CPU. llama.cpp's Vulkan backend
puts some buffers in that host-visible heap, which on these systems is backed by system RAM. Every token then crosses
the PCIe bus, and decode drops by up to 1.7x (<a href="https://github.com/ggml-org/llama.cpp/issues/27097">llama.cpp#27097</a>).</p>

<h2>Fix 1: turn on Resizable BAR</h2>
<p>In your BIOS/UEFI, enable <b>Above 4G Decoding</b> and <b>Resizable BAR</b> (AMD calls it Smart Access Memory).
This fixes the cause for every app.</p>

<h2>Fix 2: the environment variable</h2>
<p>If you can't change the BIOS, set this before starting llama-server:</p>
<pre class="block">set GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1</pre>
<p>But measure it: on an RX 9070 XT with Resizable BAR off it made Qwen3.8-27B <b>1.64x faster</b>, and
gemma-4-26B-A4B <b>7-9% slower</b>. llama.cpp treats any value, even <code>0</code>, as on, so unset it to turn it off.</p>

<h2>What localllm does</h2>
<p>It sets the variable only for models where it measured faster, and <code>localllm tune</code> measures it on your
PC and keeps it only if it is at least 1.1x faster. If you set the variable yourself, localllm leaves it alone.</p>

<h2>Other things that make it slow</h2>
<ul>
<li><b>The model doesn't fit.</b> On Windows the driver quietly moves part of it into shared system memory and decode
gets several times slower. localllm warns when this happens.</li>
<li><b>The first run of a new build.</b> The driver compiles shaders once, about 15 seconds.</li>
<li><b>An integrated GPU picked instead of the card.</b> localllm skips iGPUs; with llama.cpp alone, pass
<code>-dev Vulkan0</code>.</li>
</ul>
""", [("Why is llama.cpp slow on my AMD GPU on Windows?",
       "Usually because Resizable BAR is off. llama.cpp's Vulkan backend then uses a 256 MB host-visible heap "
       "backed by system RAM and decode drops up to 1.7x. Enable Resizable BAR in the BIOS, or set "
       "GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1 and measure whether it helps your model.")])

    langs_rows = "".join(
        f"<tr><td>{LANG_NAMES[l]}</td>"
        + "".join(f'<td class="num">{M[k]["scores"].get(f"{l}/{s}", "-")}</td>' for k in ("qwen3.8-27b-q3",
                  "gemma4-26b-a4b-qat") for s in ("global", "regional"))
        + "</tr>" for l in ("en", "zh", "es", "ja", "ar", "hi", "th"))
    guide("local-llm-for-thai-hindi-arabic", "The best local LLM for Thai, Hindi, Arabic and other languages",
          "Measured scores in seven languages, why quantization hurts some languages more, and how to test any "
          "model in your own language in a few minutes.", f"""
<h2>Scores per language</h2>
<p>Accuracy (%) on Global-MMLU-Lite (the same questions translated) and INCLUDE (each country's own exams; ThaiExam
for Thai). RX 9070 XT, llama.cpp Vulkan, zero-shot.</p>
<div class="tbl"><table>
<tr><th>Language</th><th class="num">Qwen3.8-27B Q3<br>translated</th><th class="num">local exams</th><th class="num">gemma-4-26B QAT<br>translated</th><th class="num">local exams</th></tr>
{langs_rows}
</table></div>
<p class="small muted">"-" means not measured. Margins are about &plusmn;4 and &plusmn;5 points.</p>

<h2>Lower-resource languages lose more to quantization</h2>
<p>Going from ~3.5 bits to 2 bits cost English, Chinese and Spanish 8-9 points and Hindi, Arabic and Thai 13. KL
divergence from the 8-bit model shows the same order. If you work in one of these languages, avoid 2-bit files unless
nothing else fits.</p>

<h2>Translation and math</h2>
<p>On FLORES-101 (chrF++), gemma-4 beats Qwen3.8 in Hindi (+4.7), Arabic (+2.9), Japanese (+1.6) and Thai (+1.5);
Qwen3.8 leads Chinese by 1.4. On MGSM math, gemma-4 scores {G4['scores']['th/math']} in Thai vs
{Q3['scores']['th/math']}.</p>

<h2>Measure your language</h2>
<p>Global-MMLU-Lite covers 23 languages and INCLUDE 44 countries. Against any running OpenAI-compatible server:</p>
<pre class="block">localllm eval --langs en,ko          # knowledge
localllm eval --suites math --langs de,ja
localllm eval --suites translate --langs sw,vi</pre>
<p>Each language takes a few minutes. Share the result with <code>localllm report</code> and it appears on the
<a href="/leaderboard/">leaderboard</a>.</p>
<p>Thai speakers: <a href="/th/">this site in Thai</a>.</p>
""", [("What is the best local LLM for Thai?",
       f"On our measurements, Qwen3.8-27B Q3 ({Q3['scores']['th/regional']}%) and gemma-4-26B-A4B QAT "
       f"({G4['scores']['th/regional']}%) are tied on ThaiExam; gemma-4 is ahead on Thai math and translation and "
       "1.8x faster."),
      ("Is a 2-bit model good enough for non-English use?",
       "Usually not: on Qwen3.8-27B, 2-bit cost Hindi, Arabic and Thai 13 points versus ~3.5 bits.")])

    guide("how-much-ram-does-a-local-llm-need", "How much RAM does a local LLM need?",
          f"With the model on the GPU, about {RAM_G['localllm_gb']:.1f} GB of system RAM, if the caches are sized for "
          f"one user. With llama-server's defaults it grows to about {RAM_Q['default_gb']:.0f} GB in a long chat.",
          f"""
<h2>Measured: a 30-turn chat</h2>
<p>Same model, same questions, RX 9070 XT, 32 GB RAM. System RAM used by the server after the 30th turn:</p>
{ram_bars()}
<p style="margin-top:16px">Speed was the same in both runs ({RAM_Q['localllm_tok_s']} tok/s for Qwen3.8, ~85 for
gemma-4).</p>

<h2>Where the RAM goes</h2>
<ul>
<li><b>Prompt cache.</b> llama-server keeps up to 8 GiB of earlier prompts in RAM by default
(<code>--cache-ram 8192</code>). Useful for many users; for one user, 0.5-2 GB is enough.</li>
<li><b>Context checkpoints.</b> 32 per slot by default. Each is a copy of part of the model state.</li>
<li><b>Memory-mapped weights.</b> With mmap the whole file can stay in the working set. Reading weights straight into
VRAM (<code>--load-mode none</code>) keeps them out of RAM.</li>
<li><b>Embeddings and host buffers.</b> Usually 0.3-0.5 GB each.</li>
</ul>

<h2>The settings localllm uses</h2>
<div class="tbl"><table>
<tr><th>Installed RAM</th><th class="num">--cache-ram</th><th class="num">--ctx-checkpoints</th></tr>
<tr><td>up to 16 GB</td><td class="num">512 MiB</td><td class="num">2</td></tr>
<tr><td>up to 32 GB</td><td class="num">1024 MiB</td><td class="num">4</td></tr>
<tr><td>up to 64 GB</td><td class="num">2048 MiB</td><td class="num">8</td></tr>
<tr><td>more</td><td colspan="2">llama.cpp defaults</td></tr>
</table></div>

<h2>And if the model doesn't fit the GPU?</h2>
<p>Then RAM holds part of the model. For a MoE model like gemma-4-26B-A4B, keeping some layers' experts in RAM still
gives 45 / 36 / 31 tok/s with 8 / 13 / 18 layers off the GPU (for 12 / 10 / 8 GB cards). A dense model with layers in
RAM is much slower, under 5 tok/s.</p>
<p><code>localllm doctor</code> prints the estimate for your PC: how much RAM the picked model uses and how much is
left for other apps.</p>
""", [("How much RAM does a local LLM need?",
       f"With the model fully on the GPU and caches sized for one user, about {RAM_G['localllm_gb']:.1f}-"
       f"{RAM_Q['localllm_gb']:.1f} GB of system RAM. With llama-server's defaults it grows to about "
       f"{RAM_Q['default_gb']:.1f} GB over a 30-turn chat, at the same speed.")])

    guide("local-ai-vs-cloud-ai", "Local AI or cloud AI: when to use which",
          "A local model on your GPU is private, free per token and works offline; a frontier cloud model knows more "
          "and reasons better. Most people need both. Here is how to split the work, with our measurements.", f"""
<h2>What a local model is good at today</h2>
<p>On a 16 GB card, the best models we measured score {G4['scores']['en/global']}% on translated world-knowledge
questions in English and {G4['scores']['en/math']}% on grade-school math, at {G4['tok_s_9070xt']} tok/s. That is enough
for everyday writing, summaries, translation drafts, explaining code, and questions about your own documents.</p>
<ul>
<li><b>Private by construction.</b> Nothing leaves the PC; useful for client documents, health or finance notes, code
under NDA.</li>
<li><b>No per-token bill.</b> The cost is the electricity and the GPU you already have.</li>
<li><b>Works offline</b>, on a plane or behind a strict firewall.</li>
<li><b>Predictable.</b> The model doesn't change under you; you pick when to upgrade.</li>
</ul>

<h2>Where the cloud still wins</h2>
<ul>
<li>Hard multi-step reasoning, large refactors across many files, and questions that need the newest knowledge.</li>
<li>Very long contexts: a 16 GB card holds about 78k tokens next to the model (<code>--profile max</code>); cloud
models take far more.</li>
<li>Lower-resource languages at small sizes: quantization costs Thai, Hindi and Arabic more than English (13 vs 8
points at 2 bits on Qwen3.8-27B), so check the <a href="/leaderboard/">scores in your language</a>.</li>
</ul>

<h2>Using both</h2>
<p><code>localllm route</code> can send a request to your own cloud API key when a rule says so (a prompt too long for
the local model, a language below the quality floor). It is off by default, the cost and destination are shown, and
every answer is labelled with where it came from. Keys stay in your OS keychain.</p>
<p><code>localllm savings</code> shows how many tokens your PC answered this month and, at the prices you enter, what
they would have cost in the cloud.</p>
<p>One thing localllm will not do: pass a Claude or ChatGPT <em>subscription</em> login on to another tool. Use an API
key; the providers' terms don't allow the other way and it risks your account.</p>
""", [("Is a local LLM as good as ChatGPT or Claude?",
       "Not for hard reasoning or the newest knowledge. On a 16 GB GPU the best local models we measured score about "
       f"{G4['scores']['en/global']}% on translated world-knowledge questions and {G4['scores']['en/math']}% on "
       "grade-school math in English, which is enough for everyday writing, summaries and questions about your own "
       "files, privately and without a per-token bill."),
      ("Can I use a local model and a cloud model together?",
       "Yes. localllm route sends a request to your own cloud API key only when a rule you enable says so, shows the "
       "cost first, and labels every answer with where it came from.")])

    guide("coding-agent-with-a-local-model", "Run a coding agent on a local model (Hermes, OpenCode, Aider, Claude Code)",
          "Three commands point an agent at the model on your GPU, with a context large enough for agent prompts. "
          "What works, what to expect, and how to go back.", f"""
<h2>Set it up</h2>
<pre class="block">localllm serve --profile max     # a context that fills the GPU: agents send long prompts
localllm connect hermes          # or opencode, claude-code, codex, aider, cline
hermes                           # use the agent as usual</pre>
<p><code>connect</code> changes only the base URL and the model name in the agent's own config file, keeps a backup,
and <code>localllm connect hermes --undo</code> restores it. On a 16 GB card <code>--profile max</code> gives about
78k tokens of context with gemma-4-26B-A4B; Hermes asks for 64k.</p>

<h2>What to expect</h2>
<ul>
<li>Speed: about {G4['tok_s_9070xt']} tok/s decode on an RX 9070 XT; long prompts take longer the first time and are
re-used from the prompt cache after that.</li>
<li>Quality: fine for small edits, explanations, tests and scripts. Large multi-file refactors are where a frontier
cloud model is still clearly better.</li>
<li>Agent-by-agent results on our models are still being measured (<a href="{GH}/issues/96">#96</a>).</li>
</ul>

<h2>Check the prompt cache and the savings</h2>
<p><code>localllm savings</code> shows the share of prompt tokens that came from the cache. Agents resend the same
instructions on every call, so this should be high; if it is low, the agent's prompt changes between calls and every
call is slower than it needs to be.</p>
""", [("Can I use Claude Code or OpenCode with a local model?",
       "Yes. Start the model with localllm serve --profile max, then localllm connect claude-code (or opencode, "
       "hermes, codex, aider). It writes the tool's config to use the local endpoint and backs up the old one; "
       "--undo restores it.")])

    guide("obsidian-with-a-local-llm", "Use Obsidian with a local LLM, without sending notes to the cloud",
          "Copilot for Obsidian can talk to the model on your GPU. One command sets it up and tells you which "
          "settings would still send notes to a cloud service.", f"""
<h2>Set it up</h2>
<ol>
<li>In Obsidian, install and enable the <b>Copilot</b> community plugin.</li>
<li>Start the model: <code>localllm serve</code></li>
<li>Run <code>localllm connect obsidian --path "PATH/TO/YOUR/VAULT"</code>, then reload Obsidian.</li>
</ol>
<p>The local model becomes Copilot's default chat model; your other models stay in the list.
<code>--undo --path ...</code> restores the plugin's previous settings.</p>

<h2>Check what still leaves your PC</h2>
<p>The command lists two things to look at: a cloud <b>embedding model</b> for vault search (building the index sends
chunks of your notes to it), and cloud chat models still enabled in the menu. Turn those off in Copilot's settings if
nothing should leave the PC.</p>

<h2>Local search over your notes</h2>
<p>For vault search, run a small local embedding model next to the chat model. localllm already downloads one for its
router, multilingual-e5-small (126 MB, about 100 languages). The exact commands are in the
<a href="{GH}/blob/main/docs/apis.md#obsidian">API docs</a>.</p>
""", [("Can Obsidian use a local LLM?",
       "Yes, through the Copilot for Obsidian plugin. localllm connect obsidian --path YOUR_VAULT makes the local model "
       "the default and lists any setting that would still send notes to a cloud service.")])

    cards = "".join(f'<div class="card"><h3><a href="/guides/{s}/">{t}</a></h3><p class="muted">{d}</p></div>'
                    for s, t, d, _ in GUIDES)
    body = page_head("Guides", "Guides", "Answers to the questions people ask most about running LLMs locally, "
                     "with the measurements behind them.") + \
        f'<section><div class="wrap"><div class="cards">{cards}</div></div></section>'
    layout("/guides/", "Guides", "Which local LLM fits a 16 GB GPU, why llama.cpp is slow on AMD/Intel Windows, the "
           "best local model for your language, how much RAM a local LLM needs, local vs cloud AI, coding agents and "
           "Obsidian on a local model.", body)


# ---- Thai home -------------------------------------------------------------------------------------------------------

def thai() -> None:
    body = f"""
<section class="hero"><div class="wrap hero-grid">
<div>
<div class="kicker">โอเพนซอร์ส · Windows, Linux, macOS</div>
<h1>รัน AI ในเครื่องตัวเอง ด้วยโมเดลที่ดีที่สุดเท่าที่การ์ดจอรับไหว ในคำสั่งเดียว</h1>
<p class="lead"><code>localllm</code> ดูการ์ดจอกับ RAM ของคุณ แล้วเลือกโมเดลที่ได้คะแนนสูงสุดในภาษาไทยจากที่เราวัดไว้
ซึ่งใส่การ์ดได้พอดี จากนั้นดาวน์โหลดและเปิดใช้ด้วย llama.cpp ที่ตั้งค่าไว้แล้ว ฟรี ใช้ออฟไลน์ได้หลังดาวน์โหลดครั้งแรก
ข้อมูลไม่ออกจากเครื่อง</p>
{INSTALL}
<div class="btns"><a class="btn primary" href="/th/get-started/">เริ่มใช้งาน</a><a class="btn" href="{GH}">ดูบน GitHub</a></div>
<p class="small muted">ต้องมี Python 3.9 ขึ้นไป โมเดลที่มีตอนนี้ต้องใช้การ์ดจอ VRAM 10 GB ขึ้นไป
ถ้าไม่แน่ใจ ให้รัน <code>localllm doctor</code> ก่อน จะบอกว่าการ์ดของคุณรันอะไรได้บ้าง</p>
</div>
<div>{doctor_sample()}</div>
</div></section>

<section><div class="wrap">
<h2>ทำไมต้องเรื่องภาษา</h2>
<p class="narrow">โมเดลที่เก่งภาษาอังกฤษไม่ได้แปลว่าจะเก่งภาษาไทยเท่ากัน และการบีบไฟล์ (quantization) ทำให้ภาษาไทยเสียคะแนนมากกว่า
จากที่เราวัด Qwen3.8-27B ตอนบีบเหลือ 2 บิต ภาษาอังกฤษตก 8 คะแนน แต่ภาษาไทยตก 13 คะแนน
localllm จึงเก็บคะแนนแยกตามภาษา และเลือกให้ตามภาษาของคุณ</p>
<div class="tbl"><table>
<tr><th>วัดบน RX 9070 XT 16 GB</th><th class="num">Qwen3.8-27B Q3</th><th class="num">gemma-4-26B-A4B QAT</th></tr>
<tr><td>ข้อสอบจริงของไทย (ThaiExam)</td><td class="num">{Q3['scores']['th/regional']}%</td><td class="num">{G4['scores']['th/regional']}%</td></tr>
<tr><td>โจทย์คณิตศาสตร์ภาษาไทย (MGSM)</td><td class="num">{Q3['scores']['th/math']}%</td><td class="num">{G4['scores']['th/math']}%</td></tr>
<tr><td>แปลไทย-อังกฤษ (chrF++)</td><td class="num">{Q3['scores']['th/translate']}</td><td class="num">{G4['scores']['th/translate']}</td></tr>
<tr><td>ความเร็ว</td><td class="num">~{Q3['tok_s_9070xt']} tok/s</td><td class="num">~{G4['tok_s_9070xt']} tok/s</td></tr>
</table></div>
<p class="small muted" style="margin-top:8px">คะแนนต่างกันไม่ถึง 2 คะแนนถือว่าเสมอ (ค่าคลาดเคลื่อนของข้อสอบประมาณ &plusmn;5) ถ้าเสมอจะเลือกตัวที่เร็วกว่า</p>
</div></section>

<section><div class="wrap">
<h2>ทำอะไรให้บ้าง</h2>
<div class="cards">
<div class="card"><h3>เลือกโมเดลจากผลวัดจริง</h3><p>ทุกโมเดลผ่านข้อสอบหลายภาษาบนเครื่องทดสอบเดียวกัน ไม่ต้องไล่อ่านกระทู้หรือเดาเอง</p></div>
<div class="card"><h3>ใช้ RAM น้อยลงประมาณ 4 เท่า</h3><p>คุยต่อเนื่อง 30 รอบ ใช้ RAM ราว {RAM_G['localllm_gb']:.1f} GB แทน {RAM_G['default_gb']:.1f} GB ของค่าเริ่มต้น llama.cpp ความเร็วเท่าเดิม</p></div>
<div class="card"><h3>การ์ด AMD และ Intel</h3><p>ใช้ Vulkan และแก้ปัญหาช้าบน Windows ที่ไม่ได้เปิด Resizable BAR ให้เฉพาะโมเดลที่วัดแล้วเร็วขึ้นจริง</p></div>
<div class="card"><h3>ต่อกับแอปที่มีอยู่</h3><p>API ที่ <code>127.0.0.1:8080</code> รองรับรูปแบบ OpenAI, Anthropic, Ollama และ Gemini</p></div>
</div>
</div></section>

<section><div class="wrap narrow">
<h2>ช่วยวัดผลบนเครื่องของคุณ</h2>
<p>เรามีเครื่องทดสอบเครื่องเดียว (Windows + RX 9070 XT) ถ้าคุณใช้ NVIDIA, Mac หรือการ์ดรุ่นอื่น ใช้เวลา 10-30 นาที:</p>
{term('<span class="p">$ </span>localllm tune' + chr(10) + '<span class="p">$ </span>localllm eval --langs en,th' + chr(10) + '<span class="p">$ </span>localllm report', "แชร์ผลวัด", "report-th")}
<p class="small muted"><code>report</code> จะเปิด issue บน GitHub ที่กรอกข้อมูลไว้ให้ ไม่มีชื่อผู้ใช้ path หรือ key อ่านทวนได้ก่อนส่ง
ผลจะขึ้นใน <a href="/leaderboard/#th">ตารางคะแนน</a></p>
</div></section>
"""
    layout("/th/", "localllm: รัน AI ในเครื่องตัวเอง เลือกโมเดลจากคะแนนภาษาไทยที่วัดจริง",
           "localllm เลือก ดาวน์โหลด และรันโมเดล AI ในเครื่องที่ได้คะแนนภาษาไทยสูงสุดและใส่การ์ดจอของคุณได้ "
           "รองรับ AMD, NVIDIA, Intel และ Apple ฟรีและโอเพนซอร์ส", body, lang="th")


def thai_start() -> None:
    body = page_head("เริ่มต้นใช้งาน", "ติดตั้งและเปิดใช้", "สองคำสั่ง ครั้งแรกจะดาวน์โหลด llama.cpp (~30 MB) และโมเดลหนึ่งตัว "
                     "(8-14 GB) ครั้งเดียว ก่อนดาวน์โหลดโมเดลจะแสดงว่าเครื่องคุณรันอะไรได้และถามก่อน") + f"""
<section><div class="wrap two">
<div class="prose">
<h2>ติดตั้ง</h2>
<ol class="steps">
<li><div>มี Python 3.9 ขึ้นไป (เช็กด้วย <code>python --version</code>) บน Windows ตอนติดตั้งให้ติ๊ก "Add python.exe to PATH"</div></li>
<li><div>ติดตั้ง: <code>pip install make-localllm-easier</code></div></li>
<li><div>พิมพ์ <code>localllm</code> หน้าแชทจะเปิดในเบราว์เซอร์เมื่อโมเดลโหลดเสร็จ</div></li>
</ol>
{INSTALL}
<h3>ไม่มี Python?</h3>
<p>ตั้งแต่เวอร์ชัน 0.7.1 ทุก <a href="{GH}/releases">release</a> จะมีไฟล์ <code>localllm</code> ไฟล์เดียวสำหรับ Windows,
macOS และ Linux ดาวน์โหลดแล้วรันจาก terminal ได้เลย</p>
</div>
<div class="prose">
<h2>คำสั่งที่ใช้บ่อย</h2>
<pre class="block">localllm chat        # แชทใน terminal (พิมพ์ไทยได้)
localllm --ui        # หน้าแชทที่บันทึกบทสนทนาไว้ในเครื่อง
localllm doctor      # การ์ดจอนี้รันโมเดลขนาดไหนได้ เร็วแค่ไหน ใช้ RAM เท่าไร
localllm list        # โมเดลทั้งหมดที่วัดแล้ว พร้อมคะแนนแต่ละภาษา
localllm tune        # วัดค่าที่เร็วที่สุดสำหรับเครื่องนี้ (ครั้งเดียว ไม่กี่นาที)
localllm eval --langs en,th   # วัดคะแนนภาษาไทยของ server ที่รันอยู่</pre>
<p class="small">ใช้กับแอปอื่น: base URL <code>http://127.0.0.1:8080/v1</code> ใส่ API key อะไรก็ได้
(<a href="{GH}/blob/main/docs/apis.md">ตัวอย่าง API</a>)</p>
</div>
</div></section>

<section><div class="wrap narrow">
<h2>ปัญหาที่พบบ่อย</h2>
<details><summary>พิมพ์ <code>localllm</code> แล้วไม่เจอคำสั่ง</summary><p>pip ติดตั้งไว้ในโฟลเดอร์ที่ไม่อยู่ใน PATH
ใช้ <code>python -m localllm</code> แทนได้ ผลเหมือนกัน</p></details>
<details><summary>ขึ้นว่า "no measured model fits this GPU yet"</summary><p>โมเดลที่เล็กที่สุดในตอนนี้ต้องใช้ VRAM ประมาณ
10 GB รัน <code>localllm doctor</code> เพื่อดูว่าการ์ดของคุณใส่โมเดลขนาดไหนได้</p></details>
<details><summary>ช้ากว่าตัวเลขบนเว็บ</summary><p>รัน <code>localllm tune</code> ถ้าเป็นการ์ด AMD/Intel บน Windows
ให้เปิด Resizable BAR ใน BIOS (<a href="/guides/llama-cpp-slow-on-amd-intel-windows/">อ่านเพิ่ม</a>) และถ้า doctor
เตือนว่าโมเดลล้นไปใช้ RAM ให้ปิดโปรแกรมที่ใช้การ์ดจอหนัก ๆ</p></details>
<details><summary>ไฟล์อยู่ที่ไหน ลบยังไง</summary><p>ทุกอย่างอยู่ใน <code>~/.localllm/</code> (โมเดล, llama.cpp,
บทสนทนาที่บันทึก) ลบโฟลเดอร์นี้ก็ลบหมด</p></details>
<p>เจอปัญหาอื่น <a href="{ISSUES}">เปิด issue</a> แนบผลของ <code>localllm doctor</code> มาด้วย (เขียนภาษาไทยได้)</p>
</div></section>
"""
    layout("/th/get-started/", "เริ่มต้นใช้งาน localllm: ติดตั้งและรัน AI ในเครื่อง",
           "ติดตั้งด้วย pip install make-localllm-easier แล้วพิมพ์ localllm คำสั่งที่ใช้บ่อย และวิธีแก้ปัญหาที่พบบ่อย",
           body, lang="th")


def not_found() -> None:
    body = page_head("404", "Page not found", 'Try the <a href="/">home page</a> or the <a href="/guides/">guides</a>.')
    layout("/404/", "Page not found", "This page does not exist.", body)


def write() -> None:
    OUT.mkdir(exist_ok=True)
    for old in OUT.iterdir():                          # empty it, keep the folder (a local server may have it open)
        shutil.rmtree(old) if old.is_dir() else old.unlink()
    (OUT / "assets").mkdir()
    for f in (SITE / "assets").iterdir():
        shutil.copy2(f, OUT / "assets" / f.name)
    for path, text in pages.items():
        dest = OUT / "404.html" if path == "/404/" else OUT / path.strip("/") / "index.html"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8", newline="\n")
    urls = "".join(f"<url><loc>{URL}{p}</loc></url>\n" for p in pages if p != "/404/")
    (OUT / "sitemap.xml").write_text('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/'
                                     f'schemas/sitemap/0.9">\n{urls}</urlset>\n', encoding="utf-8", newline="\n")
    (OUT / "robots.txt").write_text(f"User-agent: *\nAllow: /\nSitemap: {URL}/sitemap.xml\n", encoding="utf-8",
                                    newline="\n")
    shutil.copy2(ROOT / "llms.txt", OUT / "llms.txt")
    for f in (SITE / "static").glob("*") if (SITE / "static").exists() else []:   # e.g. a Search Console file
        shutil.copy2(f, OUT / f.name)
    (OUT / "_headers").write_text("/assets/*\n  Cache-Control: public, max-age=3600\n/*\n"
                                  "  X-Content-Type-Options: nosniff\n  Referrer-Policy: strict-origin-when-cross-origin\n",
                                  encoding="utf-8", newline="\n")


def main() -> None:
    for build in (home, features, how, uses, compare, start, leaderboard, guides, thai, thai_start, not_found):
        build()
    write()
    print(f"{len(pages)} pages -> {OUT}  ({URL})")


if __name__ == "__main__":
    main()
