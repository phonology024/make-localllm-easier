// Accessibility check of the chat page with axe-core in a real browser (#121).
//
//   node tools/a11y_check.mjs localllm/ui.html STRINGS.json      (STRINGS.json: {"en": i18n.table("en"), "ar": ...})
//
// The page is loaded with the server's endpoints mocked (no localllm running), then axe-core checks it in light and
// dark mode, English and Arabic (right to left), in each state: chat with messages, the Models tab, the key dialog.
// Exit code 1 on any serious or critical violation (WCAG 2.1 A/AA rules, contrast included). CI: .github/workflows/a11y.yml
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { chromium } from "playwright-core";

const require = createRequire(import.meta.url);
const AXE = readFileSync(require.resolve("axe-core/axe.min.js"), "utf8");
const [pagePath, stringsPath] = process.argv.slice(2);
const PAGE = readFileSync(pagePath, "utf8");
const STRINGS = JSON.parse(readFileSync(stringsPath, "utf8"));

const MODELS = {lang: "th", vram_gb: 12, ram_free_gb: 20, best: "gemma4-26b-a4b-qat", active: ["qwen3.8-27b-iq2"], models: [
  {key: "qwen3.8-27b-iq2", gb: 7.8, note: "for 10-12 GB cards", fit: "fits", experts_in_ram: 0, tok_s: 40, score: 54.2,
   badges: {regional: 54.2}, below_floor: true, qat: false, downloaded: true, active: true, download: null},
  {key: "gemma4-26b-a4b-qat", gb: 13.3, note: "MoE", fit: "spill", experts_in_ram: 8, tok_s: 45, score: 77.7,
   badges: {regional: 65.7, math: 89.6, translate: 55.6}, below_floor: false, qat: true, downloaded: false, active: false,
   download: {state: "downloading", error: null, pct: 37.5}},
  {key: "qwen3.8-27b-q3", gb: 12.2, note: "dense 27B", fit: "no", experts_in_ram: 0, tok_s: null, score: 77.2,
   badges: {regional: 67.1}, below_floor: false, qat: false, downloaded: false, active: false, download: null}]};
const CHAT = {id: "c1", title: "Hello", files: [{id: "0123456789abcdef", name: "warranty.pdf", chunks: 3}],   // #113
  messages: [{role: "user", content: "What is `localllm`?"},
  {role: "assistant", content: "A launcher.\n```\nlocalllm doctor\n```", meta: "qwen3.8-27b-iq2 · 40.0 tok/s · 1.2 s",
   sources: [{n: 1, name: "warranty.pdf", loc: "p.2", text: "The battery is covered for 24 months."}]}]};

function mock(lang, locked) {
  const routes = {"/localllm/i18n": STRINGS[lang], "/localllm/models": MODELS, "/localllm/sessions/c1": CHAT,
    "/localllm/sessions": [{id: "c1", title: "Hello", updated: 1}],
    "/localllm/catalog": {models: [], choosable: false, single: "qwen3.8-27b-iq2"},
    "/localllm/status": {loaded: true, running: "qwen3.8-27b-iq2", vram_gb: 9.1, ram_gb: 2.2, spill_gb: 3, spill_warn_gb: 1.5}};
  return `<script>window.fetch = async (p) => {
    const path = p.split("?")[0];
    if (${locked} && path !== "/localllm/i18n") return new Response("{}", {status: 401});
    const body = ${JSON.stringify(routes)}[path];
    return new Response(JSON.stringify(body === undefined ? {} : body), {headers: {"Content-Type": "application/json"}});
  };</script>`;
}

const STATES = {
  chat: async (p) => { await p.click("#sessions .sess-open"); },
  models: async (p) => { await p.click("#tab-models"); await p.waitForSelector(".mcard"); },
  key: async () => {},
};

const browser = await chromium.launch(process.env.A11Y_CHANNEL ? {channel: process.env.A11Y_CHANNEL} : {});
let bad = 0;
for (const scheme of ["light", "dark"]) {
  for (const lang of ["en", "ar"]) {
    for (const [state, enter] of Object.entries(STATES)) {
      const page = await browser.newPage({colorScheme: scheme});
      await page.setContent(PAGE.replace("<script>", mock(lang, state === "key") + "<script>"), {waitUntil: "load"});
      await page.waitForTimeout(300);
      await enter(page);
      await page.waitForTimeout(200);
      await page.addScriptTag({content: AXE});
      const res = await page.evaluate(async () => await axe.run(document, {
        runOnly: {type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "best-practice"]}}));
      const serious = res.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
      for (const v of res.violations)
        console.log(`${scheme}/${lang}/${state}: [${v.impact}] ${v.id}: ${v.help} -> ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(" | ")}`);
      bad += serious.length;
      await page.close();
    }
  }
}
await browser.close();
console.log(bad ? `${bad} serious/critical violation(s)` : "no serious or critical violations");
process.exit(bad ? 1 : 0);
