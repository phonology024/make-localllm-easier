// Menu toggle, copy buttons, leaderboard language filter. No tracking, no external requests.
(function () {
  var btn = document.querySelector(".menu-btn");
  var nav = document.querySelector(".nav");
  if (btn && nav) {
    btn.addEventListener("click", function () {
      var open = nav.classList.toggle("open");
      btn.setAttribute("aria-expanded", open ? "true" : "false");
    });
  }

  document.querySelectorAll("[data-copy]").forEach(function (b) {
    b.addEventListener("click", function () {
      var text = document.getElementById(b.getAttribute("data-copy")).innerText
        .split("\n").map(function (l) { return l.replace(/^\$ /, ""); })
        .filter(function (l) { return l && l.charAt(0) !== "#"; }).join("\n");
      var done = function () { b.textContent = "copied"; setTimeout(function () { b.textContent = "copy"; }, 1500); };
      if (navigator.clipboard) { navigator.clipboard.writeText(text).then(done, function () {}); }
    });
  });

  var board = document.getElementById("board");
  if (!board) return;
  var data = JSON.parse(document.getElementById("board-data").textContent);
  var langSel = document.getElementById("lang");
  var srcSel = document.getElementById("src");
  var names = data.languages;
  var suites = [["global", "Knowledge (translated)", "%"], ["regional", "Local exams", "%"],
                ["math", "Math", "%"], ["translate", "Translation", "chrF++"]];

  // report fields are written by whoever shared the report: never put them into HTML unescaped
  function esc(v) {
    return String(v == null ? "" : v).replace(/[&<>"']/g, function (c) {
      return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c];
    });
  }

  function cell(v, best) {
    if (typeof v !== "number") return '<td class="num muted">-</td>';
    return '<td class="num' + (best ? " best" : "") + '">' + v.toFixed(1) + "</td>";
  }

  function render() {
    var lang = langSel.value, src = srcSel.value;
    var rows = data.entries.filter(function (e) {
      if (src !== "all" && e.source !== src) return false;
      return suites.some(function (s) { return e.scores[lang + "/" + s[0]] !== undefined; });
    });
    var best = {};
    suites.forEach(function (s) {
      var k = lang + "/" + s[0];
      rows.forEach(function (e) { if (e.scores[k] !== undefined) best[k] = Math.max(best[k] || 0, e.scores[k]); });
    });
    var head = "<tr><th>Model</th><th>GPU</th><th>OS</th><th class=\"num\">Decode tok/s</th>" +
      suites.map(function (s) { return '<th class="num">' + s[1] + "<br><span class=\"muted small\">" + s[2] + "</span></th>"; }).join("") +
      "<th>Source</th></tr>";
    var body = rows.map(function (e) {
      return "<tr><td><code>" + esc(e.model) + "</code></td><td>" + esc(e.gpu) + " <span class=\"muted small\">" +
        esc(e.vram_gb) + " GB</span></td><td>" + esc(e.os) + "</td>" +
        '<td class="num">' + esc(e.tok_s || "-") + "</td>" +
        suites.map(function (s) { var k = lang + "/" + s[0]; return cell(e.scores[k], e.scores[k] === best[k] && rows.length > 1); }).join("") +
        '<td><span class="tag' + (e.source === "maintainers" ? " ours" : "") + '">' + esc(e.source) + "</span>" +
        (e.host ? ' <span class="tag" title="rented cloud machine">' + esc(e.host) + "</span>" : "") +
        (e.flags.length ? ' <span class="small" title="' + esc(e.flags.join("; ")) + '">(check)</span>' : "") + "</td></tr>";
    }).join("");
    board.innerHTML = "<table><thead>" + head + "</thead><tbody>" +
      (body || '<tr><td colspan="9" class="muted">Nothing measured in ' + esc(names[lang] || lang) +
        " yet. Run <code>localllm eval --langs en," + esc(lang) + "</code> and share it with <code>localllm report</code>.</td></tr>") +
      "</tbody></table>";
    try { history.replaceState(null, "", "#" + lang); } catch (e) {}
  }

  var start = (location.hash || "").slice(1);
  if (start && langSel.querySelector('option[value="' + start + '"]')) langSel.value = start;
  langSel.addEventListener("change", render);
  srcSel.addEventListener("change", render);
  render();
})();
