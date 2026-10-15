"""Weekly traction snapshot (#148): recorded API answers, no network."""
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import traction  # noqa: E402

DAY = dt.date(2026, 10, 19)                                    # a Monday, ISO week 43
ANSWERS = {
    f"{traction.API}/repos/{traction.REPO}": {"stargazers_count": 3, "forks_count": 2, "subscribers_count": 1,
                                              "open_issues_count": 40},
    f"{traction.API}/repos/{traction.REPO}/traffic/clones": {"count": 2005, "uniques": 441},
    f"{traction.API}/repos/{traction.REPO}/traffic/views": {"count": 310, "uniques": 52},
    f"https://pypistats.org/api/packages/{traction.PACKAGE}/overall?mirrors=false": {"data": [
        {"date": "2026-09-01", "downloads": 1000}, {"date": "2026-10-01", "downloads": 50},
        {"date": "2026-10-13", "downloads": 20}, {"date": "2026-10-19", "downloads": 7},
        {"date": "2026-10-20", "downloads": 99}]},
    f"{traction.API}/repos/{traction.REPO}/releases?per_page=100": [
        {"tag_name": "v0.7.1", "assets": [{"download_count": 12}, {"download_count": 5}]},
        {"tag_name": "v0.6.1", "assets": []}],
}


def fake(url):
    if url.startswith(f"{traction.API}/search/issues"):
        return {"total_count": 4 if "label%3Aresults" in url else 1}
    if url not in ANSWERS:
        raise OSError(f"no recorded answer for {url}")
    return ANSWERS[url]


def test_snapshot_from_recorded_answers(tmp_path):
    board = tmp_path / "leaderboard.json"
    board.write_text(json.dumps({"reports": 2}))
    s = traction.snapshot(fake, DAY, board)
    assert s["week"] == "2026-W43" and s["date"] == "2026-10-19" and "errors" not in s
    assert s["github"] == {"stars": 3, "forks": 2, "watchers": 1, "open_issues": 40}
    assert s["traffic"] == {"clones": {"count": 2005, "uniques": 441}, "views": {"count": 310, "uniques": 52}}
    assert s["pypi"] == {"last_7d": 27, "last_30d": 77}            # in the window, nothing after today
    assert s["releases"] == {"per_tag": {"v0.7.1": 17, "v0.6.1": 0}, "total": 17}
    assert s["issues"] == {"report_issues": 4, "outside_issues": 1} and s["reports"] == 2


def test_a_dead_source_is_null_with_its_error_not_a_guess(tmp_path):
    def no_traffic(url):
        if "/traffic/" in url:
            raise OSError("HTTP 403: Must have push access")
        return fake(url)
    s = traction.snapshot(no_traffic, DAY, tmp_path / "missing.json")
    assert s["traffic"] is None and "403" in s["errors"]["traffic"]
    assert s["reports"] is None and "reports" in s["errors"] and s["github"]["stars"] == 3


def test_one_line_per_iso_week(tmp_path):
    out = tmp_path / "data" / "metrics.jsonl"
    s = traction.snapshot(fake, DAY, tmp_path / "missing.json")
    assert traction.append(s, out) and not traction.append(s, out)            # same week: skipped
    later = traction.snapshot(fake, DAY + dt.timedelta(days=7), tmp_path / "missing.json")
    assert traction.append(later, out)
    weeks = [json.loads(line)["week"] for line in out.read_text(encoding="utf-8").splitlines()]
    assert weeks == ["2026-W43", "2026-W44"]


def test_only_github_gets_the_token(monkeypatch):
    seen = []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, *a):
            return b"{}"

    monkeypatch.setenv("GITHUB_TOKEN", "t0ken")
    monkeypatch.setattr(traction.urllib.request, "urlopen", lambda req, timeout: seen.append(req) or Resp())
    traction.http_json(f"{traction.API}/repos/x")
    traction.http_json("https://pypistats.org/api/packages/x/overall")
    assert seen[0].get_header("Authorization") == "Bearer t0ken" and seen[1].get_header("Authorization") is None
