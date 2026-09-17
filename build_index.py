# INDEX BUILDER - landing page for the generated reports
#
# Scans the output/ directory for the report HTML files, reads each one's
# title and summary stats (from the embedded JSON), and writes a styled
# index.html that links to them. Intended as the entry point when publishing
# the reports (e.g. to Lilypad).
#
# The index self-heals: it lists whatever reports currently exist, so adding a
# new report and re-running this picks it up automatically.
#
# Usage:
#   python build_index.py                 # scans output/
#   python build_index.py --dir output --out output/index.html

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

# Report file name -> (display title, one-line description). Files not listed
# here still appear, using their embedded <title>.
KNOWN = {
    "enterprise_risks_report.html": "Enterprise risks — full history clustering, trends, and taxonomy-gap analysis.",
    "emerging_risks_report.html": "Emerging risks — full history clustering, trends, and taxonomy-gap analysis.",
    "enterprise_risks_report_last30d.html": "Enterprise risks — most recent 30 days only.",
    "emerging_risks_report_last30d.html": "Emerging risks — most recent 30 days only.",
}

# Preferred display order (unlisted files sorted after, alphabetically).
ORDER = [
    "emerging_risks_report.html",
    "enterprise_risks_report.html",
    "emerging_risks_report_last30d.html",
    "enterprise_risks_report_last30d.html",
]


def _extract(html):
    """Pull the <title> and embedded stats/months from a report HTML file."""
    title_m = re.search(r"<title>(.*?)</title>", html, re.S)
    title = title_m.group(1).strip() if title_m else "Risk Topic Report"

    stats, months = {}, []
    line = next((l for l in html.splitlines() if l.startswith("const DATA = ")), None)
    if line:
        try:
            raw = line[len("const DATA = "):].rstrip(";").replace("<\\/", "</")
            data = json.loads(raw)
            stats = data.get("stats", {}) or {}
            tr = data.get("trends")
            if tr and tr.get("months"):
                months = tr["months"]
        except (ValueError, KeyError):
            pass
    return title, stats, months


def collect(report_dir):
    reports = []
    for path in Path(report_dir).glob("*_report*.html"):
        if path.name == "index.html":
            continue
        html = path.read_text(encoding="utf-8", errors="replace")
        title, stats, months = _extract(html)
        reports.append({
            "file": path.name,
            "title": title,
            "desc": KNOWN.get(path.name, ""),
            "stats": stats,
            "months": months,
        })

    def sort_key(r):
        return (ORDER.index(r["file"]) if r["file"] in ORDER else len(ORDER), r["file"])

    return sorted(reports, key=sort_key)


def _card(r):
    s = r["stats"]
    months = r["months"]
    coverage = ""
    if months:
        coverage = f"{months[0]} → {months[-1]} · {len(months)} months"
    bits = []
    if s.get("totalArticles"):
        bits.append(f'{s["totalArticles"]:,} articles')
    if s.get("topicCount"):
        bits.append(f'{s["topicCount"]:,} topics')
    metrics = " · ".join(bits)

    return f"""    <a class="card" href="{r['file']}">
      <div class="card-title">{r['title']}</div>
      {f'<div class="card-desc">{r["desc"]}</div>' if r['desc'] else ''}
      <div class="card-meta">{metrics}</div>
      {f'<div class="card-cov">Coverage: {coverage}</div>' if coverage else ''}
      <div class="card-go">Open report →</div>
    </a>"""


def build_html(reports):
    cards = "\n".join(_card(r) for r in reports) or \
        '<div class="empty">No reports found. Run build_report.py first.</div>'
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    return _TEMPLATE.replace("__CARDS__", cards).replace("__GENERATED__", generated)


def run(report_dir, out_path):
    reports = collect(report_dir)
    html = build_html(reports)
    out = Path(out_path)
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Indexed {len(reports)} report(s):")
    for r in reports:
        print(f"  - {r['file']}  ({r['title']})")
    print(f"Wrote index -> {out}")


def parse_args():
    p = argparse.ArgumentParser(description="Build a landing-page index for the reports.")
    p.add_argument("--dir", default="reports", help="Directory to scan (default: reports).")
    p.add_argument("--out", help="Output index path (default: <dir>/index.html).")
    return p.parse_args()


def main():
    args = parse_args()
    out = args.out or str(Path(args.dir) / "index.html")
    run(args.dir, out)


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Risk Signal Reports</title>
<style>
  :root {
    --bg: #0f1720; --panel: #172232; --panel2: #1e2c40; --text: #e6edf5;
    --muted: #93a4bb; --border: #2a3a52; --accent: #58a6ff;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
    font: 15px/1.6 -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
  header { padding: 40px 24px 24px; max-width: 900px; margin: 0 auto; }
  h1 { margin: 0 0 8px; font-size: 26px; }
  .lede { color: var(--muted); font-size: 15px; max-width: 70ch; }
  main { max-width: 900px; margin: 0 auto; padding: 8px 24px 60px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }
  .card { display: block; text-decoration: none; color: inherit;
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 20px; transition: border-color .15s, transform .15s; }
  .card:hover { border-color: var(--accent); transform: translateY(-2px); }
  .card-title { font-size: 17px; font-weight: 650; margin-bottom: 6px; }
  .card-desc { color: var(--muted); font-size: 13.5px; margin-bottom: 12px; }
  .card-meta { font-size: 13px; color: var(--text); }
  .card-cov { font-size: 12.5px; color: var(--muted); margin-top: 2px; }
  .card-go { color: var(--accent); font-size: 13px; font-weight: 600; margin-top: 14px; }
  .empty { color: var(--muted); padding: 40px; text-align: center; }
  .how { background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 18px 20px; margin-top: 24px; color: var(--muted); font-size: 13.5px; max-width: 70ch; }
  .how b { color: var(--text); }
  footer { max-width: 900px; margin: 0 auto; padding: 20px 24px 40px;
    color: var(--muted); font-size: 12px; border-top: 1px solid var(--border); }
</style>
</head>
<body>
<header>
  <h1>Risk Signal Reports</h1>
  <p class="lede">News coverage turned into risk signals: articles are fetched per risk, scored
  for sentiment, enriched with spaCy, embedded and clustered into topics with BERTopic, tracked
  month over month for signal intensity, and scored for how well each topic fits the risk that
  surfaced it. Open a report below and see its "How this was built" tab for detail.</p>
</header>
<main>
  <div class="grid">
__CARDS__
  </div>
  <div class="how">
    <b>What to look for.</b> Each report has four views: <b>Topics</b> (browse and search all
    clusters), <b>Trends over time</b> (which themes are growing per risk), <b>Candidate gaps</b>
    (topics that are intense and recent but fit their risk poorly — possible blind spots), and
    <b>How this was built</b> (the full methodology).
  </div>
</main>
<footer>Generated __GENERATED__ · self-contained HTML, no external dependencies.</footer>
</body>
</html>
"""


if __name__ == "__main__":
    main()
