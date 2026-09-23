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
}

# Preferred display order (unlisted files sorted after, alphabetically).
ORDER = [
    "emerging_risks_report.html",
    "enterprise_risks_report.html",
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


def _discovery_card(discovery_file):
    """A distinct card linking to the discovery explainer page."""
    return f"""    <a class="card discovery" href="{discovery_file}">
      <div class="card-title">Risk Discovery — finding risks off the list</div>
      <div class="card-desc">How we collect broad, un-keyworded news to surface emerging
      risks the current taxonomy doesn't cover yet.</div>
      <div class="card-meta">What we're collecting &amp; why</div>
      <div class="card-go">Learn more →</div>
    </a>"""


def build_html(reports, discovery_file=None):
    cards = "\n".join(_card(r) for r in reports) or \
        '<div class="empty">No reports found. Run build_report.py first.</div>'
    if discovery_file:
        cards += "\n" + _discovery_card(discovery_file)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    return _TEMPLATE.replace("__CARDS__", cards).replace("__GENERATED__", generated)


def _discovery_stats():
    """Live stats about the discovery corpus, if it exists."""
    path = Path("output/discovery_news.csv")
    stats = {"exists": path.exists(), "total": 0, "by_method": {}, "sources": 0,
             "latest": ""}
    if not path.exists():
        return stats
    try:
        import pandas as pd
        df = pd.read_csv(path)
        stats["total"] = len(df)
        stats["sources"] = int(df["SOURCE"].nunique()) if "SOURCE" in df else 0
        if "SEARCH_TERM_ID" in df:
            stats["by_method"] = df["SEARCH_TERM_ID"].value_counts().to_dict()
        if "PUBLISHED_DATE" in df:
            d = pd.to_datetime(df["PUBLISHED_DATE"], errors="coerce").dropna()
            if len(d):
                stats["latest"] = str(d.max())[:10]
    except Exception:
        pass
    return stats


def build_discovery_page(out_dir):
    """Write reports/discovery.html explaining the risk-discovery collection."""
    try:
        from discover_fetch import RISK_LEXICON, DEFAULT_CATEGORIES
    except Exception:
        RISK_LEXICON, DEFAULT_CATEGORIES = [], []

    s = _discovery_stats()
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    lex_chips = "".join(f'<span class="chip">{t}</span>' for t in RISK_LEXICON)
    cat_chips = "".join(f'<span class="chip">{c}</span>' for c in DEFAULT_CATEGORIES)

    if s["exists"]:
        method_bits = " · ".join(f"{v:,} {k.lower()}" for k, v in s["by_method"].items())
        corpus = (f'<div class="statline"><b>{s["total"]:,}</b> articles collected so far'
                  f' · {s["sources"]:,} distinct sources'
                  f'{" · latest " + s["latest"] if s["latest"] else ""}</div>'
                  f'<div class="statline muted">By method: {method_bits}</div>')
    else:
        corpus = ('<div class="statline muted">No discovery corpus yet — it builds up '
                  'as the scheduled fetches run.</div>')

    html = _DISCOVERY_TEMPLATE
    html = html.replace("__LEX_CHIPS__", lex_chips or "<span class='muted'>(none)</span>")
    html = html.replace("__CAT_CHIPS__", cat_chips or "<span class='muted'>(none)</span>")
    html = html.replace("__LEX_COUNT__", str(len(RISK_LEXICON)))
    html = html.replace("__CORPUS__", corpus)
    html = html.replace("__GENERATED__", generated)

    out = Path(out_dir) / "discovery.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote discovery page -> {out}")
    return out


def run(report_dir, out_path):
    reports = collect(report_dir)
    # Build the discovery explainer page first so the index can link to it.
    disco = build_discovery_page(report_dir)
    html = build_html(reports, discovery_file=disco.name if disco else None)
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
<link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700;800&family=Lora:ital,wght@0,400;1,400&display=swap" rel="stylesheet">
<style>
  /* FINRA brand palette (light theme) */
  :root {
    --core: #233E66; --accent: #0082D1; --gray: #595959;
    --green: #9EC405; --yellow: #FFCF40; --red: #FB483D;
    --bg: #f4f6f9; --panel: #ffffff; --text: #333;
    --muted: #6b7280; --border: #e2e6ec;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
    font: 15px/1.6 'Open Sans', -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
  .logoband { background: #fff; padding: 14px 40px; }
  .logoband .logo { height: 46px; width: auto; display: block; }
  .brandbar { height: 6px; background: var(--yellow); }
  header { background: var(--core); color: #fff; padding: 26px 40px; }
  header .wrap { max-width: 1000px; margin: 0 auto; }
  h1 { margin: 0 0 6px; font-size: 26px; font-weight: 800; }
  .tagline { color: #9db4d6; font-family: 'Lora', Georgia, serif; font-style: italic; font-size: 13px; margin-bottom: 10px; }
  .lede { color: #cfdaea; font-size: 15px; max-width: 74ch; }
  main { max-width: 1000px; margin: 0 auto; padding: 22px 40px 60px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }
  .card { display: block; text-decoration: none; color: inherit;
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 20px; box-shadow: 0 1px 2px rgba(16,36,66,.05); transition: border-color .15s, transform .15s, box-shadow .15s; }
  .card:hover { border-color: var(--accent); transform: translateY(-2px); box-shadow: 0 4px 12px rgba(16,36,66,.10); }
  .card-title { font-size: 17px; font-weight: 700; color: var(--core); margin-bottom: 6px; }
  .card-desc { color: var(--muted); font-size: 13.5px; margin-bottom: 12px; }
  .card-meta { font-size: 13px; color: var(--text); }
  .card-cov { font-size: 12.5px; color: var(--muted); margin-top: 2px; }
  .card-go { color: var(--accent); font-size: 13px; font-weight: 700; margin-top: 14px; }
  .card.discovery { border-left: 4px solid var(--green); }
  .card.discovery:hover { border-color: var(--green); border-left-color: var(--green); }
  .empty { color: var(--muted); padding: 40px; text-align: center; }
  .how { background: #eef3fa; border: 1px solid var(--border); border-left: 4px solid var(--accent);
    border-radius: 8px; padding: 16px 20px; margin-top: 24px; color: #2b3a52; font-size: 13.5px; max-width: 78ch; }
  .how b { color: var(--core); }
  footer { max-width: 1000px; margin: 0 auto; padding: 20px 40px 40px;
    color: var(--muted); font-size: 12px; border-top: 1px solid var(--border); }
</style>
</head>
<body>
<div class="logoband"><img class="logo" src="finra%20logo.png" alt="FINRA Enterprise Risk Management"></div>
<div class="brandbar"></div>
<header>
  <div class="wrap">
    <h1>Risk Signal Reports</h1>
    <div class="tagline">Investor protection. Market integrity.</div>
    <p class="lede">News coverage turned into risk signals: articles are fetched per risk, scored
    for sentiment, enriched with spaCy, embedded and clustered into topics with BERTopic, tracked
    month over month for signal intensity, and scored for how well each topic fits the risk that
    surfaced it. Open a report below and see its "How this was built" tab for detail.</p>
  </div>
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


_DISCOVERY_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Risk Discovery — What We're Collecting</title>
<link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700;800&family=Lora:ital,wght@0,400;1,400&display=swap" rel="stylesheet">
<style>
  /* FINRA brand palette (light theme) */
  :root {
    --core: #233E66; --accent: #0082D1; --gray: #595959;
    --green: #9EC405; --yellow: #FFCF40; --red: #FB483D;
    --bg: #f4f6f9; --panel: #ffffff; --panel2: #eef3fa; --text: #333;
    --muted: #6b7280; --border: #e2e6ec;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
    font: 15px/1.65 'Open Sans', -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
  .logoband { background: #fff; padding: 14px 24px; }
  .logoband .logo { height: 46px; width: auto; display: block; }
  .brandbar { height: 6px; background: var(--yellow); }
  .topbar { background: var(--core); color: #fff; padding: 24px 24px; }
  .topbar .wrap { max-width: 820px; margin: 0 auto; }
  a.back { color: #cfe0f5; text-decoration: none; font-size: 13px; }
  a.back:hover { text-decoration: underline; color: #fff; }
  h1 { margin: 10px 0 6px; font-size: 25px; font-weight: 800; }
  .tagline { color: #9db4d6; font-family: 'Lora', Georgia, serif; font-style: italic; font-size: 12px; margin-bottom: 10px; }
  h2 { font-size: 17px; margin: 26px 0 8px; color: var(--core); font-weight: 800; }
  .lede { color: #cfdaea; max-width: 72ch; }
  main { max-width: 820px; margin: 0 auto; padding: 4px 24px 60px; }
  .panel { background: var(--panel); border: 1px solid var(--border);
    border-radius: 12px; padding: 18px 22px; margin-top: 16px; box-shadow: 0 1px 2px rgba(16,36,66,.05); }
  p { max-width: 72ch; }
  .statline { font-size: 15px; margin: 4px 0; }
  .statline b { color: var(--core); }
  .statline.muted, .muted { color: var(--muted); font-size: 13.5px; }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
  .chip { background: var(--panel2); border: 1px solid var(--border); color: var(--gray);
    border-radius: 999px; padding: 3px 10px; font-size: 12.5px; }
  .method { border-left: 3px solid var(--green); padding-left: 14px; margin: 14px 0; }
  .method h3 { margin: 0 0 4px; font-size: 15px; color: var(--core); font-weight: 700; }
  code { background: var(--panel2); border-radius: 4px; padding: 1px 6px; font-size: 13px; }
  .note { color: var(--muted); font-size: 13px; }
  footer { max-width: 820px; margin: 0 auto; padding: 20px 24px 40px;
    color: var(--muted); font-size: 12px; border-top: 1px solid var(--border); }
</style>
</head>
<body>
<div class="logoband"><img class="logo" src="finra%20logo.png" alt="FINRA Enterprise Risk Management"></div>
<div class="brandbar"></div>
<div class="topbar">
  <div class="wrap">
    <a class="back" href="index.html">← Back to reports</a>
    <h1>Risk Discovery: finding risks that aren't on our list</h1>
    <div class="tagline">Investor protection. Market integrity.</div>
    <p class="lede">The main pipeline only ever sees news that matched one of our existing risk
    search terms — so by design it can't reveal a risk we never thought to search for. Risk
    discovery closes that blind spot: it collects a broad stream of news chosen <i>without</i>
    reference to our risk list, so we can later surface coherent, significant themes that sit far
    from every risk we currently track.</p>
  </div>
</div>
<main>

  <div class="panel">
    <h2>What we're collecting</h2>
    __CORPUS__
    <p class="note">Because the news plan serves only the last ~48 hours per request, the corpus
    accumulates forward over time — each scheduled run adds new, de-duplicated stories rather
    than reaching back into history.</p>
  </div>

  <h2>Two collection methods</h2>
  <p>Each scheduled run gathers news two complementary ways, both un-tied to our risk list:</p>

  <div class="panel">
    <div class="method">
      <h3>1 · Broad category sweep</h3>
      <p>Pulls top-tier news across broad categories with <b>no keyword filter at all</b> — a
      wide net for whatever is prominent. Category sets rotate across runs (the news API allows
      up to 5 per request), so coverage widens over time.</p>
      <div class="chips">__CAT_CHIPS__</div>
    </div>

    <div class="method">
      <h3>2 · Risk-lexicon search (__LEX_COUNT__ phrases)</h3>
      <p>Searches for the <b>language of risk itself</b> — phrases that tend to describe a
      risk becoming a concern, regardless of subject. This catches risk-shaped stories a plain
      category sweep would bury. Phrases are favored over bare words (<code>"regulators warn"</code>
      is far more precise than <code>"risk"</code>).</p>
      <div class="chips">__LEX_CHIPS__</div>
    </div>
  </div>

  <h2>How a discovered risk is identified</h2>
  <p>The collected stories are cleaned for quality, de-duplicated (including near-identical
  syndicated reprints), and clustered into themes. Each theme is then scored for
  <b>novelty</b> — how far it sits from <i>every</i> existing enterprise and emerging risk,
  measured by semantic similarity:</p>
  <div class="panel">
    <p style="margin:0"><code>novelty = 1 − (closest match to any existing risk)</code></p>
    <p class="note" style="margin:8px 0 0">A theme that is high-volume, recent, good quality, and
    far from all known risks is a candidate risk the taxonomy may not yet cover. The nearest
    existing risk is reported alongside each candidate, so it's clear whether something is
    genuinely new or just an adjacent angle on a known risk.</p>
  </div>

  <p class="note" style="margin-top:20px">Discovery is intentionally a wide, lower-precision net:
  it surfaces a ranked shortlist for human review, not a finished answer. Precision improves as
  the corpus accumulates across more days.</p>

</main>
<footer>Generated __GENERATED__ · self-contained HTML, no external dependencies.</footer>
</body>
</html>
"""


if __name__ == "__main__":
    main()
