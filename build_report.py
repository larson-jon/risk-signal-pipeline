# TOPIC REPORT BUILDER
#
# Generates a standalone, interactive HTML report from the topic-clustering
# output (the per-article and per-topic-summary CSVs written by
# topic_clustering.py). The report embeds its data as JSON and uses only
# vanilla JS/CSS, so the resulting file works by double-clicking - no server
# or internet connection required.
#
# Usage:
#   python build_report.py --risk-type enterprise
#   python build_report.py --risk-type emerging
#   python build_report.py --topics output/enterprise_risks_topics.csv \
#                          --summary output/enterprise_risks_topic_summary.csv \
#                          --out output/enterprise_report.html

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

RISK_TYPE_CONFIG = {
    "enterprise": {
        "topics": "output/enterprise_risks_topics.csv",
        "summary": "output/enterprise_risks_topic_summary.csv",
        "out": "reports/enterprise_risks_report.html",
        "title": "Enterprise Risk — Topic Explorer",
        "encoded": "data/EnterpriseRisksListEncoded.csv",
        "risk_id_col": "ENTERPRISE_RISK_ID",
        "trends": "output/enterprise_risks_trends.csv",
        "gaps": "output/enterprise_risks_gaps.csv",
    },
    "emerging": {
        "topics": "output/emerging_risks_topics.csv",
        "summary": "output/emerging_risks_topic_summary.csv",
        "out": "reports/emerging_risks_report.html",
        "title": "Emerging Risk — Topic Explorer",
        "encoded": "data/EmergingRisksListEncoded.csv",
        "risk_id_col": "EMERGING_RISK_ID",
        "trends": "output/emerging_risks_trends.csv",
        "gaps": "output/emerging_risks_gaps.csv",
    },
}


def _decode_term(term):
    """Decode a single encoded search term (mirrors news_sentiment_scraper)."""
    try:
        n = int(term)
        byte_len = (n.bit_length() + 7) // 8
        return n.to_bytes(byte_len, byteorder="little").decode("utf-8")
    except (ValueError, UnicodeDecodeError, OverflowError):
        return None


def build_risk_map(encoded_path, risk_id_col):
    """Map each risk ID to a readable label + its full list of search terms.

    The encoded CSVs don't carry a risk name, so we derive a description from
    the decoded search terms: the first term becomes the short label and the
    full set is kept for tooltips / detail.
    """
    risk_map = {}
    path = Path(encoded_path)
    if not path.exists():
        print(f"WARNING: encoded risk file not found ({path}); "
              "risk descriptions will fall back to IDs.")
        return risk_map

    df = pd.read_csv(path)
    if risk_id_col not in df or "ENCODED_TERMS" not in df:
        print(f"WARNING: {path} missing expected columns; using IDs only.")
        return risk_map

    df["_decoded"] = df["ENCODED_TERMS"].apply(_decode_term)
    for rid, group in df.groupby(risk_id_col):
        terms = [t.strip() for t in group["_decoded"].dropna().tolist() if t and t.strip()]
        if not terms:
            continue
        label = terms[0]
        # Trim quotes some terms are wrapped in.
        label = label.strip('"').strip()
        risk_map[str(int(rid))] = {"label": label, "terms": terms}
    print(f"Loaded descriptions for {len(risk_map)} risks from {path}.")
    return risk_map

# How many articles to embed per topic (keeps the file size reasonable for
# very large corpora; outliers are capped harder since they're less useful).
MAX_ARTICLES_PER_TOPIC = 60
MAX_OUTLIER_ARTICLES = 40


def _clean_num(value, default=0.0):
    try:
        f = float(value)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


def _clean_str(value):
    """String value from a CSV cell, treating NaN / 'nan' / None as empty.

    pandas reads an empty cell in an otherwise-numeric column (e.g. an optional
    risk ID) as float NaN, whose str() is 'nan'. This normalizes those to "".
    """
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    return "" if s.lower() == "nan" else s


def _article_records(topics_df, topic_id, limit):
    """Build the trimmed per-article payload for one topic."""
    group = topics_df[topics_df["TOPIC_ID"] == topic_id]
    # newest first when we have a parseable date
    if "PUBLISHED_DATE" in group:
        group = group.assign(
            _d=pd.to_datetime(group["PUBLISHED_DATE"], errors="coerce")
        ).sort_values("_d", ascending=False)
    records = []
    for _, r in group.head(limit).iterrows():
        records.append({
            "title": str(r.get("TITLE", "") or ""),
            "link": str(r.get("LINK", "") or ""),
            "date": str(r.get("PUBLISHED_DATE", "") or "")[:10],
            "source": str(r.get("SOURCE", "") or ""),
            "risk": str(r.get("RISK_ID", "") or ""),
            "sentiment": str(r.get("SENTIMENT", "") or ""),
            "compound": round(_clean_num(r.get("SENTIMENT_COMPOUND"), 0.0), 3),
            "entities": str(r.get("ENTITIES", "") or ""),
        })
    return records


def build_trends_payload(trends_df, trend_topics_df, by_risk_df=None):
    """Compact per-topic monthly intensity series + meta for the Trends view.

    When `by_risk_df` is provided, each risk's charts use intensity computed
    from ONLY that risk's articles, so a risk's lines reflect the topics most
    intense *for that risk specifically* rather than the topic's global series.
    """
    if trends_df is None or trends_df.empty:
        return None

    months = sorted(trends_df["month"].unique())
    # month -> topic -> intensity, so we can emit a dense series per topic.
    series = {}
    volume = {}
    for tid, g in trends_df.groupby("TOPIC_ID"):
        by_month = g.set_index("month")
        series[int(tid)] = [round(float(by_month["intensity"].get(m, 0.0)), 2) for m in months]
        volume[int(tid)] = [int(by_month["volume"].get(m, 0)) for m in months]

    labels, descriptions, benign_flags = {}, {}, {}
    topics = []
    for _, r in trend_topics_df.iterrows():
        tid = int(r["TOPIC_ID"])
        risk_ids = [x.strip() for x in str(r.get("RISK_IDS", "")).split(",") if x.strip()]
        # Prefer the descriptive title; fall back to the keyword label.
        label = _clean_str(r.get("TOPIC_TITLE")) or str(r.get("TOPIC_LABEL", "") or "")
        desc = str(r.get("TOPIC_DESCRIPTION", "") or "")
        benign = bool(r.get("IS_BENIGN", False))
        labels[tid], descriptions[tid], benign_flags[tid] = label, desc, benign
        topics.append({
            "id": tid,
            "label": label,
            "description": desc,
            "riskIds": risk_ids,
            "intensity": series.get(tid, []),
            "volume": volume.get(tid, []),
            "peakMonth": str(r.get("PEAK_MONTH", "") or ""),
            "peakIntensity": round(_clean_num(r.get("PEAK_INTENSITY"), 0.0), 2),
            "momentum": round(_clean_num(r.get("MOMENTUM"), 0.0), 2),
            "slope": round(_clean_num(r.get("TREND_SLOPE"), 0.0), 3),
            "benign": benign,
        })

    # ---- per-risk breakdown: risk -> [ {topic, intensity[], peak, momentum} ] ----
    by_risk = {}
    if by_risk_df is not None and not by_risk_df.empty:
        for rid, g in by_risk_df.groupby("RISK_ID"):
            rid = str(rid).replace(".0", "")
            entries = []
            for tid, tg in g.groupby("TOPIC_ID"):
                tid = int(tid)
                bm = tg.set_index("month")
                ser = [round(float(bm["intensity"].get(m, 0.0)), 2) for m in months]
                entries.append({
                    "id": tid,
                    "label": labels.get(tid, f"Topic {tid}"),
                    "description": descriptions.get(tid, ""),
                    "benign": benign_flags.get(tid, False),
                    "intensity": ser,
                    "peakIntensity": round(float(tg["peak_intensity"].iloc[0]), 2),
                    "momentum": round(float(tg["momentum"].iloc[0]), 2),
                })
            by_risk[rid] = entries

    return {"months": months, "topics": topics, "byRisk": by_risk}


def build_gaps_payload(gaps_df):
    """Compact ranked list of candidate taxonomy gaps for the Gaps view."""
    if gaps_df is None or gaps_df.empty:
        return None
    rows = []
    for _, r in gaps_df.iterrows():
        rows.append({
            "id": int(r["TOPIC_ID"]),
            "label": _clean_str(r.get("TOPIC_TITLE")) or str(r.get("TOPIC_LABEL", "") or ""),
            "description": str(r.get("TOPIC_DESCRIPTION", "") or ""),
            "riskIds": [x.strip() for x in str(r.get("RISK_IDS", "")).split(",") if x.strip()],
            "bestFitRisk": str(r.get("BEST_FIT_RISK", "") or ""),
            "bestFitRiskLabel": str(r.get("BEST_FIT_RISK_LABEL", "") or ""),
            "betterFitRisk": _clean_str(r.get("BETTER_FIT_RISK")).replace(".0", ""),
            "betterFitRiskLabel": _clean_str(r.get("BETTER_FIT_RISK_LABEL")),
            "fit": round(_clean_num(r.get("FIT"), 0.0), 3),
            "gap": round(_clean_num(r.get("GAP"), 0.0), 3),
            "gapRaw": round(_clean_num(r.get("GAP_RAW"), 0.0), 3),
            "fitPercentile": round(_clean_num(r.get("FIT_PERCENTILE"), 0.0), 3),
            "firstSeen": str(r.get("FIRST_SEEN", "") or ""),
            "isNew": bool(r.get("IS_NEW", False)),
            "peakVolume": int(_clean_num(r.get("PEAK_VOLUME"), 0)),
            "totalArticles": int(_clean_num(r.get("TOTAL_ARTICLES"), 0)),
            "quality": round(_clean_num(r.get("AVG_QUALITY"), 0.0), 3),
            "score": round(_clean_num(r.get("EMERGING_GAP_SCORE"), 0.0), 3),
        })
    return rows


def build_payload(topics_df, summary_df, risk_map=None, trends=None, gaps=None):
    """Assemble the JSON payload the HTML page renders from."""
    risk_map = risk_map or {}
    topics = []
    for _, s in summary_df.iterrows():
        tid = int(s["TOPIC_ID"])
        is_outlier = tid == -1
        limit = MAX_OUTLIER_ARTICLES if is_outlier else MAX_ARTICLES_PER_TOPIC
        risk_ids = [x.strip() for x in str(s.get("RISK_IDS", "")).split(",") if x.strip()]
        topics.append({
            "id": tid,
            "title": _clean_str(s.get("TOPIC_TITLE")) or str(s.get("TOPIC_LABEL", "") or ""),
            "label": str(s.get("TOPIC_LABEL", "") or ""),
            "description": str(s.get("TOPIC_DESCRIPTION", "") or ""),
            "count": int(_clean_num(s.get("ARTICLE_COUNT"), 0)),
            "distinctStories": int(_clean_num(s.get("DISTINCT_STORIES"), s.get("ARTICLE_COUNT", 0))),
            "riskIds": risk_ids,
            "riskCount": int(_clean_num(s.get("DISTINCT_RISK_COUNT"), 0)),
            "sentiment": str(s.get("DOMINANT_SENTIMENT", "") or ""),
            "avgCompound": round(_clean_num(s.get("AVG_SENTIMENT_COMPOUND"), 0.0), 3),
            "keywords": str(s.get("TOP_KEYWORDS", "") or ""),
            "entities": str(s.get("TOP_ENTITIES", "") or ""),
            "isOutlier": is_outlier,
            "articles": _article_records(topics_df, tid, limit),
        })

    total_articles = int(len(topics_df))
    outlier_articles = int((topics_df["TOPIC_ID"] == -1).sum())
    real_topics = sum(1 for t in topics if not t["isOutlier"])
    sentiment_counts = (
        topics_df["SENTIMENT"].fillna("Unknown").value_counts().to_dict()
        if "SENTIMENT" in topics_df else {}
    )

    return {
        "stats": {
            "totalArticles": total_articles,
            "topicCount": real_topics,
            "outlierArticles": outlier_articles,
            "sentiment": {k: int(v) for k, v in sentiment_counts.items()},
        },
        "risks": risk_map,
        "topics": topics,
        "trends": trends,
        "gaps": gaps,
    }


def render_html(title, payload):
    data_json = json.dumps(payload, ensure_ascii=False)
    # NOTE: the data is embedded verbatim; </script> can't appear in our CSV
    # values, but guard against it just in case.
    data_json = data_json.replace("</", "<\\/")
    return _HTML_TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data_json)


def run(topics_path, summary_path, out_path, title, encoded_path=None,
        risk_id_col=None, trends_path=None, gaps_path=None):
    topics_path, summary_path = Path(topics_path), Path(summary_path)
    for p in (topics_path, summary_path):
        if not p.exists():
            print(f"ERROR: missing input file: {p}")
            print("Run topic_clustering.py first to generate it.")
            sys.exit(1)

    topics_df = pd.read_csv(topics_path)
    summary_df = pd.read_csv(summary_path)
    print(f"Loaded {len(topics_df)} articles across {len(summary_df)} topics.")

    risk_map = {}
    if encoded_path and risk_id_col:
        risk_map = build_risk_map(encoded_path, risk_id_col)

    # Optional trends data (from topic_trends.py). We look for the paired
    # <trends>.csv and <trends>_topics.csv; the view is omitted if absent.
    trends_payload = None
    if trends_path:
        tp = Path(trends_path)
        meta_p = tp.with_name(tp.stem + "_topics.csv")
        if tp.exists() and meta_p.exists():
            trends_df = pd.read_csv(tp)
            trend_topics_df = pd.read_csv(meta_p)
            by_risk_p = tp.with_name(tp.stem + "_by_risk.csv")
            by_risk_df = pd.read_csv(by_risk_p) if by_risk_p.exists() else None
            trends_payload = build_trends_payload(trends_df, trend_topics_df, by_risk_df)
            extra = (f", per-risk breakdown from {by_risk_p.name}"
                     if by_risk_df is not None else " (no per-risk breakdown)")
            print(f"Loaded trends: {len(trend_topics_df)} topics over "
                  f"{trends_df['month'].nunique()} months from {tp}{extra}.")
        else:
            print(f"NOTE: trends files not found ({tp} / {meta_p}); "
                  "Trends view will be omitted. Run topic_trends.py to enable it.")

    # Optional gaps data (from topic_gaps.py).
    gaps_payload = None
    if gaps_path:
        gp = Path(gaps_path)
        if gp.exists():
            gaps_payload = build_gaps_payload(pd.read_csv(gp))
            print(f"Loaded gaps: {len(gaps_payload)} candidate topics from {gp}.")
        else:
            print(f"NOTE: gaps file not found ({gp}); Gaps view will be omitted. "
                  "Run topic_gaps.py to enable it.")

    payload = build_payload(topics_df, summary_df, risk_map, trends_payload, gaps_payload)
    html = render_html(title, payload)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote report -> {out_path}")
    print(f"Open it in a browser: {out_path.resolve()}")


def parse_args():
    p = argparse.ArgumentParser(description="Build an interactive HTML topic report.")
    p.add_argument("--risk-type", choices=["enterprise", "emerging"])
    p.add_argument("--topics", help="Per-article topics CSV.")
    p.add_argument("--summary", help="Per-topic summary CSV.")
    p.add_argument("--out", help="Output HTML path.")
    p.add_argument("--title", help="Report title.")
    p.add_argument("--encoded", help="Encoded risk-terms CSV (for risk descriptions).")
    p.add_argument("--risk-id-col", dest="risk_id_col",
                   help="Risk ID column name in the encoded CSV.")
    p.add_argument("--trends", help="Trends CSV from topic_trends.py (enables Trends view).")
    p.add_argument("--gaps", help="Gaps CSV from topic_gaps.py (enables Candidate gaps view).")
    return p.parse_args()


def main():
    args = parse_args()
    encoded = risk_id_col = trends = gaps = None
    if args.risk_type:
        cfg = RISK_TYPE_CONFIG[args.risk_type]
        topics = args.topics or cfg["topics"]
        summary = args.summary or cfg["summary"]
        out = args.out or cfg["out"]
        title = args.title or cfg["title"]
        encoded = args.encoded or cfg["encoded"]
        risk_id_col = cfg["risk_id_col"]
        trends = args.trends or cfg["trends"]
        gaps = args.gaps or cfg["gaps"]
    elif args.topics and args.summary:
        topics = args.topics
        summary = args.summary
        out = args.out or str(Path(topics).with_name(Path(topics).stem + "_report.html"))
        title = args.title or "Risk Topic Explorer"
        encoded = args.encoded
        risk_id_col = args.risk_id_col
        trends = args.trends
        gaps = args.gaps
    else:
        print("ERROR: provide --risk-type, or both --topics and --summary.")
        sys.exit(1)

    run(topics, summary, out, title, encoded, risk_id_col, trends, gaps)


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --bg: #0f1720; --panel: #172232; --panel2: #1e2c40; --text: #e6edf5;
    --muted: #93a4bb; --border: #2a3a52;
    --pos: #3fb950; --neg: #f85149; --neu: #8b98a9; --accent: #58a6ff;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
    font: 14px/1.5 -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
  header { padding: 20px 24px; border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, #16212f, #0f1720); position: sticky; top: 0; z-index: 5; }
  h1 { margin: 4px 0 4px; font-size: 20px; }
  a.back { color: var(--accent); text-decoration: none; font-size: 13px; }
  a.back:hover { text-decoration: underline; }
  .sub { color: var(--muted); font-size: 13px; }
  .stats { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 14px; }
  .stat { background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 14px; min-width: 110px; }
  .stat .n { font-size: 20px; font-weight: 700; }
  .stat .l { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
  .controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
    padding: 14px 24px; border-bottom: 1px solid var(--border); background: var(--panel); position: sticky; top: 0; }
  input[type=search], select { background: var(--panel2); color: var(--text);
    border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px; font-size: 13px; }
  input[type=search] { min-width: 260px; flex: 1; }
  label.toggle { color: var(--muted); font-size: 13px; display: flex; align-items: center; gap: 6px; cursor: pointer; }
  main { padding: 18px 24px 60px; }
  .topic { background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; margin-bottom: 12px; overflow: hidden; }
  .topic-head { display: grid; grid-template-columns: 1fr auto; gap: 12px;
    padding: 14px 16px; cursor: pointer; align-items: start; }
  .topic-head:hover { background: var(--panel2); }
  .topic-title { font-weight: 650; font-size: 15px; }
  .topic-keywords { color: var(--muted); font-size: 11.5px; margin-top: 2px; font-family: ui-monospace, monospace; }
  .topic-desc { color: var(--text); font-size: 13px; margin-top: 5px; max-width: 70ch; }
  .topic-meta { color: var(--muted); font-size: 12.5px; margin-top: 6px; }
  .chips { margin-top: 8px; display: flex; flex-wrap: wrap; gap: 6px; }
  .chip { background: var(--panel2); border: 1px solid var(--border); color: var(--muted);
    border-radius: 999px; padding: 2px 9px; font-size: 11.5px; }
  .chip.risk { color: var(--accent); border-color: #274b73; }
  .chip.risk b { color: var(--text); font-weight: 700; margin-right: 2px; }
  .chip.risk.active { background: #274b73; color: #fff; border-color: var(--accent); }
  .chip.risk.active b { color: #fff; }
  .badges { display: flex; flex-direction: column; align-items: flex-end; gap: 6px; white-space: nowrap; }
  .count { font-size: 22px; font-weight: 700; }
  .pill { border-radius: 999px; padding: 2px 10px; font-size: 12px; font-weight: 600; }
  .pill.Positive { background: rgba(63,185,80,.15); color: var(--pos); }
  .pill.Negative { background: rgba(248,81,73,.15); color: var(--neg); }
  .pill.Neutral  { background: rgba(139,152,169,.15); color: var(--neu); }
  .articles { display: none; border-top: 1px solid var(--border); }
  .topic.open .articles { display: block; }
  .art { display: grid; grid-template-columns: 1fr auto; gap: 10px;
    padding: 10px 16px; border-top: 1px solid var(--border); }
  .art:first-child { border-top: none; }
  .art a { color: var(--accent); text-decoration: none; font-weight: 550; }
  .art a:hover { text-decoration: underline; }
  .art .meta { color: var(--muted); font-size: 12px; margin-top: 3px; }
  .art .ent { color: var(--muted); font-size: 12px; margin-top: 3px; font-style: italic; }
  .art .right { text-align: right; white-space: nowrap; color: var(--muted); font-size: 12px; }
  .empty { color: var(--muted); padding: 40px; text-align: center; }
  .foot { color: var(--muted); font-size: 12px; padding: 16px 24px; border-top: 1px solid var(--border); }
  mark { background: #3b5074; color: #fff; border-radius: 2px; padding: 0 1px; }
  /* tabs */
  .tabs { display: flex; gap: 4px; margin-top: 14px; }
  .tab { background: transparent; color: var(--muted); border: 1px solid var(--border);
    border-bottom: none; border-radius: 8px 8px 0 0; padding: 8px 16px; cursor: pointer; font-size: 13px; font-weight: 600; }
  .tab.active { background: var(--panel); color: var(--text); }
  .view { display: none; }
  .view.active { display: block; }
  /* trends */
  .rising { background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 14px 16px; margin-bottom: 18px; }
  .rising h3 { margin: 0 0 10px; font-size: 14px; }
  .rising-row { display: flex; align-items: center; gap: 10px; padding: 5px 0; font-size: 13px; }
  .rising-row .arrow { color: var(--neg); font-weight: 700; min-width: 54px; }
  .rising-row .arrow.down { color: var(--pos); }
  .risk-block { background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; margin-bottom: 16px; padding: 14px 16px; }
  .risk-block h2 { font-size: 15px; margin: 0 0 2px; }
  .risk-block .rterms { color: var(--muted); font-size: 12px; margin-bottom: 10px; }
  .chart-wrap { display: grid; grid-template-columns: 1fr; gap: 8px; }
  .legend { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px; font-size: 12px; }
  .legend .item { display: flex; align-items: center; gap: 5px; color: var(--muted); cursor: default; }
  .legend .swatch { width: 12px; height: 3px; border-radius: 2px; display: inline-block; }
  svg .grid { stroke: var(--border); stroke-width: 1; }
  svg .axis { fill: var(--muted); font-size: 10px; }
  svg .line { fill: none; stroke-width: 2; }
  svg .dot { stroke: var(--bg); stroke-width: 1; }
  svg .peak { fill: var(--neg); }
  /* gap bar */
  .gapbar { position: relative; height: 16px; background: var(--panel2);
    border: 1px solid var(--border); border-radius: 4px; margin-top: 8px; max-width: 320px; }
  .gapbar-fill { height: 100%; background: linear-gradient(90deg, #d29922, #f85149); border-radius: 3px; }
  .gapbar-label { position: absolute; left: 8px; top: 0; font-size: 11px; line-height: 16px; color: var(--text); }
  /* methodology */
  .method { max-width: 80ch; background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 20px 24px; }
  .method h3 { font-size: 15px; margin: 22px 0 6px; color: var(--accent); }
  .method p { margin: 6px 0; }
  .method pre { background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
    padding: 10px 12px; font-size: 12.5px; overflow-x: auto; color: var(--text); }
  .method code { background: var(--panel2); border-radius: 4px; padding: 1px 5px; font-size: 12.5px; }
</style>
</head>
<body>
<header>
  <a class="back" href="index.html">← Back to reports</a>
  <h1>__TITLE__</h1>
  <div class="sub">Topics discovered with sentence-transformers + BERTopic, enriched with spaCy. Click a topic to see its articles.</div>
  <div class="stats" id="stats"></div>
  <div class="tabs">
    <div class="tab active" id="tab-topics" onclick="switchView('topics')">Topics</div>
    <div class="tab" id="tab-trends" onclick="switchView('trends')">Trends over time</div>
    <div class="tab" id="tab-gaps" onclick="switchView('gaps')">Candidate gaps</div>
    <div class="tab" id="tab-risks" onclick="switchView('risks')">Risks &amp; search terms</div>
    <div class="tab" id="tab-method" onclick="switchView('method')">How this was built</div>
  </div>
</header>

<div class="controls" id="topics-controls">
  <input type="search" id="q" placeholder="Search topics, keywords, entities, risks...">
  <select id="risk"><option value="">All risks</option></select>
  <select id="sentiment">
    <option value="">All sentiment</option>
    <option value="Positive">Positive</option>
    <option value="Negative">Negative</option>
    <option value="Neutral">Neutral</option>
  </select>
  <select id="sort">
    <option value="count">Sort: most articles</option>
    <option value="risk">Sort: most risks spanned</option>
    <option value="pos">Sort: most positive</option>
    <option value="neg">Sort: most negative</option>
  </select>
  <label class="toggle"><input type="checkbox" id="hideOut" checked> Hide outliers</label>
</div>

<div class="controls" id="trends-controls" style="display:none">
  <select id="trisk"><option value="">All risks</option></select>
  <select id="tsort">
    <option value="momentum">Sort risks: strongest rising signal</option>
    <option value="peak">Sort risks: highest peak intensity</option>
    <option value="id">Sort risks: by ID</option>
  </select>
  <label class="toggle"><input type="checkbox" id="hideBenign" checked> Hide benign / low-signal topics</label>
  <label class="toggle"><input type="number" id="topN" value="6" min="1" max="15" style="width:52px"> top topics per risk</label>
</div>

<main>
  <div class="view active" id="view-topics">
    <div id="list"></div>
    <div class="foot" id="foot"></div>
  </div>
  <div class="view" id="view-trends">
    <div id="trends-body"></div>
  </div>
  <div class="view" id="view-gaps">
    <div class="sub" style="margin-bottom:14px">
      Topics ranked by how <b>poorly they fit the risk that surfaced them</b> (gap), weighted by
      how intense and recent they are. High-gap, high-volume, recent topics are candidate
      aspects of risk the current taxonomy may not fully capture. Boilerplate is filtered by
      content quality.
    </div>
    <div id="gaps-body"></div>
  </div>
  <div class="view" id="view-risks">
    <div class="sub" style="margin-bottom:14px">
      The exact search terms used to fetch news for each risk. Articles are matched to a risk
      when one of its terms surfaces them, so these terms define the scope of each risk's
      coverage.
    </div>
    <div id="risks-body"></div>
  </div>
  <div class="view" id="view-method">
    <div id="method-body"></div>
  </div>
</main>

<script>
const DATA = __DATA__;

const el = (id) => document.getElementById(id);
const esc = (s) => (s || "").replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const RISKS = DATA.risks || {};

// Readable name for a risk id: its description label, falling back to the id.
function riskName(id) {
  const r = RISKS[String(id)];
  return r && r.label ? r.label : ("Risk " + id);
}
// Full search-term list for a risk, used as a hover tooltip.
function riskTerms(id) {
  const r = RISKS[String(id)];
  return r && r.terms ? r.terms.join(" · ") : "";
}

function populateRiskFilter() {
  const sel = el("risk");
  // Only offer risks that actually appear in the data, sorted numerically.
  const present = new Set();
  DATA.topics.forEach(t => t.riskIds.forEach(r => present.add(String(r))));
  const ids = [...present].sort((a, b) => Number(a) - Number(b));
  ids.forEach(id => {
    const o = document.createElement("option");
    o.value = id;
    o.textContent = `Risk ${id} — ${riskName(id)}`;
    o.title = riskTerms(id);
    sel.appendChild(o);
  });
}

function renderStats() {
  const s = DATA.stats;
  const sent = s.sentiment || {};
  const cards = [
    ["Articles", s.totalArticles],
    ["Topics", s.topicCount],
    ["Outliers", s.outlierArticles],
    ["Positive", sent.Positive || 0],
    ["Negative", sent.Negative || 0],
    ["Neutral", sent.Neutral || 0],
  ];
  el("stats").innerHTML = cards.map(([l, n]) =>
    `<div class="stat"><div class="n">${n.toLocaleString()}</div><div class="l">${l}</div></div>`).join("");
}

function highlight(text, q) {
  if (!q) return esc(text);
  const safe = esc(text);
  try {
    const re = new RegExp("(" + q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "ig");
    return safe.replace(re, "<mark>$1</mark>");
  } catch (e) { return safe; }
}

function topicMatches(t, q) {
  if (!q) return true;
  const riskText = t.riskIds.map(r => riskName(r) + " " + riskTerms(r)).join(" ");
  const hay = [t.label, t.description, t.keywords, t.entities, t.riskIds.join(" "),
    riskText, "topic " + t.id].join(" ").toLowerCase();
  return hay.includes(q.toLowerCase());
}

function render() {
  const q = el("q").value.trim();
  const riskFilter = el("risk").value;
  const sentFilter = el("sentiment").value;
  const sortBy = el("sort").value;
  const hideOut = el("hideOut").checked;

  let topics = DATA.topics.filter(t => {
    if (hideOut && t.isOutlier) return false;
    if (sentFilter && t.sentiment !== sentFilter) return false;
    if (riskFilter && !t.riskIds.map(String).includes(riskFilter)) return false;
    return topicMatches(t, q);
  });

  topics.sort((a, b) => {
    if (sortBy === "count") return b.count - a.count;
    if (sortBy === "risk") return b.riskCount - a.riskCount || b.count - a.count;
    if (sortBy === "pos") return b.avgCompound - a.avgCompound;
    if (sortBy === "neg") return a.avgCompound - b.avgCompound;
    return 0;
  });

  const list = el("list");
  if (!topics.length) {
    list.innerHTML = '<div class="empty">No topics match your filters.</div>';
    el("foot").textContent = "";
    return;
  }

  list.innerHTML = topics.map(t => {
    const risks = t.riskIds.map(r => {
      const active = riskFilter && String(r) === riskFilter ? " active" : "";
      return `<span class="chip risk${active}" title="${esc(riskTerms(r))}">` +
        `<b>R${esc(r)}</b> ${highlight(riskName(r), q)}</span>`;
    }).join("");
    const kw = (t.keywords || "").split(",").map(k => k.trim()).filter(Boolean)
      .slice(0, 8).map(k => `<span class="chip">${highlight(k, q)}</span>`).join("");
    const arts = t.articles.map(a => `
      <div class="art">
        <div>
          <a href="${esc(a.link)}" target="_blank" rel="noopener">${highlight(a.title, q)}</a>
          <div class="meta">${esc(a.source)} · ${esc(a.date)} · <span title="${esc(riskTerms(a.risk))}">R${esc(a.risk)} ${esc(riskName(a.risk))}</span></div>
          ${a.entities ? `<div class="ent">${highlight(a.entities, q)}</div>` : ""}
        </div>
        <div class="right"><span class="pill ${esc(a.sentiment)}">${esc(a.sentiment)}</span><br>${a.compound}</div>
      </div>`).join("");

    return `
      <div class="topic" data-id="${t.id}">
        <div class="topic-head" onclick="this.parentNode.classList.toggle('open')">
          <div>
            <div class="topic-title">${t.isOutlier ? "Outliers / Unclustered" : highlight(t.title, q)}</div>
            ${t.isOutlier ? "" : `<div class="topic-keywords">${highlight(t.label, q)}</div>`}
            ${t.description ? `<div class="topic-desc">${highlight(t.description, q)}</div>` : ""}
            <div class="topic-meta">Spans ${t.riskCount} risk${t.riskCount === 1 ? "" : "s"} ·
              avg sentiment ${t.avgCompound} ·
              ${t.distinctStories && t.distinctStories < t.count
                ? `${t.distinctStories.toLocaleString()} distinct stories (${t.count.toLocaleString()} articles)`
                : `${t.count.toLocaleString()} articles`} ·
              showing ${t.articles.length}</div>
            <div class="chips">${risks}${kw}</div>
          </div>
          <div class="badges">
            <div class="count">${t.count.toLocaleString()}</div>
            <span class="pill ${esc(t.sentiment)}">${esc(t.sentiment)}</span>
          </div>
        </div>
        <div class="articles">${arts || '<div class="empty">No article details stored for this topic.</div>'}</div>
      </div>`;
  }).join("");

  el("foot").textContent = `Showing ${topics.length} topic(s).`;
}

// ---------------------------------------------------------------------------
// Trends view
// ---------------------------------------------------------------------------
const TRENDS = DATA.trends;
const LINE_COLORS = ["#58a6ff","#f85149","#3fb950","#d29922","#bc8cff",
  "#39c5cf","#ff7b72","#a5d6ff","#e3b341","#7ee787"];

function switchView(name) {
  ["topics", "trends", "gaps", "risks", "method"].forEach(v => {
    const on = v === name;
    const tab = el("tab-" + v), view = el("view-" + v);
    if (tab) tab.classList.toggle("active", on);
    if (view) view.classList.toggle("active", on);
  });
  el("topics-controls").style.display = name === "topics" ? "" : "none";
  el("trends-controls").style.display = name === "trends" ? "" : "none";
}

// Build an inline SVG multi-line chart. `lines` = [{name,color,values}], and
// `months` are the shared x labels.
function lineChart(months, lines) {
  const W = 720, H = 220, padL = 40, padR = 12, padT = 12, padB = 26;
  const innerW = W - padL - padR, innerH = H - padT - padB;
  const maxY = Math.max(1, ...lines.flatMap(l => l.values));
  const xAt = i => padL + (months.length <= 1 ? innerW / 2 : (i / (months.length - 1)) * innerW);
  const yAt = v => padT + innerH - (v / maxY) * innerH;

  // gridlines + y labels (0, mid, max)
  let grid = "";
  [0, 0.5, 1].forEach(f => {
    const y = padT + innerH - f * innerH;
    grid += `<line class="grid" x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}"/>`;
    grid += `<text class="axis" x="4" y="${y + 3}">${Math.round(f * maxY)}</text>`;
  });
  // x labels (abbreviated month) - show every other if crowded
  let xlab = "";
  months.forEach((m, i) => {
    if (months.length > 7 && i % 2 === 1) return;
    xlab += `<text class="axis" text-anchor="middle" x="${xAt(i)}" y="${H - 8}">${m.slice(2)}</text>`;
  });
  // lines + dots (peak dot emphasized)
  let paths = "";
  lines.forEach(l => {
    const d = l.values.map((v, i) => `${i ? "L" : "M"}${xAt(i).toFixed(1)},${yAt(v).toFixed(1)}`).join(" ");
    paths += `<path class="line" stroke="${l.color}" d="${d}"/>`;
    const peakIdx = l.values.indexOf(Math.max(...l.values));
    l.values.forEach((v, i) => {
      const isPeak = i === peakIdx && v > 0;
      paths += `<circle class="dot" cx="${xAt(i).toFixed(1)}" cy="${yAt(v).toFixed(1)}" ` +
        `r="${isPeak ? 4 : 2.5}" fill="${l.color}"><title>${l.name}\n${months[i]}: ${v}</title></circle>`;
    });
  });
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" preserveAspectRatio="xMidYMid meet">${grid}${xlab}${paths}</svg>`;
}

function populateTrendRiskFilter() {
  if (!TRENDS) return;
  const sel = el("trisk");
  const present = new Set(Object.keys(TRENDS.byRisk || {}));
  if (!present.size) TRENDS.topics.forEach(t => t.riskIds.forEach(r => present.add(String(r))));
  [...present].sort((a, b) => Number(a) - Number(b)).forEach(id => {
    const o = document.createElement("option");
    o.value = id; o.textContent = `Risk ${id} — ${riskName(id)}`; o.title = riskTerms(id);
    sel.appendChild(o);
  });
}

function renderTrends() {
  const body = el("trends-body");
  if (!TRENDS) {
    body.innerHTML = '<div class="empty">No trends data embedded. Run topic_trends.py and rebuild with --trends.</div>';
    return;
  }
  const months = TRENDS.months;
  const hideBenign = el("hideBenign").checked;
  const riskFilter = el("trisk").value;
  const topN = Math.max(1, Math.min(15, parseInt(el("topN").value) || 6));
  const sortBy = el("tsort").value;

  let topics = TRENDS.topics.filter(t => !(hideBenign && t.benign));

  // Rising-signals callout (global, respects benign toggle).
  const rising = [...topics].sort((a, b) => b.momentum - a.momentum).slice(0, 8);
  let risingHtml = "";
  if (rising.length) {
    risingHtml = `<div class="rising"><h3>Strongest rising signals (recent momentum)</h3>` +
      rising.map(t => {
        const up = t.momentum >= 0;
        const risks = t.riskIds.map(r => `R${esc(r)}`).join(", ");
        return `<div class="rising-row"><span class="arrow ${up ? "" : "down"}">` +
          `${up ? "▲" : "▼"} ${t.momentum.toFixed(0)}</span>` +
          `<span><b>${esc(t.label)}</b> — peak ${esc(t.peakMonth)}, risks ${risks}</span></div>`;
      }).join("") + `</div>`;
  }

  // Per-risk charts use the risk-specific breakdown: each risk's lines are the
  // topics most intense for THAT risk, scored from only that risk's articles.
  const byRisk = TRENDS.byRisk || {};
  const benignById = {};
  TRENDS.topics.forEach(t => { benignById[t.id] = t.benign; });

  let riskList = Object.keys(byRisk);
  if (riskFilter) riskList = riskList.filter(r => r === riskFilter);

  // Rank each risk's topics by that risk's own peak intensity, then pick top N.
  const riskTopics = {};
  riskList.forEach(r => {
    let entries = byRisk[r].filter(e => !(hideBenign && e.benign));
    entries = entries.sort((a, b) => b.peakIntensity - a.peakIntensity).slice(0, topN);
    riskTopics[r] = entries;
  });

  const riskStat = {};
  riskList.forEach(r => {
    const mine = riskTopics[r];
    riskStat[r] = {
      momentum: Math.max(0, ...mine.map(t => t.momentum), 0),
      peak: Math.max(0, ...mine.map(t => t.peakIntensity), 0),
    };
  });
  riskList.sort((a, b) => {
    if (sortBy === "id") return Number(a) - Number(b);
    if (sortBy === "peak") return riskStat[b].peak - riskStat[a].peak;
    return riskStat[b].momentum - riskStat[a].momentum;
  });

  const blocks = riskList.map(r => {
    const mine = riskTopics[r];
    if (!mine.length) return "";
    const lines = mine.map((t, i) => ({
      name: t.label, color: LINE_COLORS[i % LINE_COLORS.length], values: t.intensity,
    }));
    const legend = mine.map((t, i) => {
      const c = LINE_COLORS[i % LINE_COLORS.length];
      const mom = t.momentum >= 0 ? `▲${t.momentum.toFixed(0)}` : `▼${Math.abs(t.momentum).toFixed(0)}`;
      return `<span class="item" title="${esc(t.description)}"><span class="swatch" style="background:${c}"></span>` +
        `${esc(t.label)} <span style="color:var(--muted)">(peak ${t.peakIntensity.toFixed(0)}, ${mom})</span></span>`;
    }).join("");
    return `<div class="risk-block">
      <h2>Risk ${esc(r)} — ${esc(riskName(r))}</h2>
      <div class="rterms">${esc(riskTerms(r))}</div>
      <div class="chart-wrap">${lineChart(months, lines)}</div>
      <div class="legend">${legend}</div>
    </div>`;
  }).join("");

  body.innerHTML = risingHtml + (blocks || '<div class="empty">No topics for this selection.</div>');
}

// ---------------------------------------------------------------------------
// Candidate gaps view
// ---------------------------------------------------------------------------
const GAPS = DATA.gaps;

function renderGaps() {
  const body = el("gaps-body");
  if (!GAPS || !GAPS.length) {
    body.innerHTML = '<div class="empty">No gaps data embedded. Run topic_gaps.py and rebuild with --gaps.</div>';
    return;
  }
  const rows = GAPS.slice(0, 40).map((t, i) => {
    const risks = t.riskIds.map(r => `<span class="chip risk"><b>R${esc(r)}</b> ${esc(riskName(r))}</span>`).join("");
    const newBadge = t.isNew ? `<span class="pill Negative" title="First seen ${esc(t.firstSeen)}">NEW</span>` : "";
    // calibrated gap bar (0..1) = 1 - percentile fit vs all risks
    const pct = Math.round(t.gap * 100);
    // Callout when a *different* risk fits this topic better than the ones that surfaced it.
    const better = t.betterFitRisk
      ? `<div class="topic-meta" style="color:var(--neg)">Fits <b>R${esc(t.betterFitRisk)} ${esc(t.betterFitRiskLabel)}</b> better than the risk(s) that surfaced it — possible taxonomy gap.</div>`
      : "";
    return `<div class="risk-block">
      <div style="display:flex; justify-content:space-between; gap:12px; align-items:start">
        <div>
          <div class="topic-title">#${i + 1}. ${esc(t.label)} ${newBadge}</div>
          ${t.description ? `<div class="topic-desc">${esc(t.description)}</div>` : ""}
          ${better}
          <div class="topic-meta">
            Surfaced by ${risks} · best assigned-risk fit: <b>R${esc(t.bestFitRisk)} ${esc(t.bestFitRiskLabel)}</b>
            · peak ${t.peakVolume} distinct stories/mo · ${t.totalArticles} articles ·
            quality ${t.quality.toFixed(2)} · first seen ${esc(t.firstSeen)}
          </div>
          <div class="gapbar"><div class="gapbar-fill" style="width:${pct}%"></div>
            <span class="gapbar-label">gap ${t.gap.toFixed(2)} (raw ${t.gapRaw.toFixed(2)})</span></div>
        </div>
        <div class="badges"><div class="count">${t.score.toFixed(1)}</div>
          <span class="l" style="color:var(--muted); font-size:11px">GAP SCORE</span></div>
      </div>
    </div>`;
  }).join("");
  body.innerHTML = rows;
}

// ---------------------------------------------------------------------------
// Risks & search terms view
// ---------------------------------------------------------------------------
function renderRisks() {
  const body = el("risks-body");
  const ids = Object.keys(RISKS);
  if (!ids.length) {
    body.innerHTML = '<div class="empty">No risk search terms were embedded in this report.</div>';
    return;
  }
  // How many topics touch each risk (from the Topics data).
  const topicCount = {};
  DATA.topics.forEach(t => {
    if (t.isOutlier) return;
    t.riskIds.forEach(r => { topicCount[String(r)] = (topicCount[String(r)] || 0) + 1; });
  });

  const rows = ids.sort((a, b) => Number(a) - Number(b)).map(id => {
    const r = RISKS[id];
    const terms = (r.terms || []).map(t => `<span class="chip">${esc(t)}</span>`).join("");
    const n = topicCount[id] || 0;
    return `<div class="risk-block">
      <h2>Risk ${esc(id)} — ${esc(r.label || ("Risk " + id))}</h2>
      <div class="rterms">${(r.terms || []).length} search term(s) ·
        appears in ${n} topic${n === 1 ? "" : "s"}</div>
      <div class="chips">${terms}</div>
    </div>`;
  }).join("");
  body.innerHTML = rows;
}

// ---------------------------------------------------------------------------
// Methodology view
// ---------------------------------------------------------------------------
function renderMethod() {
  const s = DATA.stats;
  const months = (TRENDS && TRENDS.months) ? TRENDS.months : [];
  const span = months.length ? `${months[0]} to ${months[months.length - 1]} (${months.length} months)` : "the collected period";
  const nRisks = Object.keys(RISKS).length;
  const gapsNote = GAPS && GAPS.length
    ? `${GAPS.length} topics survived the content-quality filter and were scored.`
    : "The taxonomy-gap stage was not included in this build.";
  const trendsNote = TRENDS
    ? "included" : "not included in this build";

  el("method-body").innerHTML = `
  <div class="method">
    <p class="sub">This report is produced by an automated pipeline that turns news coverage into
    risk signals. It covers <b>${s.totalArticles.toLocaleString()} articles</b> across
    <b>${span}</b>, matched to <b>${nRisks} risks</b> and grouped into
    <b>${s.topicCount.toLocaleString()} topics</b> (${s.outlierArticles.toLocaleString()} articles
    were left unclustered as outliers). Every stage runs locally and offline.</p>

    <h3>1 · Fetch</h3>
    <p>For each risk, a set of search terms is queried against the newsdata.io API (trailing
    7-day window per scheduled run, accumulated over time). Articles are downloaded and parsed
    with <code>newspaper3k</code>. Paywalled / unparseable articles are kept but flagged.</p>

    <h3>2 · Sentiment</h3>
    <p>Each article's title + summary is scored with VADER, giving a compound score
    (−1..+1) and a Positive / Negative / Neutral label.</p>

    <h3>3 · spaCy enrichment</h3>
    <p>Article text is run through spaCy (<code>en_core_web_sm</code>) to extract named entities
    (organizations, people, places, groups, laws) and cleaned noun-phrase keywords. These feed
    the topic labels and let you search by entity.</p>

    <h3>4 · Embedding, dedup &amp; clustering</h3>
    <p>Title + summary is embedded with the <code>all-MiniLM-L6-v2</code> sentence-transformer,
    so semantically similar stories sit close together. Embeddings are cached on disk, so
    re-runs only embed new articles. <b>Near-duplicate detection</b> then collapses syndicated
    reprints (cosine similarity ≥ 0.93): one representative per story goes into clustering, and
    the assigned topic propagates back to every copy — so a wire story republished across 20
    outlets counts as one story, not twenty. <b>BERTopic</b> clusters the representatives using
    c-TF-IDF for keywords; the topic count can be reduced to merge the long tail into fewer,
    cleaner themes. Topic cards show <b>distinct stories</b> alongside raw article counts.</p>

    <h3>5 · Content quality</h3>
    <p>Each article is scored 0–1 on how substantive it is. Press-release / wire sources,
    templated digests, earnings-call transcripts, market-report spam, and thin or empty text are
    penalized. This keeps boilerplate from dominating the trend and gap analysis.</p>

    <h3>6 · Trends over time <span class="sub">(${trendsNote})</span></h3>
    <p>Topics from a single full-history clustering run (so IDs stay comparable) are binned by
    calendar month. Each topic-month gets a composite <b>signal intensity</b>:</p>
    <pre>severity  = 1 + neg_share + max(0, −avg_compound)
intensity = monthly_volume × severity</pre>
    <p>So a negative surge scores higher than a neutral one of equal volume. Per topic we also
    compute a trend slope, recent <b>momentum</b> (last 3 months vs the prior 3), and peak month.
    Each risk's chart shows the topics most intense <i>for that risk specifically</i>, scored
    from only that risk's articles.</p>

    <h3>7 · Candidate gaps</h3>
    <p>The goal is to surface aspects of risk the current taxonomy may not capture. For each
    topic we embed its centroid and every risk's search terms, then compare how well the topic
    fits its <i>assigned</i> risk against how well it fits <i>all</i> risks:</p>
    <pre>fit_percentile = fraction of risks the assigned-risk fit beats
gap            = 1 − fit_percentile     (calibrated)</pre>
    <p>Because a topic centroid and a short search phrase embed at systematically lower
    similarity, the raw cosine gap is inflated for every topic. Calibrating against the
    all-risk distribution makes the gap <b>relative</b>: a high gap means several other risks
    match the topic better than the one that surfaced it — a stronger blind-spot signal. When a
    different risk fits best, it's called out explicitly. Scores combine the calibrated gap,
    intensity (distinct-story volume), and a recency boost for newly-appearing topics. Both the
    calibrated and raw gap are shown. ${gapsNote}</p>

    <h3>Reading the results</h3>
    <p>Use <b>Topics</b> to browse and search all clusters, <b>Trends over time</b> to see which
    themes are growing per risk, and <b>Candidate gaps</b> for the ranked list of possibly
    uncovered risk aspects. Known limitations: topic labels are auto-generated and occasionally
    misleading; near-duplicate detection collapses most syndication but very lightly reworded
    reprints may still slip through.</p>

    <p class="sub">Tools: sentence-transformers (sbert.net), BERTopic
    (maartengr.github.io/BERTopic), spaCy (spacy.io), VADER, newspaper3k.</p>
  </div>`;
}

// Hide tabs with no data.
if (!TRENDS) el("tab-trends").style.display = "none";
if (!GAPS || !GAPS.length) el("tab-gaps").style.display = "none";

renderStats();
populateRiskFilter();
populateTrendRiskFilter();
render();
renderTrends();
renderGaps();
renderRisks();
renderMethod();
["q", "risk", "sentiment", "sort", "hideOut"].forEach(id =>
  el(id).addEventListener(el(id).type === "checkbox" ? "change" :
    (el(id).tagName === "SELECT" ? "change" : "input"), render));
["trisk", "tsort", "hideBenign", "topN"].forEach(id =>
  el(id).addEventListener(el(id).type === "checkbox" ? "change" :
    (el(id).tagName === "SELECT" ? "change" : "input"), renderTrends));
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
