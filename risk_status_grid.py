# RISK STATUS GRID  (prototype for the ERM group)
#
# Turns the per-risk monthly trend data into a single decision surface: one tile
# per enterprise risk, each classified into an at-a-glance status ERM can act on
#   Escalating - rising fast and/or above its own normal band -> look now
#   Elevated   - running above its normal band but not accelerating
#   Stable     - within its normal range
#   Cooling    - trending down from a recent elevated level
# plus a mini sparkline, the latest complete-month level vs its baseline, and the
# top topics driving the risk right now.
#
# Design notes:
#   - The CURRENT calendar month is almost always PARTIAL (the month isn't over),
#     so its intensity is artificially low. We detect and EXCLUDE the partial
#     month from all comparisons, comparing recent COMPLETE months instead - else
#     every risk would look like it's "cooling".
#   - Baseline = trailing median of complete months; "above band" = latest
#     complete month exceeds median + k*MAD (robust to spikes).
#   - Momentum = recent 3 complete months vs the prior 3.
#
# Standalone (doesn't touch the pipeline). Reads output/enterprise_risks_*.csv.
#   python risk_status_grid.py
#   python risk_status_grid.py --risk-type emerging --out reports/risk_status.html

import argparse
import html as _html
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

CORE, ACCENT, GRAY = "#233E66", "#0082D1", "#595959"
GREEN, YELLOW, RED, ORANGE = "#9EC405", "#FFCF40", "#FB483D", "#f2872f"

# Minimum latest-month intensity for a risk to qualify as Escalating/Elevated.
# Below this, coverage is too thin to be a "look now" signal regardless of the
# relative move. Tunable via --min-volume; a starting value to set WITH ERM.
MIN_VOLUME = 10.0

# Status -> (color, blurb). Ordering also drives sort priority.
STATUS = {
    "Escalating": (RED, "Rising fast or newly above normal — review now."),
    "Elevated":   (ORANGE, "Running above its normal range — keep watch."),
    "Stable":     (GREEN, "Within its normal range."),
    "Cooling":    (ACCENT, "Easing from a recent elevated level."),
    "Quiet":      (GRAY, "Little to no coverage this period."),
}
STATUS_ORDER = {k: i for i, k in enumerate(STATUS)}


def _risk_labels(risk_type):
    path = f"data/{'Enterprise' if risk_type == 'enterprise' else 'Emerging'}RisksListEncoded.csv"
    id_col = "ENTERPRISE_RISK_ID" if risk_type == "enterprise" else "EMERGING_RISK_ID"
    try:
        from topic_clustering import load_risk_map
        m = load_risk_map(path, id_col)
        return {str(k): (v.get("label") or f"Risk {k}") for k, v in m.items()}
    except Exception:
        return {}


def _months_complete(months):
    """Return (complete_months, partial_flag). The most recent month is treated
    as partial if it matches the current calendar month."""
    if not months:
        return [], False
    now = datetime.now().strftime("%Y-%m")
    if months[-1] == now:
        return months[:-1], True
    return months, False


def _classify(series_complete, min_volume=MIN_VOLUME):
    """Classify a per-risk intensity series (complete months only).

    A minimum-volume floor keeps trivially-small risks out of the "look now"
    tiers: even if a risk's latest month is above its own (tiny) band, an
    absolute level below `min_volume` can't read Escalating/Elevated — those
    tiers are reserved for material coverage. Such risks fall back to Stable,
    or Quiet if there's essentially nothing.
    """
    s = np.array(series_complete, dtype=float)
    nonzero = s[s > 0]
    if len(s) < 2 or nonzero.sum() == 0:
        return "Quiet", {}

    latest = s[-1]
    # Robust baseline from the trailing window (exclude the latest point).
    hist = s[:-1] if len(s) > 1 else s
    median = float(np.median(hist))
    mad = float(np.median(np.abs(hist - median))) or (float(np.std(hist)) or 1.0)
    band_hi = median + 1.5 * mad

    # Momentum: mean of last 3 complete months vs prior 3.
    recent = s[-3:].mean()
    prior = s[-6:-3].mean() if len(s) >= 6 else s[:-3].mean() if len(s) > 3 else median
    denom = prior if prior > 0 else 1.0
    mom_pct = (recent - prior) / denom  # fractional change

    above = latest > band_hi
    # Volume floor: below this absolute level, coverage is too thin to warrant
    # an "escalating/elevated" flag no matter the relative move.
    material = latest >= min_volume
    detail = {
        "latest": round(latest, 1), "median": round(median, 1),
        "band_hi": round(band_hi, 1), "mom_pct": round(mom_pct * 100),
        "material": bool(material),
    }

    if material and (mom_pct >= 0.5 or (above and mom_pct >= 0.15)):
        return "Escalating", detail
    if material and above:
        return "Elevated", detail
    if mom_pct <= -0.3 and (len(hist) and median > 0):
        return "Cooling", detail
    return "Stable", detail


ARTICLES_PER_TOPIC = 6   # max articles shown per driving topic in a risk's pop-up
TOPICS_PER_RISK = 8      # max distinct topics represented in a risk's pop-up


def _load_risk_articles(risk_type):
    """Per-risk article list from the topics CSV, for the pop-up.

    Balanced ACROSS a risk's driving topics rather than a flat recency cut: we
    take the top articles from each topic (largest topics first), so every
    theme that drives the risk is represented. A naive "most-recent N" would let
    one busy topic crowd out the others (e.g. Ponzi coverage hiding the smaller
    Prediction-Markets theme). Deduped to distinct stories; the noisy
    "Outlier / Unclustered" bucket is excluded.

    Returns {risk_id: [ {title, link, source, date, sentiment, topic}, ... ]}.
    """
    path = Path(f"output/{risk_type}_risks_topics.csv")
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if "RISK_ID" not in df.columns or "LINK" not in df.columns:
        return {}
    df["RISK_ID"] = df["RISK_ID"].astype(str).str.replace(".0", "", regex=False)
    df["_d"] = pd.to_datetime(df.get("PUBLISHED_DATE"), errors="coerce")
    df["_topic"] = (df.get("TOPIC_TITLE").fillna("") if "TOPIC_TITLE" in df
                    else df.get("TOPIC_LABEL", "")).astype(str)

    def _is_noise(name):
        n = name.lower()
        return (not n) or "outlier" in n or "unclustered" in n

    def _norm_title(t):
        """Normalize a title for dedup: lowercase, drop a trailing ' - source'
        or ' | source' suffix, collapse whitespace/punctuation edges."""
        t = str(t or "").strip().lower()
        for sep in (" - ", " | ", " — "):
            if sep in t:
                t = t.split(sep)[0].strip()
        return t.strip(" .:-—|")

    by_risk = {}
    for rid, g in df.groupby("RISK_ID"):
        # order topics by how much coverage they carry for this risk (desc)
        topic_sizes = (g[~g["_topic"].map(_is_noise)]
                       .groupby("_topic").size().sort_values(ascending=False))
        # seen sets are per-RISK (not per-topic) so a story can't repeat across
        # a risk's topics, and title-level dedup catches syndicated reprints that
        # slipped into different DUP_GROUPs under slightly different URLs.
        arts, seen_groups, seen_links, seen_titles = [], set(), set(), set()
        for topic in list(topic_sizes.index)[:TOPICS_PER_RISK]:
            tg = g[g["_topic"] == topic].sort_values("_d", ascending=False)
            picked = 0
            for _, r in tg.iterrows():
                grp = r.get("DUP_GROUP")
                link = str(r.get("LINK", "") or "").strip()
                ntitle = _norm_title(r.get("TITLE"))
                if not link.startswith("http") or link in seen_links:
                    continue
                if pd.notna(grp) and grp in seen_groups:
                    continue
                if ntitle and ntitle in seen_titles:
                    continue
                seen_groups.add(grp)
                seen_links.add(link)
                if ntitle:
                    seen_titles.add(ntitle)
                arts.append({
                    "title": str(r.get("TITLE", "") or "").strip() or "(untitled)",
                    "link": link,
                    "source": str(r.get("SOURCE", "") or "").strip(),
                    "date": str(r.get("PUBLISHED_DATE", "") or "")[:10],
                    "sentiment": str(r.get("SENTIMENT", "") or "").strip(),
                    "topic": topic,
                })
                picked += 1
                if picked >= ARTICLES_PER_TOPIC:
                    break
        by_risk[rid] = arts
    return by_risk


def load_grid(risk_type, min_volume=MIN_VOLUME):
    br_path = Path(f"output/{risk_type}_risks_trends_by_risk.csv")
    tt_path = Path(f"output/{risk_type}_risks_trends_topics.csv")
    if not br_path.exists():
        raise SystemExit(f"Missing {br_path}. Run topic_trends.py first.")
    br = pd.read_csv(br_path)
    br["RISK_ID"] = br["RISK_ID"].astype(str).str.replace(".0", "", regex=False)
    months = sorted(br["month"].dropna().unique())
    complete, partial = _months_complete(months)

    # topic labels + which risks each drives, for the "drivers" list
    labels, topic_risk = {}, {}
    if tt_path.exists():
        tt = pd.read_csv(tt_path)
        for _, r in tt.iterrows():
            tid = int(r["TOPIC_ID"])
            labels[tid] = str(r.get("TOPIC_TITLE") or r.get("TOPIC_LABEL") or f"Topic {tid}")

    risk_labels = _risk_labels(risk_type)
    risk_articles = _load_risk_articles(risk_type)
    tiles = []
    for rid, g in br.groupby("RISK_ID"):
        piv = g.groupby("month")["intensity"].sum()
        full_series = [round(float(piv.get(m, 0.0)), 1) for m in months]
        comp_series = [round(float(piv.get(m, 0.0)), 1) for m in complete]
        status, detail = _classify(comp_series, min_volume=min_volume)

        # top driving topics by recent (last complete month) intensity
        recent_month = complete[-1] if complete else (months[-1] if months else None)
        drivers = []
        if recent_month is not None:
            rg = g[g["month"] == recent_month].sort_values("intensity", ascending=False)
            for _, r in rg.head(3).iterrows():
                if float(r["intensity"]) <= 0:
                    continue
                drivers.append(labels.get(int(r["TOPIC_ID"]), f"Topic {int(r['TOPIC_ID'])}"))

        tiles.append({
            "rid": rid,
            "name": risk_labels.get(rid, f"Risk {rid}"),
            "status": status,
            "detail": detail,
            "series": full_series,      # full incl. partial (sparkline shows dashed tail)
            "series_complete": comp_series,
            "drivers": drivers,
            "articles": risk_articles.get(rid, []),
        })

    tiles.sort(key=lambda t: (STATUS_ORDER.get(t["status"], 9), -t["detail"].get("latest", 0)))
    return tiles, months, complete, partial


def _sparkline(series, complete_n, w=150, h=34):
    if not series:
        return ""
    mx = max(series) or 1.0
    n = len(series)
    step = w / max(1, n - 1)
    pts = [(i * step, h - (v / mx) * (h - 4) - 2) for i, v in enumerate(series)]
    # solid path for complete months, dashed for the trailing partial month
    solid = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts[:complete_n])
    parts = [f'<svg viewBox="0 0 {w} {h}" class="spark" preserveAspectRatio="none">']
    if solid:
        parts.append(f'<polyline points="{solid}" fill="none" stroke="{CORE}" stroke-width="1.5"/>')
    if complete_n < n:  # dashed connector into the partial month
        tail = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts[complete_n - 1:])
        parts.append(f'<polyline points="{tail}" fill="none" stroke="{GRAY}" '
                     f'stroke-width="1.2" stroke-dasharray="3 2"/>')
    lx, ly = pts[complete_n - 1] if complete_n else pts[-1]
    parts.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="2.2" fill="{ACCENT}"/>')
    parts.append("</svg>")
    return "".join(parts)


def _tile(t):
    color = STATUS[t["status"]][0]
    d = t["detail"]
    name = _html.escape(t["name"])
    mom = d.get("mom_pct")
    mom_html = ""
    if mom is not None and t["status"] != "Quiet":
        arrow = "▲" if mom > 0 else ("▼" if mom < 0 else "→")
        mom_html = f'<span class="mom">{arrow} {abs(mom):.0f}% vs prior qtr</span>'
    drivers = ""
    if t["drivers"]:
        chips = "".join(f'<span class="dchip">{_html.escape(x)}</span>' for x in t["drivers"])
        drivers = f'<div class="drivers"><span class="dlbl">Driven by</span>{chips}</div>'
    level = ""
    if d:
        level = (f'<div class="lvl">latest <b>{d.get("latest","—")}</b> '
                 f'· normal ≤ {d.get("band_hi","—")}</div>')
    spark = _sparkline(t["series"], len(t["series_complete"]))
    n_art = len(t.get("articles") or [])
    mid = f"risk-{t['rid']}"
    clickable = n_art > 0
    view = (f'<div class="tile-view">{n_art} article{"s" if n_art != 1 else ""} — click to view ▸</div>'
            if clickable else '<div class="tile-view muted">No article links available</div>')
    attrs = (f'class="tile{"" if not clickable else " clickable"}" '
             f'style="border-top-color:{color}"')
    if clickable:
        attrs += f' role="button" tabindex="0" data-modal="{mid}" aria-haspopup="dialog"'
    return f"""    <div {attrs}>
      <div class="tile-top">
        <span class="status" style="background:{color}">{t['status']}</span>
        {mom_html}
      </div>
      <div class="rname">{name}</div>
      <div class="spark-wrap">{spark}</div>
      {level}
      {drivers}
      {view}
    </div>"""


def _modal(t):
    """Hidden dialog listing a risk's articles, grouped by driving topic."""
    arts = t.get("articles") or []
    if not arts:
        return ""
    color = STATUS[t["status"]][0]
    name = _html.escape(t["name"])
    mid = f"risk-{t['rid']}"
    # group by topic, preserving recency order within each group
    groups = {}
    order = []
    for a in arts:
        key = a.get("topic") or "Other coverage"
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(a)
    blocks = []
    for key in order:
        lis = []
        for a in groups[key]:
            link = _html.escape(a["link"], quote=True)
            title = _html.escape(a["title"])
            meta = " · ".join(x for x in (_html.escape(a.get("source", "")),
                                          _html.escape(a.get("date", ""))) if x)
            sent = a.get("sentiment", "")
            sdot = ""
            if sent:
                sc = {"Negative": RED, "Positive": GREEN}.get(sent, GRAY)
                sdot = f'<span class="sdot" style="background:{sc}" title="{_html.escape(sent)}"></span>'
            meta_html = f'<span class="m-meta">{sdot}{meta}</span>' if (meta or sdot) else ""
            lis.append(f'<li><a href="{link}" target="_blank" rel="noopener noreferrer">'
                       f'{title}</a>{meta_html}</li>')
        blocks.append(f'<div class="m-group"><h4>{_html.escape(key)}</h4>'
                      f'<ul>{"".join(lis)}</ul></div>')
    return f"""  <div class="modal" id="{mid}" role="dialog" aria-modal="true" aria-label="Articles for {name}">
    <div class="modal-box">
      <div class="modal-head" style="border-color:{color}">
        <div><span class="status" style="background:{color}">{t['status']}</span>
          <span class="modal-title">{name}</span></div>
        <button class="modal-x" aria-label="Close">&times;</button>
      </div>
      <div class="modal-body">
        <p class="modal-note">{len(arts)} recent article(s) driving this risk's news signal,
        grouped by theme. Links open the original source. External coverage for review — confirm
        against internal data before acting.</p>
        {"".join(blocks)}
      </div>
    </div>
  </div>"""


def _legend():
    items = "".join(
        f'<span class="lg"><span class="sw" style="background:{c}"></span>'
        f'<b>{k}</b> — {_html.escape(blurb)}</span>'
        for k, (c, blurb) in STATUS.items())
    return f'<div class="legend">{items}</div>'


def build(risk_type, out_path, min_volume=MIN_VOLUME):
    tiles, months, complete, partial = load_grid(risk_type, min_volume=min_volume)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    span = f"{complete[0]} → {complete[-1]}" if complete else "—"
    partial_note = ""
    if partial:
        partial_note = (f'<span class="pnote">Current month ({months[-1]}) is partial and '
                        f'excluded from status; shown dashed on sparklines.</span>')
    grid = "\n".join(_tile(t) for t in tiles)
    modals = "\n".join(_modal(t) for t in tiles)
    counts = {}
    for t in tiles:
        counts[t["status"]] = counts.get(t["status"], 0) + 1
    summary = " · ".join(f'{v} {k}' for k, v in
                         sorted(counts.items(), key=lambda kv: STATUS_ORDER.get(kv[0], 9)))
    title = f"{'Enterprise' if risk_type=='enterprise' else 'Emerging'} Risk Status"

    html = _TEMPLATE
    for k, v in {
        "__TITLE__": title, "__SUMMARY__": summary, "__SPAN__": span,
        "__PARTIAL__": partial_note, "__LEGEND__": _legend(), "__GRID__": grid,
        "__MODALS__": modals, "__GENERATED__": generated,
    }.items():
        html = html.replace(k, v)

    out = Path(out_path)
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote risk status grid ({len(tiles)} risks) -> {out}")
    print(f"  status mix: {summary}")


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700;800&family=Lora:ital,wght@0,400;1,400&display=swap" rel="stylesheet">
<style>
  :root{--core:#233E66;--accent:#0082D1;--gray:#595959;--green:#9EC405;--yellow:#FFCF40;
    --red:#FB483D;--bg:#f4f6f9;--panel:#fff;--text:#333;--muted:#6b7280;--border:#e2e6ec;--panel2:#eef3fa;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);
    font:15px/1.6 'Open Sans',-apple-system,Segoe UI,Roboto,Arial,sans-serif;}
  .logoband{background:#fff;padding:14px 32px;}
  .logoband .logo{height:44px;width:auto;display:block;}
  .brandbar{height:6px;background:var(--yellow);}
  .topbar{background:var(--core);color:#fff;padding:22px 32px;}
  .topbar .wrap{max-width:1120px;margin:0 auto;}
  a.back{color:#cfe0f5;text-decoration:none;font-size:13px;}
  a.back:hover{color:#fff;text-decoration:underline;}
  h1{margin:8px 0 4px;font-size:24px;font-weight:800;}
  .tagline{color:#9db4d6;font-family:'Lora',Georgia,serif;font-style:italic;font-size:12px;}
  .sub{color:#cfdaea;font-size:13.5px;margin-top:8px;}
  .sub b{color:#fff;}
  main{max-width:1120px;margin:0 auto;padding:20px 32px 60px;}
  .legend{display:flex;flex-wrap:wrap;gap:14px;background:var(--panel);border:1px solid var(--border);
    border-radius:10px;padding:12px 16px;margin-bottom:18px;font-size:12.5px;color:var(--muted);}
  .legend .lg{display:flex;align-items:center;gap:6px;}
  .legend .sw{width:11px;height:11px;border-radius:3px;display:inline-block;}
  .legend b{color:var(--core);}
  .pnote{display:block;color:var(--muted);font-size:12px;margin-bottom:14px;}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:14px;}
  .tile{background:var(--panel);border:1px solid var(--border);border-top:4px solid var(--gray);
    border-radius:10px;padding:14px 16px;box-shadow:0 1px 2px rgba(16,36,66,.05);}
  .tile-top{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:8px;}
  .status{color:#fff;font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.04em;
    border-radius:999px;padding:2px 10px;}
  .mom{color:var(--muted);font-size:11.5px;font-weight:700;}
  .rname{font-size:15px;font-weight:700;color:var(--core);line-height:1.3;min-height:38px;}
  .spark-wrap{margin:6px 0 8px;}
  .spark{width:100%;height:34px;display:block;}
  .lvl{color:var(--muted);font-size:12px;}
  .lvl b{color:var(--core);}
  .drivers{margin-top:10px;display:flex;flex-wrap:wrap;gap:5px;align-items:center;}
  .drivers .dlbl{color:var(--muted);font-size:11px;margin-right:2px;}
  .dchip{background:var(--panel2);border:1px solid var(--border);color:var(--gray);
    border-radius:999px;padding:2px 8px;font-size:11px;}
  .tile.clickable{cursor:pointer;transition:box-shadow .12s,transform .12s;}
  .tile.clickable:hover{box-shadow:0 4px 14px rgba(16,36,66,.12);transform:translateY(-2px);}
  .tile.clickable:focus{outline:2px solid var(--accent);outline-offset:2px;}
  .tile-view{margin-top:10px;color:var(--accent);font-size:11.5px;font-weight:700;}
  .tile-view.muted{color:var(--muted);font-weight:400;}
  /* modal */
  .modal{display:none;position:fixed;inset:0;z-index:50;background:rgba(16,36,66,.45);
    padding:40px 16px;overflow-y:auto;}
  .modal.open{display:block;}
  .modal-box{background:#fff;max-width:640px;margin:0 auto;border-radius:12px;
    box-shadow:0 12px 40px rgba(16,36,66,.3);overflow:hidden;}
  .modal-head{display:flex;justify-content:space-between;align-items:center;gap:12px;
    padding:14px 20px;border-bottom:3px solid var(--core);}
  .modal-title{font-weight:800;color:var(--core);font-size:16px;margin-left:8px;}
  .modal-x{background:none;border:none;font-size:26px;line-height:1;color:var(--muted);
    cursor:pointer;padding:0 4px;}
  .modal-x:hover{color:var(--core);}
  .modal-body{padding:16px 20px 22px;max-height:66vh;overflow-y:auto;}
  .modal-note{color:var(--muted);font-size:12.5px;margin:0 0 14px;}
  .m-group{margin-bottom:16px;}
  .m-group h4{margin:0 0 6px;font-size:13px;color:var(--core);font-weight:800;
    border-bottom:1px solid var(--border);padding-bottom:4px;}
  .m-group ul{margin:0;padding-left:18px;}
  .m-group li{margin:6px 0;font-size:13.5px;line-height:1.4;}
  .m-group a{color:var(--accent);text-decoration:none;}
  .m-group a:hover{text-decoration:underline;}
  .m-meta{color:var(--muted);font-size:11.5px;margin-left:8px;}
  .sdot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px;
    vertical-align:middle;}
  .how{background:#eef3fa;border:1px solid var(--border);border-left:4px solid var(--accent);
    border-radius:8px;padding:14px 18px;margin-top:22px;color:#2b3a52;font-size:13px;max-width:80ch;}
  .how b{color:var(--core);}
  footer{max-width:1120px;margin:0 auto;padding:20px 32px 40px;color:var(--muted);
    font-size:12px;border-top:1px solid var(--border);}
</style></head><body>
<div class="logoband"><img class="logo" src="finra%20logo.png" alt="FINRA"></div>
<div class="brandbar"></div>
<div class="topbar"><div class="wrap">
  <a class="back" href="index.html">← Back to reports</a>
  <h1>__TITLE__</h1>
  <div class="tagline">Investor protection. Market integrity.</div>
  <div class="sub">News-signal status by risk · <b>__SUMMARY__</b> · coverage __SPAN__</div>
</div></div>
<main>
  __LEGEND__
  __PARTIAL__
  <div class="grid">
__GRID__
  </div>
  <div class="how">
    <b>How to read this.</b> Each tile summarizes one enterprise risk's <b>news signal</b> —
    coverage volume weighted by negative sentiment — into a status for triage. <b>Escalating</b>
    and <b>Elevated</b> tiles are where to look first. Status compares recent complete months to
    each risk's own normal range, so a risk is flagged relative to itself, not to louder risks.
    This is an <b>external early-warning input for review</b>, not a risk assessment — confirm
    against internal data and KRIs before acting.
  </div>
</main>
__MODALS__
<footer>Generated __GENERATED__ · prototype for ERM review · self-contained HTML.</footer>
<script>
  (function () {
    function open(id) {
      var m = document.getElementById(id);
      if (!m) return;
      m.classList.add("open");
      document.body.style.overflow = "hidden";
      var x = m.querySelector(".modal-x");
      if (x) x.focus();
    }
    function closeAll() {
      document.querySelectorAll(".modal.open").forEach(function (m) { m.classList.remove("open"); });
      document.body.style.overflow = "";
    }
    // open from a tile
    document.querySelectorAll(".tile.clickable[data-modal]").forEach(function (tile) {
      tile.addEventListener("click", function () { open(tile.getAttribute("data-modal")); });
      tile.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(tile.getAttribute("data-modal")); }
      });
    });
    // close: X button, click on backdrop, Esc
    document.querySelectorAll(".modal").forEach(function (m) {
      m.addEventListener("click", function (e) {
        if (e.target === m || e.target.classList.contains("modal-x")) closeAll();
      });
    });
    document.addEventListener("keydown", function (e) { if (e.key === "Escape") closeAll(); });
  })();
</script>
</body></html>
"""


def main():
    p = argparse.ArgumentParser(description="Build a per-risk status grid from trend data.")
    p.add_argument("--risk-type", choices=["enterprise", "emerging"], default="enterprise")
    p.add_argument("--out", default=None)
    p.add_argument("--min-volume", type=float, default=MIN_VOLUME,
                   help=f"Min latest-month intensity to allow Escalating/Elevated (default {MIN_VOLUME}).")
    args = p.parse_args()
    out = args.out or f"reports/risk_status_{args.risk_type}.html"
    build(args.risk_type, out, min_volume=args.min_volume)


if __name__ == "__main__":
    main()
