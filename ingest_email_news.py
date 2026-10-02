# EMAIL NEWS INGEST  (Path C: curated daily digest -> clustering pipeline)
#
# A daily FINRA/financial-news digest email is a HIGHER-PRECISION source than
# the newsdata.io API: it's already curated to financially-relevant and
# FINRA-specific stories, so it sidesteps the sports/entertainment noise that
# plagues the broad feeds.
#
# Copilot (which can read the mailbox) extracts the email; this script turns it
# into the SAME CSV schema the rest of the pipeline consumes, tagged
# SEARCH_TERM_ID = "EMAIL", so clustering / trends / discovery / status all work
# unchanged. The two tools meet at a file - they never call each other.
#
# Two accepted inputs (auto-detected):
#   1. Structured CSV from Copilot, columns (case-insensitive, extras ignored):
#        TITLE, LINK, SOURCE, PUBLISHED_DATE, SUMMARY, [SECTION], [PAYWALLED]
#   2. Raw digest text (.txt) saved from the email - parsed with the format
#      heuristics below (source+date line, headline, byline, summary, the
#      "Also:/Related:/Without FINRA mention:" sub-lists, "(Paywalled)" flags).
#
# Output: appended + de-duplicated into output/email_news.csv (its own lane),
# full pipeline schema. Full article text is fetched via article_text.py when a
# non-paywalled LINK is present; otherwise the email SUMMARY is used.
#
# Usage:
#   python ingest_email_news.py --in inbox/2026-09-30.txt
#   python ingest_email_news.py --in email_news.csv
#   python ingest_email_news.py --in inbox/digest.txt --no-fulltext   # faster

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

OUTPUT_CSV = Path("output/email_news.csv")
SUMMARY_CHARS = 4000

PIPELINE_COLUMNS = [
    "RISK_ID", "SEARCH_TERM_ID", "GOOGLE_INDEX", "TITLE", "LINK",
    "PUBLISHED_DATE", "SUMMARY", "KEYWORDS", "SENTIMENT_COMPOUND",
    "SENTIMENT", "SOURCE", "CATEGORY", "QUALITY_SCORE",
]

# Lines that are navigation / chrome, not stories.
_SKIP_LINES = {
    "read article online", "read full text here", "back to index", "also:",
    "related:", "without finra mention:",
}
# A "Source, Month DD, YYYY" header line, e.g. "JDSupra, September 30, 2026".
# The source class allows curly apostrophes/dashes (e.g. "Barron’s", "24/7").
_SRC_DATE = re.compile(
    r"^(?P<source>[A-Z0-9][\w .&'’/-]{1,60}),\s+"
    r"(?P<date>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},\s+\d{4})\s*$"
)
# A trailing "- Publication (By Author) (Paywalled)" in an Also/Related list
# item. Handles hyphen, en/em dashes as the title↔source separator.
_ALSO_SRC = re.compile(r"\s[–—-]\s([^–—-]+?)(?:\s*\(By [^)]+\))?(?:\s*\(Paywalled\))?\s*$")


def _parse_date(s):
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%b. %d, %Y"):
        try:
            return datetime.strptime(s.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return ""


def _is_section_header(line):
    """Top-level section names in the digest (short, title-ish, no comma-date)."""
    s = line.strip()
    if not s or len(s) > 40:
        return False
    known = {"finra in the news", "sec", "financial regulation", "enforcement",
             "markets", "cryptocurrency", "crypto", "fintech", "banking"}
    return s.lower() in known


def parse_raw_digest(text):
    """Parse the raw email digest text into story dicts. Heuristic but tuned to
    the observed format; resilient to missing pieces."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    stories = []
    section = ""
    i, n = 0, len(lines)

    while i < n:
        raw = lines[i]
        line = raw.strip()
        if not line:
            i += 1
            continue
        low = line.lower()

        if _is_section_header(line):
            section = line
            i += 1
            continue
        if low in _SKIP_LINES:
            i += 1
            continue

        m = _SRC_DATE.match(line)
        if m:
            source = m.group("source").strip()
            date = _parse_date(m.group("date"))
            # Next non-empty line = headline.
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            if j >= n:
                break
            headline = lines[j].strip()
            # Optional byline "By ..." then summary paragraph(s) until a
            # Read.../Also:/next source-date/section boundary.
            k = j + 1
            byline = ""
            if k < n and lines[k].strip().startswith("By "):
                byline = lines[k].strip()[3:].strip()
                k += 1
            summary_parts = []
            while k < n:
                s = lines[k].strip()
                sl = s.lower()
                if (not s or sl in _SKIP_LINES or _SRC_DATE.match(s)
                        or _is_section_header(s)):
                    break
                summary_parts.append(s)
                k += 1
            summary = " ".join(summary_parts).strip()
            paywalled = "(paywalled)" in (headline + summary).lower()
            headline = re.sub(r"\s*\(Paywalled\)\s*$", "", headline).strip()

            stories.append({
                "title": headline, "source": source, "date": date,
                "summary": summary, "byline": byline, "section": section,
                "paywalled": paywalled, "link": "", "primary": True,
            })

            # Skip the "Read article online / Read full text here" line(s) and
            # blanks that sit between the summary and any "Also:/Related:" block.
            while k < n and (not lines[k].strip()
                             or lines[k].strip().lower() in ("read article online",
                                                             "read full text here")):
                k += 1

            # Collect following "Also:/Related:/Without FINRA mention:" list items
            # as linked, lower-weight sibling stories (same event, other outlets).
            sub_headers = ("also:", "related:", "without finra mention:")
            while k < n and lines[k].strip().lower() in sub_headers:
                k += 1  # consume the sub-header
                while k < n:
                    item = lines[k].strip()
                    il = item.lower()
                    if not item:
                        k += 1
                        continue
                    if (_SRC_DATE.match(item) or _is_section_header(item)
                            or il in sub_headers or il in _SKIP_LINES):
                        break
                    am = _ALSO_SRC.search(item)
                    sib_src = am.group(1).strip() if am else ""
                    sib_title = _ALSO_SRC.sub("", item).strip()
                    sib_pay = "(paywalled)" in item.lower()
                    sib_title = re.sub(r"\s*\(Paywalled\)\s*$", "", sib_title).strip()
                    sib_src = re.sub(r"\s*\(Paywalled\)\s*$", "", sib_src).strip()
                    if sib_title:
                        stories.append({
                            "title": sib_title, "source": sib_src, "date": date,
                            "summary": "", "byline": "", "section": section,
                            "paywalled": sib_pay, "link": "", "primary": False,
                        })
                    k += 1
            i = k
            continue
        i += 1

    return stories


def load_structured_csv(path):
    """Accept a Copilot-produced CSV; map columns case-insensitively."""
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}

    def col(*names):
        for nm in names:
            if nm in cols:
                return cols[nm]
        return None

    out = []
    for _, r in df.iterrows():
        title = str(r.get(col("title", "headline"), "") or "").strip()
        if not title:
            continue
        pay = str(r.get(col("paywalled"), "") or "").strip().lower() in ("1", "true", "yes", "y")
        out.append({
            "title": title,
            "source": str(r.get(col("source", "publication"), "") or "").strip(),
            "date": _normalize_any_date(str(r.get(col("published_date", "date"), "") or "")),
            "summary": str(r.get(col("summary", "snippet", "description"), "") or "").strip(),
            "byline": str(r.get(col("byline", "author"), "") or "").strip(),
            "section": str(r.get(col("section", "category"), "") or "").strip(),
            "paywalled": pay,
            "link": str(r.get(col("link", "url"), "") or "").strip(),
            "primary": True,
        })
    return out


def _normalize_any_date(s):
    s = s.strip()
    if not s:
        return ""
    d = _parse_date(s)
    if d:
        return d
    try:
        return pd.to_datetime(s, errors="coerce").strftime("%Y-%m-%d")
    except Exception:
        return ""


def to_rows(stories, fetch_fulltext=True):
    """Turn parsed stories into full-schema pipeline rows (sentiment + quality)."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    try:
        from content_quality import quality_score
    except Exception:
        quality_score = None
    if fetch_fulltext:
        try:
            from article_text import fetch_article_text
        except Exception:
            fetch_fulltext = False

    rows = []
    for idx, st in enumerate(stories):
        summary = st["summary"]
        link = st.get("link", "")
        if fetch_fulltext and link.startswith("http") and not st["paywalled"]:
            summary = fetch_article_text(link, fallback=summary) or summary
        text = f"{st['title']} {summary}".strip()
        sent = analyzer.polarity_scores(text)
        compound = sent["compound"]
        label = ("Positive" if compound >= 0.05 else
                 "Negative" if compound <= -0.05 else "Neutral")
        row = {
            "RISK_ID": -1,
            "SEARCH_TERM_ID": "EMAIL",
            "GOOGLE_INDEX": idx,
            "TITLE": st["title"],
            "LINK": link,
            "PUBLISHED_DATE": st["date"],
            "SUMMARY": summary[:SUMMARY_CHARS],
            "KEYWORDS": "",
            "SENTIMENT_COMPOUND": compound,
            "SENTIMENT": label,
            "SOURCE": st["source"],
            # Section becomes CATEGORY; prefix marks the email lane + paywalled.
            "CATEGORY": f"email:{st.get('section','') or 'general'}"
                        + (":paywalled" if st["paywalled"] else ""),
            "QUALITY_SCORE": 0,
        }
        if quality_score:
            try:
                row["QUALITY_SCORE"] = quality_score(row)
            except Exception:
                pass
        rows.append(row)
    return rows


def save(rows):
    OUTPUT_CSV.parent.mkdir(exist_ok=True)
    new_df = pd.DataFrame(rows, columns=PIPELINE_COLUMNS)
    if OUTPUT_CSV.exists():
        try:
            old = pd.read_csv(OUTPUT_CSV)
            combined = pd.concat([old, new_df], ignore_index=True)
        except Exception:
            combined = new_df
    else:
        combined = new_df
    # De-dup: prefer a real LINK; else fall back to (TITLE, SOURCE).
    combined["_k"] = combined.apply(
        lambda r: (str(r["LINK"]).strip().lower() or
                   f"{str(r['TITLE']).strip().lower()}|{str(r['SOURCE']).strip().lower()}"),
        axis=1)
    before = len(combined)
    combined = combined.drop_duplicates(subset="_k", keep="first").drop(columns="_k")
    combined["GOOGLE_INDEX"] = range(len(combined))
    combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
    return len(new_df), len(combined), before - len(combined)


def main():
    p = argparse.ArgumentParser(description="Ingest a curated news-digest email into the pipeline.")
    p.add_argument("--in", dest="inp", required=True, help="Path to the email digest (.txt) or Copilot CSV.")
    p.add_argument("--no-fulltext", action="store_true", help="Skip full-text fetch (use email summaries only).")
    args = p.parse_args()

    path = Path(args.inp)
    if not path.exists():
        print(f"ERROR: input not found: {path}")
        return

    if path.suffix.lower() == ".csv":
        stories = load_structured_csv(path)
        kind = "structured CSV"
    else:
        stories = parse_raw_digest(path.read_text(encoding="utf-8", errors="replace"))
        kind = "raw digest text"

    primaries = sum(1 for s in stories if s.get("primary"))
    print(f"Parsed {len(stories)} stories ({primaries} primary, "
          f"{len(stories)-primaries} 'also/related') from {kind}.")
    if not stories:
        print("Nothing to ingest.")
        return

    rows = to_rows(stories, fetch_fulltext=not args.no_fulltext)
    added, total, deduped = save(rows)
    print(f"Ingested {added} rows -> {OUTPUT_CSV} "
          f"({total} total after de-dup, {deduped} duplicates dropped).")
    print("Next: cluster with  python discover_score.py --input output/email_news.csv "
          "--out output/email_ranked.csv --source all")


if __name__ == "__main__":
    main()
