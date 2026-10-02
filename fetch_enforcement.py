# FINRA ENFORCEMENT FETCHER  (Path F: monthly disciplinary actions)
#
# FINRA publishes a monthly PDF compiling every disciplinary action it took -
# firms fined, individuals barred/suspended, AWCs, etc. This is ground-truth
# signal about where member-firm risk is actually materializing: the violations
# FINRA chose to sanction. Public domain.
#
# This fetcher enumerates the recent monthly PDFs from the disciplinary-actions
# landing page, downloads each, extracts text with pypdf, and splits the
# compilation into INDIVIDUAL actions (one per firm/individual) using the
# "Name (CRD #NNNN...)" header that begins each case. Each action becomes one
# row in the pipeline schema, tagged SEARCH_TERM_ID = "ENFORCEMENT", so it flows
# through clustering / discovery / status like every other source.
#
# Output: output/enforcement_news.csv (appended + de-duplicated).
#
# Usage:
#   python fetch_enforcement.py                 # recent months (default 3)
#   python fetch_enforcement.py --months 6
#   python fetch_enforcement.py --max-actions 0 # no per-month cap

import argparse
import csv
import io
import re
import time
from datetime import datetime
from pathlib import Path

import requests

try:
    import pypdf
    _HAVE_PYPDF = True
except Exception:
    _HAVE_PYPDF = False

LANDING = "https://www.finra.org/rules-guidance/oversight-enforcement/disciplinary-actions"
BASE = "https://www.finra.org"
OUTPUT_CSV = Path("output/enforcement_news.csv")

SUMMARY_CHARS = 4000
MIN_ACTION_CHARS = 120
REQUEST_TIMEOUT = 60
THROTTLE_SEC = 1.0
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0

HEADERS = {
    "User-Agent": "FINRA ERM research prototype (contact: erm-research@finra.org)",
    "Accept": "text/html,application/xhtml+xml,application/pdf",
    "Accept-Encoding": "gzip, deflate",
}

PIPELINE_COLUMNS = [
    "RISK_ID", "SEARCH_TERM_ID", "GOOGLE_INDEX", "TITLE", "LINK",
    "PUBLISHED_DATE", "SUMMARY", "KEYWORDS", "SENTIMENT_COMPOUND",
    "SENTIMENT", "SOURCE", "CATEGORY", "QUALITY_SCORE",
]

MONTHS = {m.lower(): i for i, m in enumerate(
    ["", "January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"])}

# A case header: "Firm or Person Name (CRD #12345, City, State)". CRD is FINRA's
# Central Registration Depository id; every action begins with one.
_CASE_HEADER = re.compile(
    r"([A-Z][^()\n]{2,90}?\(CRD[^)]*#\s*\d[\d,]*[^)]*\))")
# Section headers within the PDF, used to tag an action's category.
_SECTION_RE = re.compile(
    r"\b(Firms?\s+(?:Fined|Expelled|Suspended|Sanctioned|Cancelled)"
    r"|Individuals?\s+(?:Barred|Suspended|Fined|Sanctioned|Revoked)"
    r"|Complaints?\s+Filed|Decisions?\s+Issued)\b", re.I)


def _get(url, session, binary=False):
    delay = RETRY_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200 and (r.content if binary else r.text):
                if not binary and r.apparent_encoding:
                    r.encoding = r.apparent_encoding
                return r.content if binary else r.text
            if r.status_code == 404:
                return b"" if binary else ""
        except requests.RequestException:
            pass
        if attempt < MAX_RETRIES:
            time.sleep(delay)
            delay *= 2
    return b"" if binary else ""


def _clean(t):
    import html as _h
    try:
        from unidecode import unidecode
        t = unidecode(t or "")
    except Exception:
        t = t or ""
    t = _h.unescape(t)
    return re.sub(r"\s+", " ", t).strip()


def enumerate_pdfs(session, months):
    """Return [(pdf_url, YYYY-MM, 'Month YYYY'), ...] newest-first, up to `months`."""
    html = _get(LANDING, session)
    if not html:
        print("  WARN: could not load disciplinary-actions landing page")
        return []
    hrefs = re.findall(r'href="([^"]*disciplinary[-_]actions[-_][^"]*\.pdf)"', html, re.I)
    out, seen = [], set()
    for h in hrefs:
        url = BASE + h if h.startswith("/") else h
        if url in seen:
            continue
        seen.add(url)
        # date from the path: /sites/default/files/YYYY-MM/disciplinary-actions-month-year.pdf
        m = re.search(r"/(\d{4})-(\d{2})/", h)
        mm = re.search(r"disciplinary[-_]actions[-_]([a-z]+)[-_](\d{4})", h, re.I)
        if m:
            ym = f"{m.group(1)}-{m.group(2)}"
            label = (f"{mm.group(1).title()} {mm.group(2)}" if mm else ym)
        elif mm and mm.group(1).lower() in MONTHS:
            ym = f"{mm.group(2)}-{MONTHS[mm.group(1).lower()]:02d}"
            label = f"{mm.group(1).title()} {mm.group(2)}"
        else:
            continue
        out.append((url, ym, label))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:months]


def extract_pdf_text(content):
    if not content or not _HAVE_PYPDF:
        return ""
    try:
        reader = pypdf.PdfReader(io.BytesIO(content))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception:
        return ""


def split_actions(text):
    """Split a monthly compilation into individual actions on the CASE_HEADER.
    Returns [(name, action_text), ...]. Tracks the current section header so each
    action can be categorized (fined / barred / suspended / complaint...)."""
    # Find all case-header positions.
    headers = list(_CASE_HEADER.finditer(text))
    if not headers:
        return []
    actions = []
    for i, h in enumerate(headers):
        start = h.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        seg = text[start:end]
        name = _clean(h.group(1))
        body = _clean(seg)
        if len(body) >= MIN_ACTION_CHARS:
            # section = nearest preceding section header before this action
            pre = text[:start]
            secs = list(_SECTION_RE.finditer(pre))
            section = _clean(secs[-1].group(1)) if secs else "disciplinary action"
            actions.append((name, body, section))
    return actions


def fetch(months, max_actions):
    if not _HAVE_PYPDF:
        print("pypdf not installed. Run: pip install pypdf")
        return []
    session = requests.Session()
    pdfs = enumerate_pdfs(session, months)
    print(f"Found {len(pdfs)} monthly PDF(s) to process.")
    items = []
    for url, ym, label in pdfs:
        content = _get(url, session, binary=True)
        time.sleep(THROTTLE_SEC)
        if not content:
            print(f"  {label}: PDF fetch failed")
            continue
        text = extract_pdf_text(content)
        if not text:
            print(f"  {label}: no text extracted")
            continue
        actions = split_actions(text)
        if max_actions and len(actions) > max_actions:
            actions = actions[:max_actions]
        for name, body, section in actions:
            items.append({
                "name": name, "text": body, "section": section,
                "date": ym + "-01", "label": label, "link": url,
            })
        print(f"  ok {label}: {len(actions)} action(s)")
    return items


def _title(name, body):
    # "<name> - <first clause of the action>"
    first = re.split(r"(?<=[.])\s", body, maxsplit=1)[0]
    first = re.sub(re.escape(name), "", first).strip(" -\u2013")
    t = f"{name} - {first}" if first else name
    return t[:180]


def to_pipeline_rows(items):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    try:
        from content_quality import quality_score
    except Exception:
        quality_score = None
    out = []
    for i, it in enumerate(items):
        text = it["text"][:SUMMARY_CHARS]
        comp = analyzer.polarity_scores(text)["compound"]
        label = ("Positive" if comp >= 0.05 else "Negative" if comp <= -0.05 else "Neutral")
        sec = re.sub(r"\s+", "_", it["section"].lower())
        row = {
            "RISK_ID": -1,
            "SEARCH_TERM_ID": "ENFORCEMENT",
            "GOOGLE_INDEX": i,
            "TITLE": _title(it["name"], text),
            "LINK": it["link"],
            "PUBLISHED_DATE": it["date"],
            "SUMMARY": text,
            "KEYWORDS": "",
            "SENTIMENT_COMPOUND": comp,
            "SENTIMENT": label,
            "SOURCE": "FINRA Disciplinary Actions",
            "CATEGORY": f"enforcement:{sec}",
            "QUALITY_SCORE": 0,
        }
        if quality_score:
            try:
                row["QUALITY_SCORE"] = quality_score(row)
            except Exception:
                pass
        out.append(row)
    return out


def save(rows):
    import pandas as pd
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
    before = len(combined)
    # De-dup on (TITLE, PUBLISHED_DATE): same action in a re-run collapses.
    combined["_k"] = combined["TITLE"].astype(str) + "|" + combined["PUBLISHED_DATE"].astype(str)
    combined = combined.drop_duplicates(subset="_k", keep="first").drop(columns="_k")
    combined["GOOGLE_INDEX"] = range(len(combined))
    combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
    return len(new_df), len(combined), before - len(combined)


def main():
    p = argparse.ArgumentParser(description="Fetch FINRA monthly disciplinary actions into the pipeline schema.")
    p.add_argument("--months", type=int, default=3, help="How many recent monthly PDFs to process (default 3).")
    p.add_argument("--max-actions", type=int, default=0, dest="max_actions",
                   help="Cap actions kept per month (0 = all, default).")
    args = p.parse_args()

    print("#" * 60)
    print("FINRA ENFORCEMENT FETCH (monthly disciplinary actions)")
    print(f"Months: {args.months}" + (f" | cap {args.max_actions}/mo" if args.max_actions else ""))
    print("#" * 60)

    items = fetch(args.months, args.max_actions)
    if not items:
        print("\nNo actions fetched.")
        return
    rows = to_pipeline_rows(items)
    added, total, deduped = save(rows)
    print(f"\nFetched {added} actions -> {OUTPUT_CSV} "
          f"({total} total after de-dup, {deduped} duplicates dropped).")
    print("Cluster with: python discover_score.py --input output/enforcement_news.csv "
          "--out output/enforcement_ranked.csv --source all --min-topic-size 6 --min-quality 0")


if __name__ == "__main__":
    main()
