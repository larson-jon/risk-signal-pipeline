# SEC ENFORCEMENT FETCHER  (Path E: SEC enforcement actions)
#
# Pulls the SEC's two enforcement streams - the actual case actions, distinct
# from the agency's promotional press releases that fetch_regulatory.py already
# covers - into the pipeline CSV schema, tagged SEARCH_TERM_ID = "SEC_ENFORCEMENT",
# so clustering / discovery / status all work unchanged. These are public domain.
#
# Two streams (both from paginated HTML listings with a <time datetime> per row
# and a respondents cell whose <a> text is the party name - a natural title):
#
#   Litigation Releases      civil suits SEC files in federal court. Each row
#   (/enforcement-litigation/   links to a clean HTML detail page
#    litigation-releases)       (.../litigation-releases/lr-NNNNN); trafilatura
#                               extracts the summary.
#
#   Administrative Proceedings SEC-adjudicated orders (settled + litigated).
#   (/enforcement-litigation/   Each row links DIRECTLY to a PDF order
#    administrative-proceedings) (/files/litigation/admin/YYYY/<rel>.pdf);
#                               pypdf extracts the order text.
#
# Relevance note: SEC enforcement spans on-mission broker-dealer / AML / Reg BI /
# market-manipulation actions (overlapping what FINRA polices) and far-afield
# matters (issuer accounting fraud, FCPA, crypto issuers). This fetcher pulls it
# all; the mission-relevance scoring in discover_score.py sorts signal from noise.
#
# Fetch policy mirrors fetch_regulatory.py: declared contact User-Agent, throttle,
# retry/backoff. Failures degrade gracefully (skip the item) and never raise.
#
# Output: output/sec_enforcement_news.csv (appended + de-duplicated by LINK).
#
# Usage:
#   python fetch_sec_enforcement.py                       # both streams, recent
#   python fetch_sec_enforcement.py --stream litigation --limit 30
#   python fetch_sec_enforcement.py --stream admin --max-age-days 30

import argparse
import csv
import io
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

# Reuse the hardened helpers already proven in fetch_regulatory.py: polite GET
# with retry/backoff, text cleanup + mojibake repair, and the header/throttle
# constants. Keeps SEC fetch behaviour identical across modules.
from fetch_regulatory import (
    HEADERS, THROTTLE_SEC, MAX_RETRIES, PIPELINE_COLUMNS,
    _get, _clean,
)

try:
    import trafilatura
    _HAVE_TRAFILATURA = True
except Exception:
    _HAVE_TRAFILATURA = False

try:
    from pypdf import PdfReader
    _HAVE_PYPDF = True
except Exception:
    _HAVE_PYPDF = False

OUTPUT_CSV = Path("output/sec_enforcement_news.csv")
SUMMARY_CHARS = 4000
REQUEST_TIMEOUT = 40
DEFAULT_MAX_AGE_DAYS = 45
BASE = "https://www.sec.gov"

# One stream = (listing_url, category, is_pdf). The listing rows share a layout:
# a <time datetime> cell followed by a respondents cell <a href=...>Name</a>.
STREAMS = {
    "litigation": (
        f"{BASE}/enforcement-litigation/litigation-releases",
        "litigation_release", False,
    ),
    "admin": (
        f"{BASE}/enforcement-litigation/administrative-proceedings",
        "administrative_proceeding", True,
    ),
}

# A listing-table row: capture the row's <time datetime> and the respondents
# cell's link href + inner text (the party name). The DOM places the date cell
# before the respondents cell within the same <tr>, so we pair them positionally.
_ROW_TIME_RE = re.compile(r'<time[^>]*datetime="(\d{4}-\d{2}-\d{2})')
_ROW_LINK_RE = re.compile(
    r"release-view__respondents'><a href=['\"]([^'\"#?]+)['\"][^>]*>(.*?)</a>",
    re.S,
)


def _rows(html):
    """Pair each respondents-cell link with the nearest preceding <time> date,
    in page order. Returns [(href, name, date), ...] de-duplicated by href."""
    events = []
    for m in _ROW_TIME_RE.finditer(html):
        events.append((m.start(), "date", m.group(1)))
    for m in _ROW_LINK_RE.finditer(html):
        name = _clean(re.sub(r"<[^>]+>", " ", m.group(2)))
        events.append((m.start(), "link", (m.group(1), name)))
    events.sort(key=lambda e: e[0])
    out, seen, last_date = [], set(), ""
    for _, kind, val in events:
        if kind == "date":
            last_date = val
        else:
            href, name = val
            if href in seen:
                continue
            seen.add(href)
            out.append((href, name, last_date))
    return out


_LR_TITLE_RE = re.compile(r"SEC\s+(?:Charges|Obtains|Files|Settles|Announces)[^.\n]{5,160}", re.I)

# --- Boilerplate strippers -------------------------------------------------
# SEC enforcement documents open with heavy, near-identical legal scaffolding
# (caption, release numbers, "ORDER INSTITUTING..." titles, litigation-release
# headers). Embedding the raw text makes every action look alike and clustering
# keys on that shared form instead of the misconduct. We cut the preamble so the
# embedded text starts at the operative facts. Title extraction happens on the
# RAW body first, so stripping never costs us the SEC's own headline.

# AP: the operative narrative begins at the first Roman-numeral section marker
# ("I.") that follows the "ORDER INSTITUTING ... / ORDER ..." title. Everything
# before it (UNITED STATES OF AMERICA ... Release No. ... In the Matter of ...
# Respondents. ORDER ...) is caption + title boilerplate shared by every order.
# SEC order structure: section I is a procedural recital ("deems it appropriate
# ... proceedings be instituted"), II is "On the basis of this Order, the
# Commission finds that:" and the actual findings/facts begin at section III.
# Cutting to "III." lands on the misconduct narrative (respondent background,
# what they did) - the part that actually distinguishes one action from another.
_AP_FINDINGS_RE = re.compile(r"\bIII\.\s")
# If there's no "III." (shorter orders, Fair-Fund notices), fall back to cutting
# the caption/title block up to the first "I." section marker.
_AP_PREAMBLE_RE = re.compile(
    r"^.*?\bORDER\b.*?(?:\bI\.\s)",   # up to and incl. the ORDER title, to "I. "
    re.S,
)
# Last-resort cut points.
_AP_FALLBACKS = (
    "deems it appropriate",   # "The Commission deems it appropriate..." opener
    "In the Matter of",
)

# LR: operative narrative begins at "On <date>, the SEC/Commission ...". The
# preamble is "<name> U.S. SECURITIES AND EXCHANGE COMMISSION Litigation Release
# No. ... SEC v. ... filed ... <HEADLINE>". We keep the headline + narrative.
_LR_NARRATIVE_RE = re.compile(r"\bOn\s+(?:[A-Z][a-z]+\.?\s+\d{1,2},\s+20\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b")
_LR_CAPTION_RE = re.compile(
    r"^.*?Litigation Release No\.\s*\d+\s*/\s*[^\n]*?\d{4}\s*",   # thru the LR header line
    re.S,
)
_LR_VCAPTION_RE = re.compile(   # the "SEC v. ... filed ..." case caption line
    r"(?:Securities and Exchange Commission|SEC)\s+v\.\s+.*?\bfiled\b[^)]*\)\s*",
    re.S,
)


def _strip_ap_boilerplate(text):
    # Preferred: jump to section III (findings/facts) - the misconduct narrative.
    m = _AP_FINDINGS_RE.search(text)
    if m and len(text) - m.end() > 300:
        return text[m.end():].strip()
    # Fallback: drop the caption + ORDER title up to section "I.".
    cut = _AP_PREAMBLE_RE.match(text)
    if cut and len(text) - cut.end() > 200:
        return text[cut.end():].strip()
    for marker in _AP_FALLBACKS:
        i = text.find(marker)
        if i != -1 and len(text) - i > 200:
            return text[i:].strip()
    return text


def _strip_lr_boilerplate(text):
    # Preserve the headline: drop the "<name> ... Litigation Release No N / date"
    # caption and the "SEC v. ... filed ..." case line, leaving headline + body.
    out = _LR_CAPTION_RE.sub("", text, count=1)
    out = _LR_VCAPTION_RE.sub("", out, count=1)
    return out.strip() or text


# Pervasive legal register shared across most enforcement documents (measured
# across the corpus: each appears in 24-50% of actions). Left in, it dominates
# the embedding and collapses every action into one "legal document" cluster.
# We delete these phrases from the embedded text so the vectors key on the
# distinguishing facts (who, what product, what scheme) instead of court form.
# Order matters: longer phrases first so we don't leave fragments behind.
_LEGALESE = [
    "the findings herein are made pursuant to",
    "on the basis of this order and",
    "on the basis of this order",
    "the securities and exchange commission",
    "securities and exchange commission",
    "united states district court",
    "the commission deems it appropriate",
    "in the public interest",
    "without admitting or denying",
    "offers of settlement", "offer of settlement",
    "cease-and-desist", "cease and desist",
    "securities exchange act of 1934", "exchange act of 1934",
    "securities act of 1933",
    "administrative proceeding",
    "pursuant to section", "pursuant to rule",
    "the commission finds", "the commission further finds",
    "it is hereby ordered",
    "remedial sanctions",
    "final consent judgment", "consent judgment", "final judgment",
    "the complaint alleges", "the sec's complaint",
    "prejudgment interest", "civil penalty", "civil penalties",
    "permanently enjoin", "permanently enjoining",
    "disgorgement",
    "respondents", "respondent",
    "defendants", "defendant",
]
_LEGALESE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in _LEGALESE) + r")\b",
    re.I,
)


def _scrub_legalese(text):
    """Remove shared legal-register phrases so clustering keys on facts, not
    court form. Collapses the whitespace the deletions leave behind."""
    out = _LEGALESE_RE.sub(" ", text)
    out = re.sub(r"\s{2,}", " ", out)
    # Drop orphaned punctuation left where phrases were (" , ", " . .").
    out = re.sub(r"\s+([,.;:])", r"\1", out)
    return out.strip()


def _extract_html(html):
    """Extract the readable body of an LR detail page via trafilatura."""
    try:
        text = trafilatura.extract(html, include_comments=False,
                                   include_tables=False, favor_precision=True) or ""
    except Exception:
        text = ""
    return _clean(text)


# pypdf can mis-decode the font-embedded smart punctuation in SEC order PDFs,
# surfacing curly quotes/dashes as stray latin-1 glyphs (o-circumflex, o-umlaut,
# etc.). _fix_mojibake (tuned for HTML) does not catch these, so normalise the
# PDF-origin set to plain ASCII before cleanup.
_PDF_PUNCT = {
    "\u00f4": '"', "\u00f6": '"',   # curly double quotes (open/close)
    "\u00d5": "'", "\u00d4": "'",   # curly single quotes
    "\u0092": "'", "\u0093": '"', "\u0094": '"',
    "\u00f1": "-", "\u00d1": "-",   # en/em dash variants
    "\u00ae": "(R)",
}


def _extract_pdf(content):
    """Extract text from an AP order PDF (first ~8 pages is plenty for the
    operative facts; orders run long with boilerplate after that)."""
    try:
        reader = PdfReader(io.BytesIO(content))
        pages = reader.pages[:8]
        raw = "\n".join((p.extract_text() or "") for p in pages)
        for bad, good in _PDF_PUNCT.items():
            raw = raw.replace(bad, good)
        return _clean(raw)
    except Exception:
        return ""


def _title_for(name, body, is_pdf):
    """Prefer the SEC's own headline when present (LR bodies carry a 'SEC Charges
    ...' line); otherwise fall back to the respondent name from the listing."""
    if not is_pdf and body:
        m = _LR_TITLE_RE.search(body)
        if m:
            return _clean(m.group(0))
    return name or "SEC enforcement action"


def enumerate_stream(stream_key, session, limit):
    """Return [(url, name, date, category, is_pdf), ...] for one stream."""
    listing_url, category, is_pdf = STREAMS[stream_key]
    html, reason = _get(listing_url, session)
    if not html:
        print(f"  WARN: could not load listing {listing_url} ({reason})")
        return []
    out = []
    for href, name, date in _rows(html):
        url = href if href.startswith("http") else BASE + href
        out.append((url, name, date, category, is_pdf))
        if len(out) >= limit:
            break
    return out


def _get_bytes(url, session, retries=MAX_RETRIES):
    """GET returning raw bytes, with the same retry/backoff policy as
    fetch_regulatory._get (which returns text only). Used for AP order PDFs.
    Returns (content, reason) where reason is 'ok' / 'not_found' / 'failed:...'."""
    delay = 2.0
    reason = "failed"
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            code = r.status_code
            if code == 200 and r.content:
                return r.content, "ok"
            if code == 404:
                return b"", "not_found"
            reason = f"HTTP {code}"
        except requests.RequestException as e:
            reason = type(e).__name__
        if attempt < retries:
            time.sleep(delay)
            delay *= 2
    return b"", f"failed:{reason}"


def fetch(streams, limit, max_age_days):
    if not _HAVE_TRAFILATURA:
        print("trafilatura not installed. Run: pip install trafilatura")
        return []
    if not _HAVE_PYPDF:
        print("pypdf not installed. Run: pip install pypdf")
        return []
    session = requests.Session()
    cutoff = ((datetime.now() - timedelta(days=max_age_days)).strftime("%Y-%m-%d")
              if max_age_days else None)

    rows = []
    tally = {"ok": 0, "failed": 0, "not_found": 0, "too_old": 0, "thin": 0}
    for stream_key in streams:
        print(f"\n{stream_key}: enumerating actions...")
        items = enumerate_stream(stream_key, session, limit)
        print(f"  found {len(items)} candidate action(s); fetching"
              f"{' (cutoff ' + cutoff + ')' if cutoff else ''}...")
        for url, name, date, category, is_pdf in items:
            if cutoff and date and date < cutoff:
                tally["too_old"] += 1
                continue
            # AP orders are PDFs (need raw bytes); LR detail pages are HTML.
            if is_pdf:
                payload, reason = _get_bytes(url, session)
            else:
                payload, reason = _get(url, session)
            time.sleep(THROTTLE_SEC)
            if not payload:
                if reason == "not_found":
                    tally["not_found"] += 1
                    print(f"    404 (skipped): {url[:78]}")
                else:
                    tally["failed"] += 1
                    print(f"    FAILED ({reason}) after {MAX_RETRIES} tries: {url[:66]}")
                continue
            raw = _extract_pdf(payload) if is_pdf else _extract_html(payload)
            if len(raw) < 150:
                tally["thin"] += 1
                print(f"    thin extract ({len(raw)}c): {name[:54]}")
                continue
            # Title from the RAW body (keeps the SEC headline / caption); then
            # strip the legal preamble so the stored/embedded text starts at the
            # operative facts and clustering keys on misconduct, not boilerplate.
            title = _title_for(name, raw, is_pdf)
            body = _strip_ap_boilerplate(raw) if is_pdf else _strip_lr_boilerplate(raw)
            body = _scrub_legalese(body)
            rows.append({
                "title": title, "respondent": name, "link": url, "date": date,
                "text": body[:SUMMARY_CHARS], "category": category,
            })
            tally["ok"] += 1
            print(f"    ok [{date or 'n/a':10}] {title[:62]}")

    print(f"\nFetch tally: {tally['ok']} ok | {tally['too_old']} older than cutoff | "
          f"{tally['thin']} thin | {tally['not_found']} not-found (404) | "
          f"{tally['failed']} failed after retries")
    return rows


def to_pipeline_rows(items):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    try:
        from content_quality import quality_score
    except Exception:
        quality_score = None
    out = []
    for i, it in enumerate(items):
        blob = f"{it['title']} {it['text']}"
        comp = analyzer.polarity_scores(blob)["compound"]
        label = ("Positive" if comp >= 0.05 else "Negative" if comp <= -0.05 else "Neutral")
        row = {
            "RISK_ID": -1,
            "SEARCH_TERM_ID": "SEC_ENFORCEMENT",
            "GOOGLE_INDEX": i,
            "TITLE": it["title"],
            "LINK": it["link"],
            "PUBLISHED_DATE": it["date"],
            "SUMMARY": it["text"],
            "KEYWORDS": "",
            "SENTIMENT_COMPOUND": comp,
            "SENTIMENT": label,
            "SOURCE": "SEC Enforcement",
            "CATEGORY": f"sec_enforcement:{it['category']}",
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
    combined = combined.drop_duplicates(subset="LINK", keep="first")
    combined["GOOGLE_INDEX"] = range(len(combined))
    combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
    return len(new_df), len(combined), before - len(combined)


def main():
    p = argparse.ArgumentParser(
        description="Fetch SEC enforcement actions (litigation releases + "
                    "administrative proceedings) into the pipeline schema.")
    p.add_argument("--stream", choices=["litigation", "admin", "all"], default="all")
    p.add_argument("--limit", type=int, default=30,
                   help="Max actions per stream listing (default 30).")
    p.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS,
                   dest="max_age_days",
                   help=f"Only keep actions newer than N days (default "
                        f"{DEFAULT_MAX_AGE_DAYS}; pass 0 for no age filter).")
    args = p.parse_args()

    streams = {"litigation": ["litigation"], "admin": ["admin"],
               "all": ["litigation", "admin"]}[args.stream]
    print("#" * 60)
    print("SEC ENFORCEMENT FETCH (litigation releases + admin proceedings)")
    print(f"Streams: {streams} | per-stream limit: {args.limit}"
          f"{' | max age ' + str(args.max_age_days) + 'd' if args.max_age_days else ' | no age filter'}")
    print("#" * 60)

    items = fetch(streams, args.limit, args.max_age_days)
    if not items:
        print("\nNo actions fetched.")
        return
    rows = to_pipeline_rows(items)
    added, total, deduped = save(rows)
    print(f"\nFetched {added} action(s) -> {OUTPUT_CSV} "
          f"({total} total after de-dup, {deduped} duplicates dropped).")
    print("Cluster with: python discover_score.py --input output/sec_enforcement_news.csv "
          "--out output/sec_enforcement_ranked.csv --source all --min-topic-size 6 --min-quality 0")


if __name__ == "__main__":
    main()
