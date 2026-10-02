# REGULATORY RELEASE FETCHER  (Path D: SEC + FINRA primary sources)
#
# Pulls primary regulatory releases - the ground truth the news digest only
# reports on secondhand - into the same pipeline CSV schema, tagged
# SEARCH_TERM_ID = "SEC" or "FINRA", so clustering / discovery / status all work
# unchanged. These sources are public domain.
#
# Sources (listing page -> individual releases):
#   SEC press releases     : /newsroom/press-releases          (2026-NN-slug)
#   SEC speeches/statements: /newsroom/speeches-statements      (slug-MMDDYY)
#   FINRA news releases     : /media-center/newsreleases/YYYY/slug
#   FINRA notices           : /rules-guidance/notices/NN-NN
#
# Fetch policy: SEC's fair-access rules require a declared User-Agent and
# throttling; we set a UA with contact info and sleep between requests. Each
# release page is extracted to clean text via trafilatura. Failures degrade
# gracefully (skip the item) and never raise.
#
# Output: output/regulatory_news.csv (appended + de-duplicated by LINK).
#
# Usage:
#   python fetch_regulatory.py                      # all sources, recent items
#   python fetch_regulatory.py --source sec --limit 15
#   python fetch_regulatory.py --source finra --max-age-days 30

import argparse
import csv
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

try:
    import trafilatura
    _HAVE_TRAFILATURA = True
except Exception:
    _HAVE_TRAFILATURA = False

OUTPUT_CSV = Path("output/regulatory_news.csv")
SUMMARY_CHARS = 4000
REQUEST_TIMEOUT = 30
THROTTLE_SEC = 1.0       # polite delay between release fetches
MAX_RETRIES = 3          # attempts per URL before giving up
RETRY_BACKOFF = 2.0      # seconds, multiplied each retry (2, 4, 8...)
DEFAULT_MAX_AGE_DAYS = 45  # keep ~6 weeks of releases by default

HEADERS = {
    "User-Agent": "FINRA ERM research prototype (contact: erm-research@finra.org)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Encoding": "gzip, deflate",
}

PIPELINE_COLUMNS = [
    "RISK_ID", "SEARCH_TERM_ID", "GOOGLE_INDEX", "TITLE", "LINK",
    "PUBLISHED_DATE", "SUMMARY", "KEYWORDS", "SENTIMENT_COMPOUND",
    "SENTIMENT", "SOURCE", "CATEGORY", "QUALITY_SCORE",
]

LISTINGS = {
    "SEC": [
        ("https://www.sec.gov/newsroom/press-releases", "press release",
         re.compile(r'/newsroom/press-releases/20\d{2}-\d+[^"#?]*')),
        ("https://www.sec.gov/newsroom/speeches-statements", "statement",
         re.compile(r'/newsroom/speeches-statements/[a-z0-9][^"#?]*-\d{6}')),
    ],
    "FINRA": [
        ("https://www.finra.org/media-center/newsreleases", "news release",
         re.compile(r'/media-center/newsreleases/20\d{2}/[^"#?]+')),
        ("https://www.finra.org/rules-guidance/notices", "notice",
         re.compile(r'/rules-guidance/notices/(?:\d{2}-\d{2}|[a-z-]*notice-?\d+)[^"#?]*')),
    ],
    # Commodity Futures Trading Commission: numbered press releases, with
    # <time> tags on the listing. Relevant to prediction markets / event
    # contracts, which keep surfacing as an emerging risk.
    "CFTC": [
        ("https://www.cftc.gov/PressRoom/PressReleases", "press release",
         re.compile(r'/PressRoom/PressReleases/\d{4}-\d{2}')),
    ],
    # Consumer Financial Protection Bureau: newsroom with <time> tags. Relevant
    # as brokerages expand into consumer credit (e.g. Robinhood Gold).
    "CFPB": [
        ("https://www.consumerfinance.gov/about-us/newsroom/?categories=press-release",
         "press release",
         # Article slugs under /about-us/newsroom/ (no year segment); require a
         # multi-word slug so we skip the bare /newsroom/ and filter links.
         re.compile(r'/about-us/newsroom/[a-z0-9][a-z0-9-]{15,}/')),
    ],
    # North American Securities Administrators Association: state securities
    # regulators. WordPress site, absolute post URLs (/{postid}/{slug}/), no
    # <time> on the listing - date falls back to the post page. Strong investor-
    # protection signal (fraud trends, investor alerts) often ahead of federal.
    "NASAA": [
        ("https://www.nasaa.org/category/newsroom/current-headlines/", "news release",
         re.compile(r'https://www\.nasaa\.org/\d+/[a-z0-9][^"#?]*')),
    ],
}
BASE = {"SEC": "https://www.sec.gov", "FINRA": "https://www.finra.org",
        "CFTC": "https://www.cftc.gov", "CFPB": "https://www.consumerfinance.gov",
        "NASAA": "https://www.nasaa.org"}

# A <time datetime="YYYY-MM-DD..."> tag, used to date rows on listing pages.
_TIME_RE = re.compile(r'<time[^>]*datetime="(\d{4}-\d{2}-\d{2})')


def _rows_with_dates(html, link_pat):
    """Pair each release link in a listing page with the nearest preceding
    <time datetime> (listing pages render date then title in table rows).
    Returns [(href, date_or_empty), ...] in page order, de-duplicated."""
    # Tokenize the page into (position, kind, value) for time tags and links.
    events = []
    for m in _TIME_RE.finditer(html):
        events.append((m.start(), "date", m.group(1)))
    for m in link_pat.finditer(html):
        events.append((m.start(), "link", m.group(0)))
    events.sort(key=lambda e: e[0])
    out, seen, last_date = [], set(), ""
    for _, kind, val in events:
        if kind == "date":
            last_date = val
        else:
            if val in seen:
                continue
            seen.add(val)
            out.append((val, last_date))
    return out


def _get(url, session, retries=MAX_RETRIES):
    """GET with retry + exponential backoff. Retries transient failures
    (timeouts, connection errors, 429/5xx); gives up immediately on a hard 404.
    Returns (html, reason): html is "" on failure, and reason is one of
    "ok", "not_found", or "failed:<last reason>" so callers can report accurately."""
    delay = RETRY_BACKOFF
    reason = "failed"
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            code = r.status_code
            if code == 200 and r.text:
                # Decode with the page's actual charset so smart quotes/dashes
                # render correctly (SEC/FINRA pages are UTF-8 but requests can
                # mis-guess from headers).
                if r.apparent_encoding:
                    r.encoding = r.apparent_encoding
                return r.text, "ok"
            if code == 404:
                return "", "not_found"  # permanent; no point retrying
            reason = f"HTTP {code}"      # 403/429/5xx or empty body: retryable
        except requests.RequestException as e:
            reason = type(e).__name__   # Timeout, ConnectionError, etc.
        if attempt < retries:
            time.sleep(delay)
            delay *= 2
    return "", f"failed:{reason}"


def _enumerate(agency, session, per_source_limit):
    """Return [(url, title_hint, category, list_date), ...] for an agency,
    with the publish date read from the listing page."""
    out, seen = [], set()
    for listing_url, category, pat in LISTINGS[agency]:
        html, _reason = _get(listing_url, session)
        if not html:
            print(f"  WARN: could not load listing {listing_url} ({_reason})")
            continue
        n = 0
        for h, date in _rows_with_dates(html, pat):
            full = BASE[agency] + h if h.startswith("/") else h
            if full in seen:
                continue
            seen.add(full)
            slug = h.rstrip("/").split("/")[-1]
            hint = re.sub(r"^20\d{2}-\d+-", "", slug).replace("-", " ").strip()
            out.append((full, hint, category, date))
            n += 1
            if n >= per_source_limit:
                break
        time.sleep(THROTTLE_SEC)
    return out


_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)
_H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.S | re.I)
_DATE_RES = [
    re.compile(r'"datePublished"\s*:\s*"(\d{4}-\d{2}-\d{2})'),
    re.compile(r'\b(20\d{2}-\d{2}-\d{2})\b'),
    re.compile(r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},\s+20\d{2})\b'),
]


import html as _htmlmod

# Common mojibake: UTF-8 bytes misread as Windows-1252/Latin-1. Map the ones
# that show up in SEC/FINRA titles (smart quotes, dashes, (R)) back to ASCII-ish.
_MOJIBAKE = {
    "\u00c6": "'", "\u00d5": "'", "\u00d4": "'",   # curly apostrophes
    "\u00d2": '"', "\u00d3": '"',                      # curly quotes
    "\u00f9": "\u2014", "\u00d1": "\u2013",         # em / en dash
    "\u00ab": "\u00ae",                                 # registered mark
    "\u0089": "", "\u0092": "'", "\u0093": '"', "\u0094": '"', "\u0096": "-",
}


def _fix_mojibake(t):
    if not t:
        return t
    # First try the principled fix: re-encode a bad latin-1 decode as utf-8.
    try:
        if any(ch in t for ch in _MOJIBAKE):
            fixed = t.encode("latin-1", "ignore").decode("utf-8", "ignore")
            if fixed and "\ufffd" not in fixed:
                t = fixed
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    for bad, good in _MOJIBAKE.items():
        t = t.replace(bad, good)
    return t


def _clean(t):
    t = _htmlmod.unescape(t or "")
    t = _fix_mojibake(t)
    return " ".join(t.split()).strip()


def _extract_title(html, fallback):
    m = _H1_RE.search(html) or _TITLE_RE.search(html)
    def _from(match):
        if not match:
            return ""
        t = _clean(re.sub(r"<[^>]+>", "", match.group(1)))
        # strip a trailing site-name suffix after a pipe/dash
        t = re.split(r"\s*[|\u2013\u2014]\s*(?:SEC\.gov|FINRA\.org|CFTC|"
                     r"Consumer Financial Protection Bureau|NASAA|U\.S\. Securities)", t)[0]
        return t.strip()

    h1 = _from(_H1_RE.search(html))
    title_tag = _from(_TITLE_RE.search(html))
    # CFTC <h1> is just "Release Number NNNN-YY"; prefer the <title> headline.
    if h1 and not re.match(r"(?i)release number\s", h1):
        return h1
    if title_tag:
        return title_tag
    if h1:
        return h1
    return _clean(fallback).title()


def _extract_date(html):
    for rx in _DATE_RES:
        m = rx.search(html)
        if m:
            s = m.group(1)
            for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%b. %d, %Y"):
                try:
                    return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
                except ValueError:
                    continue
    return ""


def fetch(agencies, per_source_limit, max_age_days):
    if not _HAVE_TRAFILATURA:
        print("trafilatura not installed. Run: pip install trafilatura")
        return []
    session = requests.Session()
    cutoff = (datetime.now() - timedelta(days=max_age_days)).strftime("%Y-%m-%d") if max_age_days else None

    rows = []
    tally = {"ok": 0, "failed": 0, "not_found": 0, "too_old": 0, "thin": 0}
    for agency in agencies:
        print(f"\n{agency}: enumerating releases...")
        links = _enumerate(agency, session, per_source_limit)
        print(f"  found {len(links)} candidate releases; fetching"
              f"{' (cutoff ' + cutoff + ')' if cutoff else ''}...")
        for url, hint, category, list_date in links:
            html, reason = _get(url, session)  # retries internally
            time.sleep(THROTTLE_SEC)
            if not html:
                if reason == "not_found":
                    tally["not_found"] += 1
                    print(f"    404 (skipped): {url[:78]}")
                else:
                    tally["failed"] += 1
                    print(f"    FAILED ({reason}) after {MAX_RETRIES} tries: {url[:70]}")
                continue
            try:
                text = trafilatura.extract(html, include_comments=False,
                                           include_tables=False, favor_precision=True) or ""
            except Exception:
                text = ""
            text = _clean(text)
            title = _extract_title(html, hint)
            # Prefer the listing-page date; fall back to a date parsed from the
            # release body (FINRA notices carry it; SEC pages don't reliably).
            date = list_date or _extract_date(html)
            if cutoff and date and date < cutoff:
                tally["too_old"] += 1
                continue
            if len(text) < 150:
                tally["thin"] += 1
                print(f"    thin extract ({len(text)}c): {title[:58]}")
                continue
            rows.append({
                "agency": agency, "title": title, "link": url,
                "date": date, "text": text[:SUMMARY_CHARS], "category": category,
            })
            tally["ok"] += 1
            print(f"    ok [{date or 'n/a':10}] {title[:66]}")

    print(f"\nFetch tally: {tally['ok']} ok · {tally['too_old']} older than cutoff · "
          f"{tally['thin']} thin · {tally['not_found']} not-found (404) · "
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
            "SEARCH_TERM_ID": it["agency"],       # "SEC" or "FINRA"
            "GOOGLE_INDEX": i,
            "TITLE": it["title"],
            "LINK": it["link"],
            "PUBLISHED_DATE": it["date"],
            "SUMMARY": it["text"],
            "KEYWORDS": "",
            "SENTIMENT_COMPOUND": comp,
            "SENTIMENT": label,
            "SOURCE": it["agency"],
            "CATEGORY": f"reg:{it['agency'].lower()}:{it['category'].replace(' ', '_')}",
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
    p = argparse.ArgumentParser(description="Fetch SEC/FINRA/CFTC/CFPB/NASAA primary releases into the pipeline schema.")
    p.add_argument("--source", choices=["sec", "finra", "cftc", "cfpb", "nasaa", "all"],
                   default="all")
    p.add_argument("--limit", type=int, default=20, dest="per_source_limit",
                   help="Max releases per listing page (default 20).")
    p.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS, dest="max_age_days",
                   help=f"Only keep releases newer than N days (default {DEFAULT_MAX_AGE_DAYS}; "
                        "pass 0 for no age filter).")
    args = p.parse_args()

    agencies = {
        "sec": ["SEC"], "finra": ["FINRA"], "cftc": ["CFTC"],
        "cfpb": ["CFPB"], "nasaa": ["NASAA"],
        "all": ["SEC", "FINRA", "CFTC", "CFPB", "NASAA"],
    }[args.source]
    print("#" * 60)
    print("REGULATORY RELEASE FETCH (SEC + FINRA primary sources)")
    print(f"Sources: {agencies} | per-source limit: {args.per_source_limit}"
          f"{' | max age ' + str(args.max_age_days) + 'd' if args.max_age_days else ' | no age filter'}")
    print("#" * 60)

    items = fetch(agencies, args.per_source_limit, args.max_age_days)
    if not items:
        print("\nNo releases fetched.")
        return
    rows = to_pipeline_rows(items)
    added, total, deduped = save(rows)
    print(f"\nFetched {added} releases -> {OUTPUT_CSV} "
          f"({total} total after de-dup, {deduped} duplicates dropped).")
    print("Cluster with: python discover_score.py --input output/regulatory_news.csv "
          "--out output/regulatory_ranked.csv --source all")


if __name__ == "__main__":
    main()
