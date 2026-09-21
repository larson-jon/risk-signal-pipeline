# RISK DISCOVERY FETCH  (Path B)
#
# The rest of the pipeline only ever sees news that matched an existing risk's
# search terms, so it can't surface risks nobody searched for. This script
# fetches a BROAD, UN-KEYWORDED stream of news (by category, from top-tier
# domains) over a multi-month window, so the discovery-scoring stage can find
# coherent, significant themes that sit FAR from every existing risk - i.e.
# candidate risks outside the current taxonomy.
#
# Source: newsdata.io /api/1/archive (historical; from_date/to_date). No `q`.
#   - 5 credits per request; 50 articles/credit on paid plans.
#   - Basic plan reaches ~6 months back (4-month default is within that).
#   - category / country / language / prioritydomain supported; failed or
#     zero-result requests are NOT charged.
#
# Output: output/discovery_news.csv in the SAME schema the scraper writes, so
# it flows straight into the clustering pipeline. Discovery articles carry
# RISK_ID = -1 and SEARCH_TERM_ID = "DISCOVERY" (they matched no risk).
#
# Usage:
#   python discover_fetch.py                       # last 4 months, default cats
#   python discover_fetch.py --months 4 --categories business,politics
#   python discover_fetch.py --dry-run             # no API calls; prints plan
#   python discover_fetch.py --max-requests 200    # cap credit spend

import argparse
import csv
import datetime as dt
import os
import random
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from dateutil import parser as dateparser

# Load NEWS_DATA_API_KEY (and any other vars) from a local .env if present.
# Safe no-op if python-dotenv isn't installed or no .env exists.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ---------------------------------------------------------------------------
# SSL setup - identical policy to news_sentiment_scraper.py.
# ---------------------------------------------------------------------------
def setup_ssl_verification():
    cert_path = Path(__file__).parent / "combined-certs.pem"
    if os.getenv("GITHUB_ACTIONS"):
        print("Running in GitHub Actions - using default SSL verification")
        return True
    if cert_path.exists():
        print(f"Using corporate certificate: {cert_path}")
        return str(cert_path)
    print("No certificate found - SSL verification uses defaults")
    return True


DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
VERIFY_SSL = setup_ssl_verification()
BASE_URL = "https://newsdata.io/api/1/archive"
LATEST_URL = "https://newsdata.io/api/1/latest"

# Broad categories that capture finance/regulatory-adjacent risk signal without
# naming any specific risk. newsdata.io supports up to 5 categories (Free/Basic).
DEFAULT_CATEGORIES = ["business", "politics", "technology", "world"]

# Risk-LEXICON queries: the *language of risk itself*, so we catch risk-shaped
# stories regardless of subject - without pre-naming any topic. Phrases are
# favored over bare words ("regulators warn" >> "risk") because single risk
# words appear constantly in benign contexts; phrases are far higher-precision.
# Each entry is one newsdata.io `q` search. Tuned toward finance/regulatory
# emergence. `q` is capped at 100 chars on the free plan - all are well under.
RISK_LEXICON = [
    # emergence / novelty framing
    "emerging risk", "systemic risk", "unprecedented", "first-of-its-kind",
    "novel threat", "unintended consequences", "regulatory loophole",
    "unregulated", "regulatory gap",
    # authority raising an alarm
    "regulators warn", "watchdog warns", "SEC warns", "warns of risk",
    "flagged as a risk", "under scrutiny", "calls for regulation",
    # trouble / disruption framing
    "growing threat", "mounting concern", "market disruption",
    "vulnerability exposed", "systemic threat",
]

OUTPUT_CSV = "output/discovery_news.csv"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36",
]


def get_source_name(url):
    domain = urlparse(url).netloc.replace("www.", "")
    parts = domain.split(".")
    return parts[-2] if len(parts) > 2 else domain


def _date_chunks(start, end, chunk_days):
    """Yield (from_date, to_date) windows covering [start, end] in chunks.

    The archive endpoint returns newest-first within a window and paginates,
    but chunking by date keeps each window's result set bounded and makes the
    crawl resumable/observable.
    """
    cur = start
    step = dt.timedelta(days=chunk_days)
    while cur <= end:
        chunk_end = min(cur + step - dt.timedelta(days=1), end)
        yield cur, chunk_end
        cur = chunk_end + dt.timedelta(days=1)


def _score_sentiment(analyzer, text):
    scores = analyzer.polarity_scores(text)
    c = scores["compound"]
    label = "Positive" if c >= 0.05 else "Negative" if c <= -0.05 else "Neutral"
    return round(c, 4), label


# newspaper3k is optional: it enriches summaries via full-text parsing, but if
# it isn't installed we fall back to the API's own title/description, which is
# enough for clustering. (newspaper3k can be hard to build on newer Python.)
try:
    from newspaper import Article, Config as _NewspaperConfig
    _HAVE_NEWSPAPER = True
except Exception:
    _HAVE_NEWSPAPER = False


def _make_config():
    if not _HAVE_NEWSPAPER:
        return None
    cfg = _NewspaperConfig()
    cfg.fetch_images = False
    cfg.memoize_articles = False
    cfg.request_timeout = 30
    return cfg


def _parse_article(url, config):
    """Best-effort full-text parse; returns (summary, keywords).

    No-op (returns empty) when newspaper3k is unavailable - the caller then
    falls back to the API-provided description.
    """
    if not _HAVE_NEWSPAPER or config is None:
        return "", ""
    try:
        article = Article(url, config=config)
        article.download()
        if article.download_state == 2:
            article.parse()
            summary = article.summary if article.summary else (article.text[:500] if article.text else "")
            keywords = ", ".join(article.keywords) if article.keywords else ""
            return summary, keywords
    except Exception:
        pass
    return "", ""


def fetch(months, categories, country, language, prioritydomain,
          chunk_days, max_requests, per_window_pages, dry_run):
    api_key = os.getenv("NEWS_DATA_API_KEY")
    today = dt.date.today()
    start = today - dt.timedelta(days=int(months * 30.44))

    print("#" * 60)
    print("RISK DISCOVERY FETCH (broad, un-keyworded archive)")
    print(f"  window     : {start} -> {today}  (~{months} months)")
    print(f"  categories : {categories}")
    print(f"  country    : {country} | language: {language} | prioritydomain: {prioritydomain}")
    print(f"  chunking   : {chunk_days}-day windows, up to {per_window_pages} page(s) each")
    print(f"  max requests: {max_requests}  (archive = 5 credits each)")
    print("#" * 60)

    # Plan the crawl (chunks x categories x pages) so credit spend is visible.
    windows = list(_date_chunks(start, today, chunk_days))
    planned = len(windows) * len(categories) * per_window_pages
    est_credits = planned * 5
    print(f"Planned up to {planned} requests across {len(windows)} date windows "
          f"x {len(categories)} categories x {per_window_pages} pages "
          f"(~{est_credits} credits max).")

    if dry_run:
        print("\n--dry-run: no API calls made. Sample request that WOULD be issued:")
        w0 = windows[0] if windows else (start, today)
        sample = {
            "apikey": "YOUR_KEY", "category": categories[0], "country": country,
            "language": language, "prioritydomain": prioritydomain,
            "from_date": w0[0].isoformat(), "to_date": w0[1].isoformat(),
        }
        print(f"  GET {BASE_URL}?{'&'.join(f'{k}={v}' for k, v in sample.items())}")
        return

    if not api_key:
        print("ERROR: NEWS_DATA_API_KEY not set. Set it and re-run (or use --dry-run).")
        sys.exit(1)

    from newspaper import Config
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    try:
        from content_quality import quality_score
    except Exception:
        quality_score = None

    analyzer = SentimentIntensityAnalyzer()
    config = Config()
    config.fetch_images = False
    config.memoize_articles = False
    config.request_timeout = 30
    if VERIFY_SSL and VERIFY_SSL is not True:
        os.environ["REQUESTS_CA_BUNDLE"] = VERIFY_SSL
        os.environ["SSL_CERT_FILE"] = VERIFY_SSL

    seen_links = set()
    all_rows = []
    requests_made = 0
    global_index = 0

    for (w_start, w_end) in windows:
        for category in categories:
            if requests_made >= max_requests:
                print(f"Hit max-requests cap ({max_requests}); stopping crawl.")
                break
            next_page = None
            for _page in range(per_window_pages):
                if requests_made >= max_requests:
                    break
                params = {
                    "apikey": api_key,
                    "category": category,
                    "country": country,
                    "language": language,
                    "from_date": w_start.isoformat(),
                    "to_date": w_end.isoformat(),
                }
                if prioritydomain:
                    params["prioritydomain"] = prioritydomain
                if next_page:
                    params["page"] = next_page

                try:
                    resp = requests.get(BASE_URL, params=params, timeout=30, verify=VERIFY_SSL)
                    requests_made += 1
                except Exception as e:
                    print(f"  request error ({category} {w_start}): {e}")
                    break

                if resp.status_code != 200:
                    print(f"  api {resp.status_code} ({category} {w_start}..{w_end}): "
                          f"{resp.text[:160]}")
                    break

                data = resp.json()
                results = data.get("results", []) or []
                if DEBUG_MODE:
                    print(f"  {category} {w_start}..{w_end} p{_page+1}: {len(results)} articles")

                for item in results:
                    url = item.get("link") or ""
                    if not url or url.lower().strip() in seen_links:
                        continue
                    seen_links.add(url.lower().strip())
                    title = item.get("title") or ""
                    published = item.get("pubDate", "") or ""
                    source = item.get("source_id") or get_source_name(url)

                    summary, keywords = _parse_article(url, config)
                    compound, label = _score_sentiment(analyzer, f"{title} {summary}")
                    global_index += 1
                    row = {
                        "RISK_ID": -1,
                        "SEARCH_TERM_ID": "DISCOVERY",
                        "GOOGLE_INDEX": global_index,
                        "TITLE": title,
                        "LINK": url,
                        "PUBLISHED_DATE": published,
                        "SUMMARY": summary[:500],
                        "KEYWORDS": keywords,
                        "SENTIMENT_COMPOUND": compound,
                        "SENTIMENT": label,
                        "SOURCE": source,
                        "CATEGORY": category,
                        "QUALITY_SCORE": 0,
                    }
                    if quality_score:
                        try:
                            row["QUALITY_SCORE"] = quality_score(row)
                        except Exception:
                            pass
                    all_rows.append(row)

                next_page = data.get("nextPage")
                if not next_page:
                    break
                time.sleep(1)
        if requests_made >= max_requests:
            break

    print(f"\nFetched {len(all_rows)} unique articles in {requests_made} requests "
          f"(~{requests_made * 5} credits).")
    _save(all_rows)


def _save(rows):
    out = Path(OUTPUT_CSV)
    out.parent.mkdir(exist_ok=True)
    columns = ["RISK_ID", "SEARCH_TERM_ID", "GOOGLE_INDEX", "TITLE", "LINK",
               "PUBLISHED_DATE", "SUMMARY", "KEYWORDS", "SENTIMENT_COMPOUND",
               "SENTIMENT", "SOURCE", "CATEGORY", "QUALITY_SCORE"]
    new_df = pd.DataFrame(rows, columns=columns)

    if out.exists():
        existing = pd.read_csv(out)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["LINK"], keep="first")
    else:
        combined = new_df

    if combined.empty:
        print("No articles to save.")
        return
    combined.to_csv(out, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
    print(f"Wrote {len(combined)} total discovery articles -> {out}")


def fetch_latest(categories, country, language, prioritydomain, per_category_pages):
    """Free-tier discovery: broad un-keyworded fetch from /latest (past 48h).

    The archive endpoint is paid-only, so on the free plan we pull the last
    48 hours by category (no `q`) and accumulate over repeated runs - the same
    forward-accumulation pattern the risk scraper already uses.
    """
    api_key = os.getenv("NEWS_DATA_API_KEY")
    if not api_key:
        print("ERROR: NEWS_DATA_API_KEY not set (put it in .env or the environment).")
        sys.exit(1)

    print("#" * 60)
    print("RISK DISCOVERY FETCH (broad, un-keyworded, /latest 48h)")
    print(f"  categories : {categories}")
    print(f"  country    : {country} | language: {language} | prioritydomain: {prioritydomain}")
    print(f"  pages/cat  : up to {per_category_pages}")
    print("#" * 60)

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    try:
        from content_quality import quality_score
    except Exception:
        quality_score = None

    analyzer = SentimentIntensityAnalyzer()
    config = _make_config()
    if not _HAVE_NEWSPAPER:
        print("  note: newspaper3k not available - using API description as summary.")
    if VERIFY_SSL and VERIFY_SSL is not True:
        os.environ["REQUESTS_CA_BUNDLE"] = VERIFY_SSL
        os.environ["SSL_CERT_FILE"] = VERIFY_SSL

    seen_links = set()
    all_rows = []
    global_index = 0
    requests_made = 0

    for category in categories:
        next_page = None
        for _page in range(per_category_pages):
            params = {
                "apikey": api_key, "category": category, "country": country,
                "language": language,
            }
            if prioritydomain:
                params["prioritydomain"] = prioritydomain
            if next_page:
                params["page"] = next_page
            try:
                resp = requests.get(LATEST_URL, params=params, timeout=30, verify=VERIFY_SSL)
                requests_made += 1
            except Exception as e:
                print(f"  request error ({category}): {e}")
                break
            if resp.status_code != 200:
                print(f"  api {resp.status_code} ({category}): {resp.text[:160]}")
                break
            data = resp.json()
            results = data.get("results", []) or []
            print(f"  {category} p{_page+1}: {len(results)} articles")
            for item in results:
                url = item.get("link") or ""
                if not url or url.lower().strip() in seen_links:
                    continue
                seen_links.add(url.lower().strip())
                title = item.get("title") or ""
                published = item.get("pubDate", "") or ""
                source = item.get("source_id") or get_source_name(url)
                summary, keywords = _parse_article(url, config)
                if not summary:
                    # Fall back to the API-provided description/content.
                    summary = str(item.get("description") or item.get("content") or "")
                compound, label = _score_sentiment(analyzer, f"{title} {summary}")
                global_index += 1
                row = {
                    "RISK_ID": -1, "SEARCH_TERM_ID": "DISCOVERY",
                    "GOOGLE_INDEX": global_index, "TITLE": title, "LINK": url,
                    "PUBLISHED_DATE": published, "SUMMARY": summary[:500],
                    "KEYWORDS": keywords, "SENTIMENT_COMPOUND": compound,
                    "SENTIMENT": label, "SOURCE": source, "CATEGORY": category,
                    "QUALITY_SCORE": 0,
                }
                if quality_score:
                    try:
                        row["QUALITY_SCORE"] = quality_score(row)
                    except Exception:
                        pass
                all_rows.append(row)
            next_page = data.get("nextPage")
            if not next_page:
                break
            time.sleep(1)

    print(f"\nFetched {len(all_rows)} unique articles in {requests_made} requests.")
    _save(all_rows)


def fetch_lexicon(lexicon, country, language, prioritydomain, per_query_pages,
                  tag="LEXICON"):
    """Risk-lexicon discovery: query /latest with risk-INDICATOR phrases as `q`.

    Rather than browsing categories, this searches the *language of risk*
    ("regulators warn", "emerging risk", ...) so we surface risk-shaped stories
    on any subject - including ones outside the current taxonomy. Each phrase is
    one search; results accumulate + dedup into the shared discovery corpus.
    """
    api_key = os.getenv("NEWS_DATA_API_KEY")
    if not api_key:
        print("ERROR: NEWS_DATA_API_KEY not set (put it in .env or the environment).")
        sys.exit(1)

    print("#" * 60)
    print("RISK DISCOVERY FETCH (risk-lexicon queries, /latest 48h)")
    print(f"  lexicon terms : {len(lexicon)}")
    print(f"  country       : {country} | language: {language} | prioritydomain: {prioritydomain}")
    print(f"  pages/term    : up to {per_query_pages}")
    print("#" * 60)

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    try:
        from content_quality import quality_score
    except Exception:
        quality_score = None

    analyzer = SentimentIntensityAnalyzer()
    config = _make_config()
    if not _HAVE_NEWSPAPER:
        print("  note: newspaper3k not available - using API description as summary.")
    if VERIFY_SSL and VERIFY_SSL is not True:
        os.environ["REQUESTS_CA_BUNDLE"] = VERIFY_SSL
        os.environ["SSL_CERT_FILE"] = VERIFY_SSL

    seen_links = set()
    all_rows = []
    global_index = 0
    requests_made = 0

    for phrase in lexicon:
        next_page = None
        for _page in range(per_query_pages):
            params = {
                "apikey": api_key, "q": phrase, "country": country,
                "language": language,
            }
            if prioritydomain:
                params["prioritydomain"] = prioritydomain
            if next_page:
                params["page"] = next_page
            try:
                resp = requests.get(LATEST_URL, params=params, timeout=30, verify=VERIFY_SSL)
                requests_made += 1
            except Exception as e:
                print(f"  request error ('{phrase}'): {e}")
                break
            if resp.status_code != 200:
                print(f"  api {resp.status_code} ('{phrase}'): {resp.text[:140]}")
                break
            data = resp.json()
            results = data.get("results", []) or []
            print(f"  '{phrase}' p{_page+1}: {len(results)} articles")
            for item in results:
                url = item.get("link") or ""
                if not url or url.lower().strip() in seen_links:
                    continue
                seen_links.add(url.lower().strip())
                title = item.get("title") or ""
                published = item.get("pubDate", "") or ""
                source = item.get("source_id") or get_source_name(url)
                summary, keywords = _parse_article(url, config)
                if not summary:
                    summary = str(item.get("description") or item.get("content") or "")
                compound, label = _score_sentiment(analyzer, f"{title} {summary}")
                global_index += 1
                row = {
                    "RISK_ID": -1, "SEARCH_TERM_ID": tag,
                    "GOOGLE_INDEX": global_index, "TITLE": title, "LINK": url,
                    "PUBLISHED_DATE": published, "SUMMARY": summary[:500],
                    "KEYWORDS": keywords, "SENTIMENT_COMPOUND": compound,
                    "SENTIMENT": label, "SOURCE": source,
                    "CATEGORY": f"lex:{phrase}", "QUALITY_SCORE": 0,
                }
                if quality_score:
                    try:
                        row["QUALITY_SCORE"] = quality_score(row)
                    except Exception:
                        pass
                all_rows.append(row)
            next_page = data.get("nextPage")
            if not next_page:
                break
            time.sleep(1)

    print(f"\nFetched {len(all_rows)} unique articles in {requests_made} requests "
          f"across {len(lexicon)} lexicon terms.")
    _save(all_rows)


def parse_args():
    p = argparse.ArgumentParser(description="Broad un-keyworded news fetch for risk discovery.")
    p.add_argument("--months", type=float, default=4.0, help="Lookback window in months (default 4).")
    p.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES),
                   help="Comma-separated newsdata.io categories (max 5 on Free/Basic).")
    p.add_argument("--country", default="us")
    p.add_argument("--language", default="en")
    p.add_argument("--prioritydomain", default="top",
                   help="top|medium|low - restrict to higher-authority domains (default top).")
    p.add_argument("--chunk-days", type=int, default=7, help="Date-window size per request batch.")
    p.add_argument("--per-window-pages", type=int, default=1,
                   help="Pages to pull per (window,category). Each page = 5 credits.")
    p.add_argument("--max-requests", type=int, default=500,
                   help="Hard cap on API requests (archive = 5 credits each).")
    p.add_argument("--dry-run", action="store_true", help="Plan only; make no API calls.")
    p.add_argument("--probe", action="store_true",
                   help="Make a few tiny archive calls to detect plan/archive access, "
                        "then exit. Zero-result/failed calls are not charged.")
    p.add_argument("--latest", action="store_true",
                   help="Free-tier mode: fetch the past 48h from /latest by category "
                        "(no archive). Accumulates over repeated runs.")
    p.add_argument("--lexicon", action="store_true",
                   help="Risk-lexicon mode: query /latest with risk-indicator phrases "
                        "(the language of risk) instead of categories, to surface "
                        "risk-shaped stories on any subject.")
    p.add_argument("--pages", type=int, default=1,
                   help="Pages per category/term (10 articles/page on free tier).")
    return p.parse_args()


def probe():
    """Detect archive access + history depth with a handful of cheap calls.

    Strategy: hit /archive for narrow 1-day windows at increasing look-back
    distances. The first success proves archive access; the furthest success
    bounds the plan's history depth (Free = none, Basic = ~6mo, Pro = ~2yr).
    """
    api_key = os.getenv("NEWS_DATA_API_KEY")
    if not api_key:
        print("ERROR: NEWS_DATA_API_KEY not set. Set it and re-run --probe.")
        sys.exit(1)

    today = dt.date.today()
    # Look-back checkpoints (days back): ~2d, ~1mo, ~4mo, ~6mo, ~13mo, ~25mo.
    checkpoints = [2, 30, 120, 180, 400, 750]
    print("#" * 60)
    print("ARCHIVE PLAN PROBE (single-day windows; uncharged if empty/failed)")
    print("#" * 60)

    archive_ok = None
    furthest_ok = None
    for days_back in checkpoints:
        day = today - dt.timedelta(days=days_back)
        params = {
            "apikey": api_key, "category": "business", "country": "us",
            "language": "en", "from_date": day.isoformat(), "to_date": day.isoformat(),
        }
        try:
            resp = requests.get(BASE_URL, params=params, timeout=30, verify=VERIFY_SSL)
        except Exception as e:
            print(f"  ~{days_back}d back ({day}): request error: {e}")
            continue

        status = resp.status_code
        try:
            data = resp.json()
        except Exception:
            data = {}
        n = len(data.get("results", []) or [])
        msg = data.get("message") or data.get("results") or ""

        if status == 200:
            archive_ok = True if archive_ok is None else archive_ok
            furthest_ok = days_back
            print(f"  ~{days_back}d back ({day}): OK, {n} article(s)")
        elif status in (401, 403):
            print(f"  ~{days_back}d back ({day}): {status} access denied -> {str(msg)[:120]}")
            if archive_ok is None:
                archive_ok = False
            break
        elif status == 422:
            # 422 is often "date outside your plan's allowed range".
            print(f"  ~{days_back}d back ({day}): 422 out-of-range -> {str(msg)[:120]}")
        else:
            print(f"  ~{days_back}d back ({day}): {status} -> {str(msg)[:120]}")
        time.sleep(1)

    print("-" * 60)
    if archive_ok is False:
        print("VERDICT: No /archive access — likely the FREE plan (latest-only, 48h).")
        print("  -> For discovery, accumulate via /latest going forward instead of a")
        print("     4-month backfill.")
    elif archive_ok:
        if furthest_ok is not None and furthest_ok >= 120:
            print(f"VERDICT: Archive access confirmed back to at least ~{furthest_ok} days.")
            print("  -> 4-month discovery backfill is supported. Run without --probe.")
        else:
            print(f"VERDICT: Archive works but only shallow (~{furthest_ok}d). Plan may be")
            print("     limited; a full 4-month backfill may hit out-of-range past that.")
    else:
        print("VERDICT: Inconclusive (no clear success or denial). Check the messages above.")


def main():
    args = parse_args()
    if args.probe:
        probe()
        return
    categories = [c.strip() for c in args.categories.split(",") if c.strip()][:5]
    if args.lexicon:
        fetch_lexicon(
            lexicon=RISK_LEXICON, country=args.country, language=args.language,
            prioritydomain=args.prioritydomain, per_query_pages=args.pages,
        )
        return
    if args.latest:
        fetch_latest(
            categories=categories, country=args.country, language=args.language,
            prioritydomain=args.prioritydomain, per_category_pages=args.pages,
        )
        return
    fetch(
        months=args.months, categories=categories, country=args.country,
        language=args.language, prioritydomain=args.prioritydomain,
        chunk_days=args.chunk_days, max_requests=args.max_requests,
        per_window_pages=args.per_window_pages, dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
