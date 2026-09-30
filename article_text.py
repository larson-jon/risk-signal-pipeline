# FULL-TEXT ARTICLE EXTRACTION  (prototype for pipeline stage 1 enrichment)
#
# Replaces the abandoned newspaper3k full-text parse with trafilatura, which is
# actively maintained, pure-Python (installs on 3.13), and better at isolating
# main article content from nav/ads/boilerplate.
#
# Design goals:
#   - Best-effort: fetch + extract the article body from its URL.
#   - Cached: every fetched URL is stored on disk (JSON) so re-runs and repeated
#     articles never re-download. Mirrors the embedding_cache pattern.
#   - Polite: per-request timeout, rotating user-agent, courteous delay is the
#     caller's job (this module fetches one URL per call).
#   - Graceful: any failure (network, paywall, empty extract) returns "" so the
#     caller can fall back to the API-provided description. Never raises.
#
# Standalone for now (not yet wired into discover_fetch.py / news_sentiment_
# scraper.py) so its extraction success rate can be measured on real URLs first.
#
# Usage (library):
#   from article_text import fetch_article_text
#   text = fetch_article_text(url, fallback=api_description)
#
# Usage (CLI benchmark):
#   python article_text.py --sample 40           # sample URLs from output/*.csv
#   python article_text.py --url https://...      # test a single URL

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import requests

try:
    import trafilatura
    _HAVE_TRAFILATURA = True
except Exception:  # pragma: no cover - import guard
    _HAVE_TRAFILATURA = False

CACHE_DIR = Path("output/.text_cache")
CACHE_INDEX = CACHE_DIR / "index.json"

# Minimum characters for an extraction to count as a "real" full-text hit.
# Below this we treat it as a failed/thin parse and let the caller fall back.
MIN_TEXT_CHARS = 200
# How much extracted text to keep (enough for NER + noun phrases; keeps the
# cache and any downstream CSV from bloating). None = keep all.
MAX_TEXT_CHARS = 6000

REQUEST_TIMEOUT = 20

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0 Safari/537.36",
]


def _key(url):
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def _load_index():
    if CACHE_INDEX.exists():
        try:
            return json.loads(CACHE_INDEX.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_index(index):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_INDEX.write_text(json.dumps(index), encoding="utf-8")


# In-process cache of the on-disk index so batch runs don't re-read it per call.
_INDEX = None


def _get_cached(url):
    global _INDEX
    if _INDEX is None:
        _INDEX = _load_index()
    return _INDEX.get(_key(url))


def _set_cached(url, text):
    global _INDEX
    if _INDEX is None:
        _INDEX = _load_index()
    _INDEX[_key(url)] = text
    _save_index(_INDEX)


def _clean(text):
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


def _download_html(url, session=None):
    """Fetch raw HTML with a timeout and a rotating user-agent. Returns '' on
    any error. trafilatura has its own fetcher, but using requests lets us reuse
    the project's SSL/user-agent conventions and a shared session."""
    headers = {"User-Agent": random.choice(_USER_AGENTS)}
    try:
        get = (session or requests).get
        resp = get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200 or not resp.text:
            return ""
        return resp.text
    except Exception:
        return ""


def extract_from_html(html):
    """Run trafilatura over raw HTML; return cleaned main-content text or ''."""
    if not _HAVE_TRAFILATURA or not html:
        return ""
    try:
        text = trafilatura.extract(
            html, include_comments=False, include_tables=False,
            no_fallback=False, favor_precision=True,
        )
    except Exception:
        return ""
    text = _clean(text)
    if MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
    return text


def fetch_article_text(url, fallback="", session=None, use_cache=True):
    """Best-effort full article text for `url`.

    Returns the extracted body when it's substantial (>= MIN_TEXT_CHARS),
    otherwise returns `fallback` (typically the API description). Never raises.
    Caches the extracted text per URL so repeat calls are free.
    """
    if not url:
        return _clean(fallback)

    if use_cache:
        cached = _get_cached(url)
        if cached is not None:
            return cached if len(cached) >= MIN_TEXT_CHARS else _clean(fallback)

    html = _download_html(url, session=session)
    text = extract_from_html(html)

    if use_cache:
        # Cache whatever we extracted (even ""), so we don't refetch dead URLs.
        _set_cached(url, text)

    if len(text) >= MIN_TEXT_CHARS:
        return text
    return _clean(fallback)


# --------------------------------------------------------------------------
# CLI benchmark: measure extraction success rate on real URLs.
# --------------------------------------------------------------------------

def _sample_urls(n):
    """Pull a random sample of article URLs from the output CSVs."""
    import pandas as pd
    urls = []
    for p in Path("output").glob("*.csv"):
        try:
            df = pd.read_csv(p, usecols=lambda c: c in ("LINK", "SUMMARY"))
        except Exception:
            continue
        if "LINK" not in df.columns:
            continue
        for _, row in df.iterrows():
            link = str(row.get("LINK", "") or "").strip()
            summ = str(row.get("SUMMARY", "") or "").strip()
            if link.startswith("http"):
                urls.append((link, summ))
    # de-dup by URL, keep a random sample
    seen, uniq = set(), []
    random.shuffle(urls)
    for link, summ in urls:
        if link in seen:
            continue
        seen.add(link)
        uniq.append((link, summ))
    return uniq[:n]


def _benchmark(sample, delay):
    session = requests.Session()
    hits = fails = 0
    total_full = total_fallback = 0
    print(f"Testing {len(sample)} URLs (trafilatura available: {_HAVE_TRAFILATURA})\n")
    for i, (url, summ) in enumerate(sample, 1):
        # Bypass cache for a true fetch measurement, but still populate it.
        html = _download_html(url, session=session)
        text = extract_from_html(html)
        _set_cached(url, text)
        ok = len(text) >= MIN_TEXT_CHARS
        if ok:
            hits += 1
            total_full += len(text)
        else:
            fails += 1
            total_fallback += len(_clean(summ))
        host = url.split("/")[2] if "://" in url else url[:40]
        print(f"  [{i:>3}] {'FULL ' if ok else 'fall '} "
              f"{len(text):>5}c  {host[:40]}")
        if delay:
            time.sleep(delay)

    n = len(sample) or 1
    print("\n" + "=" * 56)
    print(f"Full-text extracted : {hits}/{n}  ({100*hits/n:.0f}%)")
    print(f"Fell back to summary: {fails}/{n}  ({100*fails/n:.0f}%)")
    if hits:
        print(f"Avg full-text length: {total_full // hits:,} chars "
              f"(vs ~{(total_fallback // fails) if fails else 0} char summaries)")
    print("=" * 56)


def main():
    p = argparse.ArgumentParser(description="Benchmark trafilatura full-text extraction.")
    p.add_argument("--sample", type=int, default=40, help="How many URLs to sample from output/*.csv.")
    p.add_argument("--url", help="Test a single URL instead of sampling.")
    p.add_argument("--delay", type=float, default=0.5, help="Politeness delay between fetches (s).")
    args = p.parse_args()

    if not _HAVE_TRAFILATURA:
        print("trafilatura is not installed. Run: pip install trafilatura")
        return

    if args.url:
        text = fetch_article_text(args.url, use_cache=False)
        print(f"Extracted {len(text)} chars:\n")
        print(text[:1500])
        return

    sample = _sample_urls(args.sample)
    if not sample:
        print("No article URLs found in output/*.csv.")
        return
    _benchmark(sample, args.delay)


if __name__ == "__main__":
    main()
