# ONE-TIME BACKFILL: upgrade existing discovery SUMMARYs with full article text.
#
# The free news plan only serves the last ~48h, so re-fetching won't add full
# text to articles already collected. This script walks the existing
# output/discovery_news.csv, fetches each article's full body via trafilatura
# (article_text.fetch_article_text, cached per URL), and replaces the short
# API summary with the extracted body when extraction succeeds. Rows where
# extraction fails keep their existing summary.
#
# Resumable: the per-URL text cache means re-running only fetches URLs not yet
# tried. Safe: writes to a temp file then swaps, so an interrupted run won't
# corrupt the corpus.
#
# Usage:
#   python backfill_fulltext.py                 # lexicon rows only (default)
#   python backfill_fulltext.py --source all    # every row
#   python backfill_fulltext.py --limit 100     # first N (smoke test)

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

from article_text import fetch_article_text, MIN_TEXT_CHARS, _HAVE_TRAFILATURA

CSV = Path("output/discovery_news.csv")
SUMMARY_CHARS = 4000


def main():
    ap = argparse.ArgumentParser(description="Backfill full article text into discovery_news.csv.")
    ap.add_argument("--source", choices=["lexicon", "all"], default="lexicon",
                    help="Which rows to backfill (default: lexicon, since scoring is lexicon-only).")
    ap.add_argument("--limit", type=int, default=0, help="Only process the first N target rows (0 = all).")
    ap.add_argument("--delay", type=float, default=0.4, help="Politeness delay between network fetches (s).")
    args = ap.parse_args()

    if not _HAVE_TRAFILATURA:
        print("trafilatura not installed. Run: pip install trafilatura")
        return
    if not CSV.exists():
        print(f"ERROR: {CSV} not found.")
        return

    df = pd.read_csv(CSV)
    df["SUMMARY"] = df["SUMMARY"].fillna("").astype(str)

    # Which rows to target
    mask = pd.Series(True, index=df.index)
    if args.source == "lexicon" and "SEARCH_TERM_ID" in df.columns:
        mask = df["SEARCH_TERM_ID"] == "LEXICON"
    targets = df[mask].index.tolist()
    if args.limit:
        targets = targets[:args.limit]

    print(f"Corpus: {len(df)} rows. Backfilling {len(targets)} '{args.source}' rows.")
    print(f"Full-text cache: output/.text_cache/ (resumable).\n")

    def _write():
        """Safe write: temp then atomic replace, so an interrupted run never
        corrupts the corpus."""
        tmp = CSV.with_suffix(".csv.tmp")
        df.to_csv(tmp, index=False, encoding="utf-8")
        tmp.replace(CSV)

    session = requests.Session()
    upgraded = kept = 0
    start = time.time()
    for n, i in enumerate(targets, 1):
        url = str(df.at[i, "LINK"] or "").strip()
        if not url.startswith("http"):
            kept += 1
            continue
        text = fetch_article_text(url, fallback="", session=session)  # "" so we detect real hits
        if len(text) >= MIN_TEXT_CHARS:
            df.at[i, "SUMMARY"] = text[:SUMMARY_CHARS]
            upgraded += 1
        else:
            kept += 1  # leave the existing summary in place
        if n % 50 == 0 or n == len(targets):
            rate = n / max(1e-9, time.time() - start)
            eta = (len(targets) - n) / max(1e-9, rate)
            # Checkpoint the CSV every 50 rows so progress survives an
            # interruption (overnight stop, sleep, etc.) - the run is resumable
            # from the URL cache, and the corpus always reflects work done.
            _write()
            print(f"  {n}/{len(targets)}  upgraded {upgraded}  kept {kept}  "
                  f"({rate:.1f}/s, ETA {eta/60:.1f} min)  [checkpointed]", flush=True)
        if args.delay:
            time.sleep(args.delay)

    _write()

    elapsed = (time.time() - start) / 60
    print(f"\nDone in {elapsed:.1f} min. Upgraded {upgraded} rows to full text, "
          f"kept {kept} on existing summary.")
    up = df.loc[targets, "SUMMARY"].astype(str).str.len().mean() if targets else 0
    print(f"Avg SUMMARY length for targeted rows now: {int(up):,} chars.")


if __name__ == "__main__":
    main()
