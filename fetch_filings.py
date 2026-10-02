# RISK-FILING FETCHER  (Path E: SEC EDGAR 10-K / 10-Q Risk Factors)
#
# Public companies must disclose their material risks in Item 1A "Risk Factors"
# of their annual (10-K) and quarterly (10-Q) filings. For FINRA-regulated
# broker-dealers and asset managers, that section is self-reported, primary-
# source risk signal - free and legal via SEC EDGAR.
#
# This fetcher uses the edgartools library (which parses a filing into a typed
# object with an addressable Risk Factors section - far more reliable than
# regex on the 5 MB filing HTML, which trips over tables of contents and
# cross-references). Each firm's Risk Factors section is split into individual
# risk factors (one per sub-heading/paragraph) so each becomes its own row -
# otherwise one firm would be a single 50k-char blob and clustering couldn't
# compare risks across firms.
#
# Output: output/filings_news.csv in the pipeline schema, tagged
# SEARCH_TERM_ID = "FILING". Appended + de-duplicated.
#
# Usage:
#   python fetch_filings.py                       # default watchlist, latest 10-K each
#   python fetch_filings.py --tickers SCHW,MS,LPLA --forms 10-K
#   python fetch_filings.py --split-factors        # one row per risk factor (default)
#   python fetch_filings.py --whole                 # one row per filing (whole section)

import argparse
import csv
import re
from pathlib import Path

# A starter watchlist of large, public, FINRA-member-affiliated firms
# (broker-dealers, custodians, asset managers). Edit freely.
DEFAULT_TICKERS = [
    "SCHW",   # Charles Schwab
    "MS",     # Morgan Stanley
    "GS",     # Goldman Sachs
    "LPLA",   # LPL Financial
    "RJF",    # Raymond James
    "SF",     # Stifel
    "HOOD",   # Robinhood
    "IBKR",   # Interactive Brokers
    "AMP",    # Ameriprise
    "BLK",    # BlackRock
]

OUTPUT_CSV = Path("output/filings_news.csv")
EDGAR_IDENTITY = "FINRA ERM research prototype erm-research@finra.org"
MAX_FACTOR_CHARS = 4000
MIN_FACTOR_CHARS = 200

PIPELINE_COLUMNS = [
    "RISK_ID", "SEARCH_TERM_ID", "GOOGLE_INDEX", "TITLE", "LINK",
    "PUBLISHED_DATE", "SUMMARY", "KEYWORDS", "SENTIMENT_COMPOUND",
    "SENTIMENT", "SOURCE", "CATEGORY", "QUALITY_SCORE",
]


def _clean(t):
    """Normalize whitespace and fix common filing-text mojibake / artifacts."""
    if not t:
        return ""
    try:
        from unidecode import unidecode
        t = unidecode(t)   # edgartools ships unidecode; collapses smart quotes etc.
    except Exception:
        pass
    t = t.replace("\xa0", " ")
    return re.sub(r"\s+", " ", t).strip()


_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "our",
    "we", "us", "may", "could", "would", "that", "this", "these", "those", "as",
    "is", "are", "be", "by", "it", "its", "from", "at", "which", "such", "any",
}


def _tokens(text):
    """Content-word token set for Jaccard similarity (lowercased, destopped)."""
    words = re.findall(r"[a-z]{3,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / len(a | b)


def _dedup_factors(factors, threshold=0.6):
    """Collapse near-duplicate risk factors within one filing.

    10-Ks often carry a bullet-point *summary* of risks AND a detailed section
    restating the same risks at length - that double-counts every risk. We
    compare each factor's content-word set; when two overlap heavily, we keep
    the LONGER (more detailed) one and drop the shorter. Also drops a short
    factor that is largely contained in a longer one (containment >= 0.8),
    which catches summary-bullet vs full-paragraph pairs.
    """
    # Longest first so the detailed version is the one we keep.
    ordered = sorted(factors, key=len, reverse=True)
    kept, kept_tokens = [], []
    for f in ordered:
        ft = _tokens(f)
        dup = False
        for kt in kept_tokens:
            if _jaccard(ft, kt) >= threshold:
                dup = True
                break
            # containment: most of this (shorter) factor's words are in a kept one
            if ft and len(ft & kt) / len(ft) >= 0.8:
                dup = True
                break
        if not dup:
            kept.append(f)
            kept_tokens.append(ft)
    # restore original document order for readability
    order_index = {id(f): i for i, f in enumerate(factors)}
    kept.sort(key=lambda f: order_index.get(id(f), 0))
    return kept


def _split_risk_factors(section_text):
    """Split a Risk Factors section into individual factors, then de-duplicate.

    10-K risk factors are typically short bold sub-headings followed by a
    paragraph. We split on blank-line boundaries first; if the section is one
    run, fall back to sentence-run boundaries. Near-duplicate units (summary
    bullets vs detailed paragraphs) are then collapsed.
    """
    text = section_text
    parts = re.split(r"\n\s*\n+", text)
    if len(parts) < 3:
        # single-blob fallback: break into ~paragraph chunks on sentence runs
        parts = re.split(r"(?<=[.])\s+(?=[A-Z][a-z]+(?:\s+\w+){2,}\s+(?:could|may|would|risk|adversely|our|we|the))",
                         text)
    factors = []
    for p in parts:
        p = _clean(p)
        if len(p) >= MIN_FACTOR_CHARS:
            factors.append(p[:MAX_FACTOR_CHARS])
    before = len(factors)
    factors = _dedup_factors(factors)
    if before != len(factors):
        print(f"      dedup: {before} -> {len(factors)} units")
    return factors


def _factor_title(factor_text):
    """Derive a short title from the first sentence/clause of a risk factor."""
    first = re.split(r"(?<=[.])\s", factor_text, maxsplit=1)[0]
    first = first.strip()
    return (first[:140] + "...") if len(first) > 140 else first


def fetch(tickers, forms, split_factors, per_form_limit):
    from edgar import set_identity, Company
    set_identity(EDGAR_IDENTITY)

    items = []
    for ticker in tickers:
        try:
            company = Company(ticker)
        except Exception as e:
            print(f"  {ticker}: company lookup failed ({e})")
            continue
        name = getattr(company, "name", ticker)
        for form in forms:
            try:
                # amendments=False excludes 10-K/A amendments, which usually
                # restate only part of the filing (often Part III) and omit the
                # Risk Factors section - that was why IBKR/AMP returned nothing.
                try:
                    filings = company.get_filings(form=form, amendments=False).head(per_form_limit)
                except TypeError:
                    # older edgartools without the amendments kwarg: filter by hand
                    allf = company.get_filings(form=form)
                    filings = [f for f in allf if str(getattr(f, "form", "")).upper() == form][:per_form_limit]
            except Exception as e:
                print(f"  {ticker} {form}: filings lookup failed ({e})")
                continue
            got = 0
            for f in filings:
                try:
                    obj = f.obj()
                    rf = getattr(obj, "risk_factors", None)
                    rf = str(rf) if rf else ""
                except Exception as e:
                    print(f"  {ticker} {form} {f.accession_no}: parse failed ({e})")
                    continue
                if not rf or len(rf) < MIN_FACTOR_CHARS:
                    print(f"  {ticker} {form} {f.filing_date}: no risk factors found")
                    continue
                date = str(f.filing_date)
                link = getattr(f, "filing_url", "") or getattr(f, "homepage_url", "") or ""
                units = _split_risk_factors(rf) if split_factors else [_clean(rf)[:MAX_FACTOR_CHARS]]
                for u in units:
                    items.append({
                        "ticker": ticker, "name": name, "form": form,
                        "date": date, "link": link, "text": u,
                    })
                got += 1
                print(f"  ok {ticker:5} {form} {date}  ->  {len(units)} risk unit(s)")
            if got == 0:
                print(f"  {ticker} {form}: nothing fetched")
    return items


def to_pipeline_rows(items):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    try:
        from content_quality import quality_score
    except Exception:
        quality_score = None

    out = []
    for i, it in enumerate(items):
        text = it["text"]
        comp = analyzer.polarity_scores(text)["compound"]
        label = ("Positive" if comp >= 0.05 else "Negative" if comp <= -0.05 else "Neutral")
        # Title = the risk's own lede (no ticker/form prefix). Keeping the
        # ticker out of TITLE prevents "SCHW 10-K" boilerplate from polluting
        # the topic-clustering keywords; the firm is retained in SOURCE/CATEGORY.
        title = _factor_title(text)
        row = {
            "RISK_ID": -1,
            "SEARCH_TERM_ID": "FILING",
            "GOOGLE_INDEX": i,
            "TITLE": title[:180],
            "LINK": it["link"],
            "PUBLISHED_DATE": it["date"],
            "SUMMARY": text,
            "KEYWORDS": "",
            "SENTIMENT_COMPOUND": comp,
            "SENTIMENT": label,
            "SOURCE": it["name"],
            "CATEGORY": f"filing:{it['form'].lower().replace('-', '')}:{it['ticker']}",
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
    # De-dup on (TITLE, PUBLISHED_DATE) so re-runs of the same filing collapse.
    combined["_k"] = (combined["TITLE"].astype(str) + "|" + combined["PUBLISHED_DATE"].astype(str))
    combined = combined.drop_duplicates(subset="_k", keep="first").drop(columns="_k")
    combined["GOOGLE_INDEX"] = range(len(combined))
    combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
    return len(new_df), len(combined), before - len(combined)


def main():
    p = argparse.ArgumentParser(description="Fetch SEC EDGAR Risk Factors into the pipeline schema.")
    p.add_argument("--tickers", default=",".join(DEFAULT_TICKERS),
                   help="Comma-separated tickers (default: a FINRA-firm watchlist).")
    p.add_argument("--forms", default="10-K",
                   help="Comma-separated filing forms to pull (default 10-K; e.g. '10-K,10-Q').")
    p.add_argument("--per-form-limit", type=int, default=1, dest="per_form_limit",
                   help="Most-recent N filings per form per firm (default 1).")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--split-factors", action="store_true", default=True,
                     help="One row per individual risk factor (default).")
    grp.add_argument("--whole", action="store_true",
                     help="One row per filing (whole Risk Factors section).")
    args = p.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    forms = [f.strip().upper() for f in args.forms.split(",") if f.strip()]
    split = not args.whole

    print("#" * 60)
    print("RISK-FILING FETCH (SEC EDGAR Item 1A Risk Factors)")
    print(f"Firms: {len(tickers)} | forms: {forms} | "
          f"{'split into factors' if split else 'whole section'}")
    print("#" * 60)

    items = fetch(tickers, forms, split, args.per_form_limit)
    if not items:
        print("\nNo risk factors fetched.")
        return
    rows = to_pipeline_rows(items)
    added, total, deduped = save(rows)
    print(f"\nFetched {added} risk units -> {OUTPUT_CSV} "
          f"({total} total after de-dup, {deduped} duplicates dropped).")
    print("Cluster with: python discover_score.py --input output/filings_news.csv "
          "--out output/filings_ranked.csv --source all --min-topic-size 6")


if __name__ == "__main__":
    main()
