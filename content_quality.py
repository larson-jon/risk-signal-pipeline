# CONTENT QUALITY SCORING
#
# Scores each article 0..1 on how likely it is to be *substantive news* rather
# than boilerplate: press-release / wire spam, templated digests, earnings-call
# transcripts, commodity-forecast tickers, or thin (unparsed / paywalled) text.
#
# This fills the QUALITY_SCORE column that already exists in the pipeline
# schema (previously always 0). Downstream stages use it to keep boilerplate
# from dominating trend and taxonomy-gap analysis.
#
# The penalties are data-driven from the sources and title patterns actually
# seen in the fetched corpus. Thresholds are deliberately conservative: a low
# score flags a *likely* low-signal article, not a certainty.

import re

# Sources that are overwhelmingly press releases, syndication, or SEO/ticker
# spam in this corpus. Matched as a substring against the SOURCE domain slug.
_LOW_QUALITY_SOURCES = {
    "einpresswire", "globenewswire", "prnewswire", "businesswire", "accesswire",
    "cision", "menafn", "webpronews", "bundle_app", "newspub_live", "headtopics",
    "financialcontent", "insidermonkey", "baseballnewssource", "completeaitraining",
    "defenseworld", "analyticsinsight", "investingnews", "beforeitsnews",
    "postregister", "hastingstribune", "kdhnews", "mankatofreepress",
}

# Mid-quality aggregators: not spam, but often thin / rewritten. Small penalty.
_MID_QUALITY_SOURCES = {
    "fool", "benzinga", "investing_us", "forexlive", "yahoo", "zerohedge",
}

# Title patterns that mark templated / non-story content.
_BOILERPLATE_TITLE = re.compile(
    r"(daily summary|summarybrief|summary brief|trending summary|news summary|"
    r"earnings call (transcript|highlights)|q\d\s*\d{4}\s*earnings|"
    r"eps forecast|price forecast|forecast:?\s|market (hits|to hit|size)|"
    r"\bcagr\b|weekly recap|threatsday|est state|puerto ap|"
    r"winds|cloudy|mph|forecast:)",
    re.I,
)

# A "market report" press-release shape, e.g.
# "PW Consulting: X Market Hits $Y at Z% CAGR ... to 2032".
_MARKET_REPORT = re.compile(r"market.*(\$|\bUSD\b|billion|million|cagr|20\d\d)", re.I)


def _source_slug(source):
    return str(source or "").strip().lower()


def quality_score(row):
    """Return a 0..1 quality score for one article row (dict-like).

    Starts at 1.0 and subtracts penalties. Higher = more substantive.
    """
    score = 1.0
    title = str(row.get("TITLE", "") or "")
    summary = str(row.get("SUMMARY", "") or "").strip()
    source = _source_slug(row.get("SOURCE"))

    # --- source reputation ---
    if any(s in source for s in _LOW_QUALITY_SOURCES):
        score -= 0.5
    elif any(s == source for s in _MID_QUALITY_SOURCES):
        score -= 0.15

    # --- templated / boilerplate titles ---
    if _BOILERPLATE_TITLE.search(title):
        score -= 0.4
    if _MARKET_REPORT.search(title):
        score -= 0.25

    # --- thin content (parse failed / paywalled) ---
    if not summary:
        score -= 0.3
    elif len(summary) < 120:
        score -= 0.1

    # Legal-disclaimer boilerplate that some PR wires inject as "summary".
    if "legal disclaimer" in summary.lower() or "provides this news content" in summary.lower():
        score -= 0.3

    return round(max(0.0, min(1.0, score)), 3)


def score_frame(df):
    """Return a Series of quality scores for a DataFrame of articles."""
    return df.apply(quality_score, axis=1)


if __name__ == "__main__":
    # quick smoke test with representative rows
    samples = [
        {"TITLE": "SEC charges firm in $50M fraud scheme", "SUMMARY": "The Securities and "
         "Exchange Commission today announced charges against a broker-dealer accused of "
         "orchestrating a large-scale fraud affecting hundreds of retail investors.",
         "SOURCE": "reuters"},
        {"TITLE": "AP Trending SummaryBrief at 9:24 p.m. EDT", "SUMMARY": "", "SOURCE": "menafn"},
        {"TITLE": "PW Consulting: Metal Aerosol Can Market Hits $132.08M in 2025, Forecast 3.9% CAGR",
         "SUMMARY": "", "SOURCE": "globenewswire"},
        {"TITLE": "Coda Octopus (CODA) Q3 2026 Earnings Call Transcript",
         "SUMMARY": "Operator: Welcome to the call.", "SOURCE": "fool"},
    ]
    for s in samples:
        print(f"{quality_score(s):.2f}  {s['TITLE'][:60]}")
