# Email digest → clustering pipeline

A daily FINRA/financial-news digest email is a **higher-precision source** than
the newsdata.io API: it's already curated to financially-relevant and
FINRA-specific stories, so it avoids the sports/entertainment noise that fills
the broad feeds. This doc explains how to get that email into the pipeline.

## The design in one line

Copilot (which can read the mailbox) **extracts** the email to a file; this repo's
`ingest_email_news.py` **normalizes** it into the pipeline's CSV schema; the
existing clustering/trends/status code runs on it unchanged. The two tools meet
at a file — they never call each other.

```
Daily email  →  Copilot  →  a file in inbox/  →  ingest_email_news.py  →  output/email_news.csv  →  discover_score.py
 (M365)        (extract)      (handoff)            (normalize+score)         (pipeline schema)        (cluster)
```

## Step 1 — Copilot extracts the email

You run this in Microsoft 365 Copilot (it has mailbox access; this tool does not).
Two output options — **either works**, because the ingest script auto-detects the input.

### Option A — save the raw email text (simplest)
Copilot prompt:

> "Open today's [NAME] financial-news digest email. Output its full body as plain
> text, preserving the section headers (e.g. 'FINRA in the News', 'SEC',
> 'Financial Regulation'), each story's 'Source, Date' line, headline, byline,
> summary paragraph, and the 'Also:' / 'Related:' / 'Without FINRA mention:'
> lists. Save it as a .txt file."

Save it into this repo's `inbox/` folder, e.g. `inbox/2026-09-30.txt`.

### Option B — structured CSV (more robust if the email HTML is messy)
Copilot prompt:

> "From today's [NAME] financial-news digest email, extract every story as a CSV
> row. Columns: TITLE, LINK, SOURCE, PUBLISHED_DATE, SUMMARY, SECTION, PAYWALLED.
> SECTION is the digest heading the story appears under. PAYWALLED is yes/no
> based on a '(Paywalled)' marker. Include the stories listed under 'Also:' /
> 'Related:' / 'Without FINRA mention:' as their own rows. Save as a .csv file."

Save it as e.g. `inbox/2026-09-30.csv`.

> **Tip:** Option A is easiest to produce but relies on the digest's text
> structure staying consistent. Option B is more work for Copilot but less
> fragile. Start with A; switch to B if the parser misses stories.

## Step 2 — ingest it

```powershell
python ingest_email_news.py --in inbox/2026-09-30.txt
# or:  python ingest_email_news.py --in inbox/2026-09-30.csv
# add --no-fulltext to skip fetching full article text (faster; summaries only)
```

What the script does:
- Parses the digest into stories (primary stories + the "Also/Related" siblings).
- Tags each row `RISK_ID = -1`, `SEARCH_TERM_ID = "EMAIL"`, `CATEGORY = email:<section>`
  (plus `:paywalled` where flagged) — its own lane, like LEXICON/DISCOVERY.
- Scores VADER sentiment and a content-quality score.
- Optionally fetches full article text (via `article_text.py`, trafilatura) for
  non-paywalled links, falling back to the email summary.
- Appends to `output/email_news.csv`, de-duplicating by LINK (or TITLE+SOURCE
  when no link). Safe to run every day — it accumulates and dedupes.

## Step 3 — cluster it

The output is the same schema the pipeline consumes, so:

```powershell
python discover_score.py --input output/email_news.csv --out output/email_ranked.csv --source all
```

Then regenerate the reports (`python build_index.py`) to see the email-sourced
themes in the discovery page, dot plot, and candidate list. (Use `--source all`
because email rows are tagged `EMAIL`, not `LEXICON`/`DISCOVERY`.)

## Why this source is good — and the honest limits

**Good:**
- Pre-curated for financial + FINRA relevance → far less noise.
- The "Also:/Related:" lists are a built-in **importance signal** (a story
  covered by 15 outlets matters more) and a natural dedup grouping.
- Reuses the whole pipeline — clustering, trends, status board, article links.

**Limits:**
- The handoff is **semi-manual**: Copilot writes the file, you drop it in `inbox/`
  and run the script. Fully automatic email→pipeline would need a Graph API
  mailbox integration — a larger, IT-governed project. Manual drop is the
  pragmatic prototype.
- Extraction quality depends on the email's structure. Eyeball the first few
  `ingest_email_news.py` runs ("Parsed N stories…") and spot-check `email_news.csv`.
- `(Paywalled)` links won't full-text extract — they keep the email summary.
- The parser is tuned to the current digest format; if the email layout changes,
  the raw-text parser (Option A) may need adjusting — Option B is the fallback.

## File layout

```
inbox/                      # drop Copilot's extracted email files here (gitignored)
ingest_email_news.py        # parse + normalize + score -> output/email_news.csv
output/email_news.csv       # accumulated email corpus, pipeline schema
output/email_ranked.csv     # clustered/scored themes (from discover_score.py)
```
