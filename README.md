# risk-signal-pipeline

A pipeline that turns news coverage into risk signals. It fetches news articles for a
set of enterprise and emerging risk search terms, scores their sentiment, enriches them
with spaCy NLP, and groups them into topics through clustering.

## Pipeline stages

1. **Fetch** — pull recent articles for each risk search term from the newsdata.io API,
   parse them with `newspaper3k`, and score sentiment with VADER.
2. **Sentiment** — each article is labeled Positive / Negative / Neutral with a compound
   score.
3. **spaCy NLP** — extract named entities (organizations, people, places, groups, laws) and
   cleaned noun-phrase keywords from article text (`spacy_enrichment.py`).
4. **Topic clustering** — embed articles with `sentence-transformers`, cluster them into
   topics with BERTopic, and surface which risks cluster together (`topic_clustering.py`).
5. **Content quality** — score each article 0–1 on how substantive it is, penalizing
   press-release / wire boilerplate, templated digests, and thin text (`content_quality.py`).
   Fills the previously-unused `QUALITY_SCORE` column.
6. **Trend tracking** — bin each topic's articles by month and score month-over-month
   *signal intensity*, so growing or fading themes stand out (`topic_trends.py`).
7. **Taxonomy gaps** — measure how loosely each topic fits the risk that surfaced it
   (embedding distance) plus novelty, to flag possible uncovered risk aspects (`topic_gaps.py`).
8. **Report** — build standalone, interactive HTML explorers (topics, trends, gaps, and a
   methodology tab) in `reports/`, with a landing-page `index.html` (`build_report.py`,
   `build_index.py`).

> All stages are implemented. Stage 1–2 run in `news_sentiment_scraper.py`; stages 3–5 run in
> `topic_clustering.py` (which invokes `spacy_enrichment.py` and `content_quality.py`); stage
> 6 in `topic_trends.py`; stage 7 in `topic_gaps.py`; stage 8 in `build_report.py` +
> `build_index.py`. Data CSVs are written to `output/`; publishable HTML reports to `reports/`.

### How the topic stage works

`topic_clustering.py` reads a sentiment CSV, then:

- runs each article's title + summary through **spaCy** to pull named entities and
  noun-phrase keywords ([spacy.io](https://spacy.io/)),
- embeds the same text with **sentence-transformers** so semantically similar stories sit
  close together ([sbert.net](https://sbert.net/)),
- clusters the embeddings with **BERTopic** to form interpretable topics
  ([BERTopic docs](https://maartengr.github.io/BERTopic/index.html)).

Clustering parameters (UMAP neighbors, HDBSCAN cluster size) scale with the number of
articles, so the stage works on both small samples and larger production runs.

### How the trend stage works

The goal is to see whether risk-related themes are **growing or fading over time**, and to
spot emerging aspects of risk the current taxonomy may not capture. To keep topic IDs
comparable across months, `topic_trends.py` uses the topics from a single full-history
clustering run, then bins each topic's articles by calendar month and scores a composite
**signal intensity** per topic per month:

```
volume       = number of articles for the topic that month
neg_share    = fraction of those articles labeled Negative
avg_compound = mean VADER compound score that month (-1..+1)
severity     = 1 + neg_share + max(0, -avg_compound)      (>= 1)
intensity    = volume * severity
```

So a negative surge scores higher than a neutral one of equal volume, while intensity never
falls below raw volume. Per topic it also computes a trend slope, recent **momentum**
(last 3 months vs the prior 3), and peak month/intensity. The outlier cluster is always
excluded, and boilerplate / low-signal topics (overwhelmingly neutral and thinly spread)
are flagged `IS_BENIGN` so they can be dropped from the view.

## Project layout

```
.
├── news_sentiment_scraper.py   # fetch + sentiment (stages 1-2)
├── spacy_enrichment.py         # spaCy entity + keyword enrichment (stage 3)
├── content_quality.py          # article quality scoring (stage 5)
├── topic_clustering.py         # sentence-transformers + BERTopic + quality (stages 3-5)
├── topic_trends.py             # monthly signal-intensity trends (stage 6)
├── topic_gaps.py               # taxonomy-fit / novelty scoring (stage 7)
├── build_report.py             # interactive HTML reports (stage 8)
├── build_index.py              # landing-page index over the reports
├── download_model.py           # fetch embedding model for offline use
├── data/                       # encoded risk search-term lists
│   ├── EnterpriseRisksListEncoded.csv
│   └── EmergingRisksListEncoded.csv
├── output/                     # generated CSVs (sentiment, topics, trends, gaps)
├── reports/                    # publishable HTML reports + index.html
├── tests/                      # test suite
├── .github/workflows/          # CI / scheduled runs
└── requirements.txt
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# one-time: download the spaCy English model used by the topic stage
python -m spacy download en_core_web_sm
```

### Embedding model (offline / restricted networks)

The topic stage embeds articles with the `all-MiniLM-L6-v2` sentence-transformers
model. It is normally downloaded from Hugging Face on first use. On networks where
`huggingface.co` is blocked or returning errors, fetch a local copy once:

```powershell
python download_model.py
```

This creates an `all-MiniLM-L6-v2-local/` folder that `topic_clustering.py` discovers
automatically and loads fully offline. You can also point at any local model folder or hub
name with the `EMBEDDING_MODEL` / `LOCAL_EMBEDDING_MODEL` environment variables.

## Usage

The scraper needs a newsdata.io API key in the `NEWS_DATA_API_KEY` environment variable.

```powershell
$env:NEWS_DATA_API_KEY = "your-api-key"

# fetch + score enterprise risk news
python news_sentiment_scraper.py --risk-type enterprise

# or emerging risk news
python news_sentiment_scraper.py --risk-type emerging
```

Results are written to `output/enterprise_risks_online_sentiment.csv` or
`output/emerging_risks_online_sentiment.csv`. Re-runs append and de-duplicate by article link.

### Output columns

| Column | Description |
| --- | --- |
| RISK_ID | Risk identifier the article was matched to |
| SEARCH_TERM_ID | Search term that surfaced the article |
| TITLE / LINK | Article headline and URL |
| PUBLISHED_DATE | Publish date reported by the source |
| SUMMARY / KEYWORDS | Extracted summary and keywords |
| SENTIMENT_COMPOUND | VADER compound score |
| SENTIMENT | Positive / Negative / Neutral label |
| SOURCE | Publisher domain |

### Topic clustering

Once a sentiment CSV exists, cluster it into topics:

```powershell
# uses output/enterprise_risks_online_sentiment.csv
python topic_clustering.py --risk-type enterprise

# or emerging
python topic_clustering.py --risk-type emerging

# or point at any sentiment CSV directly
python topic_clustering.py --input output/enterprise_risks_online_sentiment.csv
```

This writes two files per risk type:

- `output/<type>_risks_topics.csv` — every article with its assigned `TOPIC_ID`,
  `TOPIC_LABEL`, a readable `TOPIC_DESCRIPTION`, spaCy `ENTITIES`, and `NLP_KEYWORDS`.
- `output/<type>_risks_topic_summary.csv` — one row per topic with article count,
  which `RISK_IDS` fall into it, the dominant sentiment, top keywords / entities, and a
  generated `TOPIC_DESCRIPTION`.

The topic summary is the quickest way to see clusters of risk: topics spanning several
distinct `RISK_IDS` point to themes that cut across the risk landscape.

Add `--days 30` to cluster only the last N days (anchored to the most recent article date);
outputs get a `_last30d` suffix. Set `EMBEDDING_MODEL` to swap the sentence-transformers
model (defaults to `all-MiniLM-L6-v2`).

### Trend tracking

Cluster the **full history** first (no `--days`), then build monthly trends:

```powershell
python topic_trends.py --risk-type enterprise
python topic_trends.py --risk-type emerging

# exclude boilerplate / low-signal topics
python topic_trends.py --risk-type emerging --drop-benign
```

This writes:

- `output/<type>_risks_trends.csv` — one row per (topic, month) with `volume`, `neg_share`,
  `avg_compound`, `share_of_coverage`, `severity`, and composite `intensity`.
- `output/<type>_risks_trends_topics.csv` — one row per topic with `PEAK_MONTH`,
  `PEAK_INTENSITY`, `TREND_SLOPE`, `MOMENTUM`, and the `IS_BENIGN` flag.

The console prints the topics with the strongest recent momentum — the emerging signals
worth a closer look.

### Interactive report

Build a standalone HTML explorer (embeds its data; no server or internet needed):

```powershell
# picks up the topic + trend CSVs for the risk type automatically
python build_report.py --risk-type enterprise
python build_report.py --risk-type emerging
```

The report has two tabs:

- **Topics** — searchable, filterable topic cards (by risk, sentiment) with descriptions,
  keyword / entity chips, and drill-in to the underlying articles.
- **Trends over time** — per-risk line charts of top topics' monthly signal intensity, a
  "strongest rising signals" callout, and a toggle to hide benign topics.

Pass `--trends <csv>` (or `--topics/--summary`) to point at specific files, or omit trends
to build a topics-only report.

## License

MIT — see [LICENSE](LICENSE).
