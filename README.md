# risk-signal-pipeline

A pipeline that turns news coverage into risk signals. It fetches news articles for a
set of enterprise and emerging risk search terms, scores their sentiment, enriches them
with spaCy NLP, and groups them into topics through clustering.

## Pipeline stages

1. **Fetch** — pull recent articles for each risk search term from the newsdata.io API,
   parse them with `newspaper3k`, and score sentiment with VADER.
2. **Sentiment** — each article is labeled Positive / Negative / Neutral with a compound
   score.
3. **spaCy NLP** *(planned)* — extract named entities (organizations, people, places) and
   cleaned keywords / noun phrases from article text.
4. **Topic clustering** *(planned)* — group articles into topics with BERTopic so recurring
   themes surface across the risk landscape.

> Stages 1–2 are implemented today in `news_sentiment_scraper.py`. Stages 3–4 are the
> next work items and their dependencies are already listed in `requirements.txt`.

## Project layout

```
.
├── news_sentiment_scraper.py   # fetch + sentiment (stages 1-2)
├── data/                       # encoded risk search-term lists
│   ├── EnterpriseRisksListEncoded.csv
│   └── EmergingRisksListEncoded.csv
├── output/                     # generated sentiment CSVs
├── tests/                      # test suite
├── .github/workflows/          # CI / scheduled runs
└── requirements.txt
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

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

## License

MIT — see [LICENSE](LICENSE).
