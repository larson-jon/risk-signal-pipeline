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

> All four stages are implemented. Stages 1–2 run in `news_sentiment_scraper.py`; stages 3–4
> run in `topic_clustering.py`, which reads the sentiment CSVs the scraper produces.

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

## Project layout

```
.
├── news_sentiment_scraper.py   # fetch + sentiment (stages 1-2)
├── spacy_enrichment.py         # spaCy entity + keyword enrichment (stage 3)
├── topic_clustering.py         # sentence-transformers + BERTopic (stages 3-4)
├── data/                       # encoded risk search-term lists
│   ├── EnterpriseRisksListEncoded.csv
│   └── EmergingRisksListEncoded.csv
├── output/                     # generated sentiment + topic CSVs
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
  `TOPIC_LABEL`, spaCy `ENTITIES`, and `NLP_KEYWORDS`.
- `output/<type>_risks_topic_summary.csv` — one row per topic showing article count,
  which `RISK_IDS` fall into it, the dominant sentiment, and the top keywords / entities.

The topic summary is the quickest way to see clusters of risk: topics spanning several
distinct `RISK_IDS` point to themes that cut across the risk landscape.

Set `EMBEDDING_MODEL` to swap the sentence-transformers model (defaults to
`all-MiniLM-L6-v2`).

## License

MIT — see [LICENSE](LICENSE).
