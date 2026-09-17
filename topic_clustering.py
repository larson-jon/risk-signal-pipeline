# TOPIC CLUSTERING - Stage 4 (with Stage 3 spaCy enrichment)
#
# Reads the sentiment CSV produced by news_sentiment_scraper.py, enriches
# each article with spaCy (named entities + noun-phrase keywords), embeds
# the text with sentence-transformers, and clusters the articles into topics
# with BERTopic. Writes:
#
#   1. A per-article file with the assigned TOPIC_ID / TOPIC_LABEL plus the
#      spaCy entities and keywords.
#   2. A per-topic summary that shows which RISK_IDs cluster together, the
#      dominant sentiment, and the topic's top keywords.
#
# Usage:
#   python topic_clustering.py --risk-type enterprise
#   python topic_clustering.py --risk-type emerging
#   python topic_clustering.py --input output/enterprise_risks_online_sentiment.csv

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# SSL setup - mirrors news_sentiment_scraper so huggingface / sentence-
# transformers model downloads work in the corporate (FINRA) environment.
# ---------------------------------------------------------------------------
def setup_ssl_verification():
    cert_path = Path(__file__).parent / "combined-certs.pem"
    if os.getenv("GITHUB_ACTIONS"):
        print("Running in GitHub Actions - using default SSL verification")
        return True
    if cert_path.exists():
        print(f"Using corporate certificate: {cert_path}")
        cert = str(cert_path)
        # sentence-transformers downloads models over https via requests/hf_hub
        os.environ["REQUESTS_CA_BUNDLE"] = cert
        os.environ["SSL_CERT_FILE"] = cert
        os.environ["CURL_CA_BUNDLE"] = cert
        return cert
    print("No certificate found - relying on default SSL verification")
    return True


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# Default embedding model. Can be a sentence-transformers hub name or a local
# folder path. See resolve_embedding_model() for how a local copy is discovered.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Candidate local model folders, checked in order. A local copy lets the stage
# run when huggingface.co is blocked or returning 503s (as happens on some
# corporate networks). Drop a full sentence-transformers model folder at one of
# these paths, or set EMBEDDING_MODEL / LOCAL_EMBEDDING_MODEL to an explicit path.
_LOCAL_MODEL_CANDIDATES = [
    os.getenv("LOCAL_EMBEDDING_MODEL"),
    str(Path(__file__).parent / "all-MiniLM-L6-v2-local"),
    str(Path(__file__).parent / "models" / "all-MiniLM-L6-v2-local"),
]


def _is_model_folder(path):
    """A usable sentence-transformers folder has a modules.json manifest."""
    try:
        return path and Path(path).is_dir() and (Path(path) / "modules.json").exists()
    except OSError:
        return False


def resolve_embedding_model():
    """Return the embedding model to load, preferring a local copy.

    Resolution order:
      1. EMBEDDING_MODEL if it points at an existing local model folder.
      2. The first existing folder in _LOCAL_MODEL_CANDIDATES.
      3. EMBEDDING_MODEL as-is (treated as a hub name, downloaded on demand).

    When a local folder is used we force Hugging Face offline mode so the load
    never blocks on a network call.
    """
    if _is_model_folder(EMBEDDING_MODEL):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print(f"Using local embedding model: {EMBEDDING_MODEL}")
        return EMBEDDING_MODEL

    for candidate in _LOCAL_MODEL_CANDIDATES:
        if _is_model_folder(candidate):
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            print(f"Using local embedding model: {candidate}")
            return candidate

    print(
        f"Using embedding model from Hugging Face hub: '{EMBEDDING_MODEL}'.\n"
        "  If huggingface.co is blocked, place a local model folder at "
        "'all-MiniLM-L6-v2-local' or set LOCAL_EMBEDDING_MODEL."
    )
    return EMBEDDING_MODEL

RISK_TYPE_CONFIG = {
    "enterprise": {
        "input": "output/enterprise_risks_online_sentiment.csv",
        "articles_out": "output/enterprise_risks_topics.csv",
        "summary_out": "output/enterprise_risks_topic_summary.csv",
    },
    "emerging": {
        "input": "output/emerging_risks_online_sentiment.csv",
        "articles_out": "output/emerging_risks_topics.csv",
        "summary_out": "output/emerging_risks_topic_summary.csv",
    },
}


def _safe_print(text):
    """Print text even if the console encoding can't represent every char.

    Windows consoles often default to cp1252, which chokes on non-ASCII
    characters that legitimately appear in topic labels (accented names,
    etc.). The CSV files are written UTF-8 regardless; this only guards the
    convenience console output so it never crashes the run.
    """
    try:
        print(text)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        print(text.encode(enc, errors="replace").decode(enc, errors="replace"))


def build_documents(df):
    """Combine title + summary into a single text per article for embedding."""
    titles = df["TITLE"].fillna("").astype(str)
    summaries = df["SUMMARY"].fillna("").astype(str) if "SUMMARY" in df else ""
    docs = (titles + ". " + summaries).str.strip()
    # Collapse whitespace
    docs = docs.str.replace(r"\s+", " ", regex=True).str.strip()
    return docs.tolist()


def enrich_with_spacy(df):
    """Add ENTITIES and NLP_KEYWORDS columns using the spaCy module."""
    from spacy_enrichment import enrich_batch

    titles = df["TITLE"].fillna("").astype(str)
    summaries = df["SUMMARY"].fillna("").astype(str) if "SUMMARY" in df else ""
    texts = (titles + ". " + summaries).tolist()

    entity_col, keyword_col = [], []
    for enriched in enrich_batch(texts):
        entity_col.append(enriched["entity_str"])
        keyword_col.append(enriched["keyword_str"])

    df = df.copy()
    df["ENTITIES"] = entity_col
    df["NLP_KEYWORDS"] = keyword_col
    return df


def make_topic_model(n_docs):
    """Create a BERTopic model with parameters that adapt to dataset size.

    HDBSCAN and UMAP defaults assume a fairly large corpus. For the small
    sample CSVs committed to the repo we shrink neighbor / cluster sizes so
    the pipeline still produces topics instead of erroring or marking every
    article as an outlier.
    """
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer

    embedding_model = SentenceTransformer(resolve_embedding_model())

    # Vectorizer for the c-TF-IDF topic keywords: drop English stop words and
    # single-character tokens so topic labels stay meaningful.
    vectorizer_model = CountVectorizer(
        stop_words="english",
        min_df=1,
        ngram_range=(1, 2),
    )

    # Scale clustering granularity to corpus size.
    min_topic_size = max(2, min(10, n_docs // 20))

    common = dict(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
        verbose=DEBUG_MODE,
    )

    # For small corpora, configure UMAP + HDBSCAN explicitly so they don't
    # fail on too few points.
    if n_docs < 200:
        from umap import UMAP
        from hdbscan import HDBSCAN

        n_neighbors = max(2, min(15, n_docs - 1))
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=min(5, max(2, n_docs - 2)),
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            min_samples=1,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        return BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, **common)

    return BERTopic(**common)


def topic_label(topic_model, topic_id):
    """Build a short human-readable label from a topic's top keywords."""
    if topic_id == -1:
        return "Outlier / Unclustered"
    words = topic_model.get_topic(topic_id)
    if not words:
        return f"Topic {topic_id}"
    top = [w for w, _ in words[:4] if w]
    return " / ".join(top) if top else f"Topic {topic_id}"


def dominant(series):
    """Return the most common non-null value in a series (ties broken by count)."""
    vals = series.dropna().tolist()
    if not vals:
        return ""
    return Counter(vals).most_common(1)[0][0]


def summarize_topics(df, topic_model):
    """Build a per-topic summary keyed back to the risk IDs."""
    rows = []
    for topic_id, group in df.groupby("TOPIC_ID"):
        risk_ids = sorted(
            str(r) for r in group["RISK_ID"].dropna().unique()
        )
        # top keywords from the c-TF-IDF model
        words = topic_model.get_topic(topic_id) if topic_id != -1 else []
        top_keywords = ", ".join(w for w, _ in words[:8] if w) if words else ""

        # most common entities across the cluster
        entity_counter = Counter()
        for cell in group.get("ENTITIES", pd.Series(dtype=str)).dropna():
            for ent in str(cell).split(", "):
                ent = ent.strip()
                if ent:
                    entity_counter[ent] += 1
        top_entities = ", ".join(e for e, _ in entity_counter.most_common(8))

        rows.append({
            "TOPIC_ID": topic_id,
            "TOPIC_LABEL": topic_label(topic_model, topic_id),
            "ARTICLE_COUNT": len(group),
            "RISK_IDS": ", ".join(risk_ids),
            "DISTINCT_RISK_COUNT": len(risk_ids),
            "DOMINANT_SENTIMENT": dominant(group.get("SENTIMENT", pd.Series(dtype=str))),
            "AVG_SENTIMENT_COMPOUND": round(
                pd.to_numeric(
                    group.get("SENTIMENT_COMPOUND", pd.Series(dtype=float)),
                    errors="coerce",
                ).mean(),
                4,
            ),
            "TOP_KEYWORDS": top_keywords,
            "TOP_ENTITIES": top_entities,
        })

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    # Real topics first (by size), outliers last.
    summary["_sort"] = summary["TOPIC_ID"].apply(lambda t: (t == -1, -1))
    summary = summary.sort_values(
        by=["_sort", "ARTICLE_COUNT"], ascending=[True, False]
    ).drop(columns="_sort")
    return summary


def run(input_path, articles_out, summary_out):
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}")
        print("Run news_sentiment_scraper.py first to generate it.")
        sys.exit(1)

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} articles from {input_path}")

    if df.empty:
        print("Input has no articles - nothing to cluster.")
        sys.exit(0)

    # Stage 3 - spaCy enrichment
    print("Enriching articles with spaCy (entities + noun-phrase keywords)...")
    df = enrich_with_spacy(df)

    # Stage 4 - embed + cluster
    docs = build_documents(df)
    n_docs = len(docs)

    if n_docs < 3:
        print(f"Only {n_docs} article(s) - too few to cluster meaningfully.")
        df["TOPIC_ID"] = -1
        df["TOPIC_LABEL"] = "Outlier / Unclustered"
    else:
        print(f"Embedding and clustering {n_docs} articles...")
        topic_model = make_topic_model(n_docs)
        topics, _ = topic_model.fit_transform(docs)
        df["TOPIC_ID"] = topics
        df["TOPIC_LABEL"] = [topic_label(topic_model, t) for t in topics]

        n_topics = len({t for t in topics if t != -1})
        n_outliers = sum(1 for t in topics if t == -1)
        print(f"Found {n_topics} topic(s); {n_outliers} article(s) left as outliers.")

    # ---- write per-article output ----
    Path(articles_out).parent.mkdir(exist_ok=True)
    df.to_csv(articles_out, index=False, encoding="utf-8")
    print(f"Wrote per-article topics -> {articles_out}")

    # ---- write per-topic summary ----
    if "TOPIC_ID" in df and n_docs >= 3:
        summary = summarize_topics(df, topic_model)
    else:
        # trivial summary when we couldn't cluster
        summary = summarize_topics(df, _DummyModel())
    summary.to_csv(summary_out, index=False, encoding="utf-8")
    print(f"Wrote topic summary -> {summary_out}")

    if not summary.empty:
        print("\nTopic overview (top 25 by article count):")
        cols = ["TOPIC_ID", "TOPIC_LABEL", "ARTICLE_COUNT", "DISTINCT_RISK_COUNT",
                "DOMINANT_SENTIMENT"]
        overview = summary[summary["TOPIC_ID"] != -1].head(25)
        if overview.empty:
            overview = summary
        _safe_print(overview[cols].to_string(index=False))


class _DummyModel:
    """Stand-in so summarize_topics works when clustering was skipped."""

    def get_topic(self, _topic_id):
        return []


def parse_args():
    p = argparse.ArgumentParser(
        description="Cluster risk news articles into topics with spaCy + BERTopic."
    )
    p.add_argument("--risk-type", choices=["enterprise", "emerging"])
    p.add_argument("--input", help="Path to a sentiment CSV (overrides --risk-type).")
    p.add_argument("--articles-out", help="Path for per-article topic CSV.")
    p.add_argument("--summary-out", help="Path for per-topic summary CSV.")
    return p.parse_args()


def main():
    setup_ssl_verification()
    args = parse_args()

    if args.input:
        input_path = args.input
        articles_out = args.articles_out or str(
            Path(input_path).with_name(Path(input_path).stem + "_topics.csv")
        )
        summary_out = args.summary_out or str(
            Path(input_path).with_name(Path(input_path).stem + "_topic_summary.csv")
        )
    elif args.risk_type:
        cfg = RISK_TYPE_CONFIG[args.risk_type]
        input_path = cfg["input"]
        articles_out = args.articles_out or cfg["articles_out"]
        summary_out = args.summary_out or cfg["summary_out"]
    else:
        print("ERROR: provide --risk-type {enterprise|emerging} or --input <csv>")
        sys.exit(1)

    print("#" * 50)
    print("Risk News Topic Clustering (spaCy + sentence-transformers + BERTopic)")
    print(f"Input: {input_path}")
    print("#" * 50)

    run(input_path, articles_out, summary_out)
    print("#" * 50)
    print("Done.")


if __name__ == "__main__":
    main()
