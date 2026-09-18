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

# Cosine similarity at/above which two articles are treated as near-duplicates
# (syndicated wire stories). Tuned to catch reprints without merging distinct
# stories on the same subject.
DUP_SIMILARITY = float(os.getenv("DUP_SIMILARITY", "0.93"))

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
        "encoded": "data/EnterpriseRisksListEncoded.csv",
        "risk_id_col": "ENTERPRISE_RISK_ID",
    },
    "emerging": {
        "input": "output/emerging_risks_online_sentiment.csv",
        "articles_out": "output/emerging_risks_topics.csv",
        "summary_out": "output/emerging_risks_topic_summary.csv",
        "encoded": "data/EmergingRisksListEncoded.csv",
        "risk_id_col": "EMERGING_RISK_ID",
    },
}


def _with_suffix(path, suffix):
    """Insert `suffix` before the file extension (no-op if suffix is empty)."""
    if not suffix:
        return path
    p = Path(path)
    return str(p.with_name(p.stem + suffix + p.suffix))


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


def get_embedding_model():
    """Load the sentence-transformer once (offline-aware)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(resolve_embedding_model())


def embed_documents(docs, embedding_model):
    """Embed documents via the persistent cache (so re-runs are fast)."""
    from embedding_cache import encode_cached
    return encode_cached(embedding_model, docs, model_name=EMBEDDING_MODEL)


def dedup_documents(docs, embeddings, quality=None, threshold=DUP_SIMILARITY):
    """Group near-duplicate documents and pick one representative per group.

    Syndicated wire stories appear many times with near-identical text and
    inflate a topic's apparent volume. We collapse them: documents whose
    embeddings are >= `threshold` cosine similar (and share a cheap blocking
    key) are treated as one story for clustering. Clustering runs on the
    representatives only; the assigned topic is later propagated to every
    member of the group.

    To stay near-linear on tens of thousands of docs we only compare within
    blocks keyed by a normalized prefix of the document text, and also collapse
    exact-text duplicates directly.

    Returns
    -------
    group_of : list[int]   group id per input doc (same length as docs)
    reps : list[int]       representative doc index for each group id
    """
    import re

    n = len(docs)
    quality = quality if quality is not None else [1.0] * n

    def block_key(text):
        t = re.sub(r"[^a-z0-9 ]", "", (text or "").lower())
        t = re.sub(r"\s+", " ", t).strip()
        return t[:40]  # first ~40 chars of normalized text

    # Bucket doc indices by blocking key; exact-text dups share a key anyway.
    blocks = {}
    for i, d in enumerate(docs):
        blocks.setdefault(block_key(d), []).append(i)

    group_of = [-1] * n
    reps = []

    for _key, idxs in blocks.items():
        # Within a block, greedily assign each doc to an existing group whose
        # representative is similar enough, else start a new group.
        local_reps = []  # (group_id, rep_index)
        for i in idxs:
            assigned = None
            for gid, rep in local_reps:
                if float(embeddings[i] @ embeddings[rep]) >= threshold:
                    assigned = gid
                    # keep the higher-quality / longer doc as representative
                    if (quality[i], len(docs[i])) > (quality[rep], len(docs[rep])):
                        for j, (g2, _r) in enumerate(local_reps):
                            if g2 == gid:
                                local_reps[j] = (gid, i)
                        reps[gid] = i
                    break
            if assigned is None:
                gid = len(reps)
                reps.append(i)
                local_reps.append((gid, i))
                assigned = gid
            group_of[i] = assigned

    return group_of, reps


def make_topic_model(n_docs, embedding_model, nr_topics=None):
    """Create a BERTopic model with parameters that adapt to dataset size.

    HDBSCAN and UMAP defaults assume a fairly large corpus. For the small
    sample CSVs committed to the repo we shrink neighbor / cluster sizes so
    the pipeline still produces topics instead of erroring or marking every
    article as an outlier.

    `nr_topics` (int or "auto") is passed to BERTopic to merge the long tail of
    tiny topics into fewer, more interpretable themes.
    """
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer

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
        nr_topics=nr_topics,
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


def _distinct_keywords(words, limit=5):
    """Pick readable, non-redundant keywords from a topic's (word, score) list.

    BERTopic often returns overlapping n-grams (e.g. "ponzi", "ponzi scheme",
    "scheme"). We keep the most informative variant and drop terms that are
    substrings of an already-kept phrase, so the description reads cleanly.
    """
    kept = []
    for word, _ in words:
        w = (word or "").strip()
        if not w:
            continue
        wl = w.lower()
        # skip exact duplicates (case-insensitive)
        if any(wl == k.lower() for k in kept):
            continue
        # skip if this term overlaps a kept term as a sub/superstring
        redundant = any(wl in k.lower() or k.lower() in wl for k in kept)
        if redundant:
            # prefer the longer, more specific phrase
            for i, k in enumerate(kept):
                if k.lower() in wl and len(w) > len(k):
                    kept[i] = w
            continue
        kept.append(w)
        if len(kept) >= limit:
            break
    # Final guard: collapse any case-insensitive duplicates that the
    # substring-replacement step may have reintroduced, preserving order.
    seen, unique = set(), []
    for k in kept:
        if k.lower() not in seen:
            seen.add(k.lower())
            unique.append(k)
    return unique


def _humanize_keywords(keywords):
    """Join keywords into a natural phrase: 'a, b, c and d'."""
    if not keywords:
        return ""
    if len(keywords) == 1:
        return keywords[0]
    return ", ".join(keywords[:-1]) + " and " + keywords[-1]


def topic_description(topic_model, topic_id, group, rep_docs=None, risk_map=None):
    """Compose a unique, readable one-line description for a topic.

    Combines the topic's distinguishing keywords, its dominant sentiment, the
    risks it touches, and (when available) its most representative article
    title as a concrete example.
    """
    if topic_id == -1:
        return ("Articles that did not fit any coherent topic "
                f"({len(group)} unclustered).")

    words = topic_model.get_topic(topic_id) or []
    keywords = _distinct_keywords(words, limit=5)
    theme = _humanize_keywords(keywords) or f"topic {topic_id}"

    sentiment = (dominant(group.get("SENTIMENT", pd.Series(dtype=str))) or "mixed").lower()
    n = len(group)

    # Risk context, using descriptions when we have them.
    risk_map = risk_map or {}
    risk_ids = sorted({str(r) for r in group.get("RISK_ID", pd.Series(dtype=str)).dropna()},
                      key=lambda x: (len(x), x))
    if risk_ids:
        named = []
        for rid in risk_ids[:3]:
            desc = risk_map.get(rid, {}).get("label")
            named.append(f"{desc}" if desc else f"risk {rid}")
        risk_part = "; ".join(named)
        if len(risk_ids) > 3:
            risk_part += f"; +{len(risk_ids) - 3} more"
        risk_clause = f" Linked to {risk_part}."
    else:
        risk_clause = ""

    # A concrete example headline, if we have representative docs. Prefer a
    # headline that actually mentions one of the topic keywords, so the example
    # reflects the theme rather than an incidental representative doc.
    example = ""
    if rep_docs:
        heads = []
        for d in rep_docs:
            head = (d or "").strip().split(". ")[0].strip()
            if 15 <= len(head) <= 140:
                heads.append(head)
        kw_lower = [k.lower() for k in keywords]
        for head in heads:
            hl = head.lower()
            if any(k in hl for k in kw_lower):
                example = head
                break
        if not example:
            example = heads[0] if heads else (rep_docs[0] or "").strip()[:140]

    desc = f"Coverage of {theme}"
    desc += f", with mostly {sentiment} sentiment" if sentiment != "mixed" else ""
    desc += f" ({n} articles)."
    if example:
        desc += f' For example: "{example}."'
    desc += risk_clause
    return desc


def summarize_topics(df, topic_model, rep_docs_map=None, risk_map=None):
    """Build a per-topic summary keyed back to the risk IDs."""
    rep_docs_map = rep_docs_map or {}
    rows = []
    for topic_id, group in df.groupby("TOPIC_ID"):
        risk_ids = sorted(
            str(r) for r in group["RISK_ID"].dropna().unique()
        )
        # top keywords from the c-TF-IDF model
        words = topic_model.get_topic(topic_id) if topic_id != -1 else []
        top_keywords = ", ".join(w for w, _ in words[:8] if w) if words else ""

        description = topic_description(
            topic_model, topic_id, group,
            rep_docs=rep_docs_map.get(topic_id),
            risk_map=risk_map,
        )

        # most common entities across the cluster
        entity_counter = Counter()
        for cell in group.get("ENTITIES", pd.Series(dtype=str)).dropna():
            for ent in str(cell).split(", "):
                ent = ent.strip()
                if ent:
                    entity_counter[ent] += 1
        top_entities = ", ".join(e for e, _ in entity_counter.most_common(8))

        # Distinct stories = unique dup-groups (syndicated reprints collapsed).
        distinct_stories = (
            int(group["DUP_GROUP"].nunique()) if "DUP_GROUP" in group else len(group)
        )

        rows.append({
            "TOPIC_ID": topic_id,
            "TOPIC_LABEL": topic_label(topic_model, topic_id),
            "TOPIC_DESCRIPTION": description,
            "ARTICLE_COUNT": len(group),
            "DISTINCT_STORIES": distinct_stories,
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
            "AVG_QUALITY": round(
                pd.to_numeric(
                    group.get("QUALITY_SCORE", pd.Series(dtype=float)),
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


def _decode_term(term):
    """Decode a single encoded search term (same scheme as the scraper)."""
    try:
        n = int(term)
        byte_len = (n.bit_length() + 7) // 8
        return n.to_bytes(byte_len, byteorder="little").decode("utf-8")
    except (ValueError, UnicodeDecodeError, OverflowError):
        return None


def load_risk_map(encoded_path, risk_id_col):
    """Map risk ID -> {label, terms} by decoding the encoded search-term CSV.

    Used to phrase topic descriptions in terms of the risks they touch. Returns
    an empty map (falling back to bare IDs) if the file is missing.
    """
    risk_map = {}
    if not encoded_path:
        return risk_map
    path = Path(encoded_path)
    if not path.exists():
        return risk_map
    try:
        df = pd.read_csv(path)
    except Exception:
        return risk_map
    if risk_id_col not in df or "ENCODED_TERMS" not in df:
        return risk_map
    df["_decoded"] = df["ENCODED_TERMS"].apply(_decode_term)
    for rid, group in df.groupby(risk_id_col):
        terms = [t.strip().strip('"').strip()
                 for t in group["_decoded"].dropna().tolist() if t and t.strip()]
        if terms:
            risk_map[str(int(rid))] = {"label": terms[0], "terms": terms}
    return risk_map


def filter_recent(df, days):
    """Keep only articles published within `days` of the most recent article.

    We anchor the window to the latest PUBLISHED_DATE in the data (not today's
    date) so the filter behaves predictably even if the scraper hasn't run in a
    while. Rows with an unparseable date are dropped from the windowed view.
    """
    dates = pd.to_datetime(df["PUBLISHED_DATE"], errors="coerce")
    latest = dates.max()
    if pd.isna(latest):
        print("WARNING: no parseable PUBLISHED_DATE values; skipping date filter.")
        return df
    cutoff = latest - pd.Timedelta(days=days)
    mask = dates >= cutoff
    kept = df[mask].copy()
    print(
        f"Date filter: last {days} days "
        f"({cutoff.date()} to {latest.date()}) -> {len(kept)} of {len(df)} articles"
    )
    return kept


def run(input_path, articles_out, summary_out, days=None,
        encoded_path=None, risk_id_col=None, nr_topics=None):
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

    if days:
        df = filter_recent(df, days)
        if df.empty:
            print("No articles in the selected window - nothing to cluster.")
            sys.exit(0)

    risk_map = load_risk_map(encoded_path, risk_id_col)

    # Content quality - fill the QUALITY_SCORE column so downstream trend / gap
    # analysis can filter boilerplate (press releases, digests, transcripts).
    try:
        from content_quality import score_frame
        df["QUALITY_SCORE"] = score_frame(df)
        lowq = int((df["QUALITY_SCORE"] < 0.5).sum())
        print(f"Scored content quality: {lowq} of {len(df)} articles are low-quality "
              f"(<0.5).")
    except Exception as e:
        print(f"WARNING: content quality scoring failed ({e}); leaving QUALITY_SCORE as-is.")

    # Stage 3 - spaCy enrichment
    print("Enriching articles with spaCy (entities + noun-phrase keywords)...")
    df = enrich_with_spacy(df)

    # Stage 4 - embed + cluster
    docs = build_documents(df)
    n_docs = len(docs)
    rep_docs_map = {}

    # Defaults so the columns always exist.
    df["DUP_GROUP"] = range(n_docs)
    df["IS_DUPLICATE"] = False

    if n_docs < 3:
        print(f"Only {n_docs} article(s) - too few to cluster meaningfully.")
        df["TOPIC_ID"] = -1
        df["TOPIC_LABEL"] = "Outlier / Unclustered"
        topic_model = _DummyModel()
    else:
        embedding_model = get_embedding_model()
        print(f"Embedding {n_docs} articles (cached)...")
        embeddings = embed_documents(docs, embedding_model)

        # --- near-duplicate dedup ---
        quality = pd.to_numeric(df.get("QUALITY_SCORE"), errors="coerce").fillna(1.0).tolist()
        group_of, reps = dedup_documents(docs, embeddings, quality=quality)
        df["DUP_GROUP"] = group_of
        # First occurrence of each group is the kept representative; rest dup.
        df["IS_DUPLICATE"] = [gi != reps[g] for gi, g in zip(range(n_docs), group_of)]
        n_groups = len(reps)
        n_dupes = n_docs - n_groups
        print(f"Near-duplicate dedup: {n_docs} articles -> {n_groups} distinct "
              f"stories ({n_dupes} near-duplicates collapsed).")

        rep_docs = [docs[i] for i in reps]
        rep_embeddings = embeddings[reps]

        if len(rep_docs) < 3:
            print("Too few distinct stories to cluster; leaving all as outliers.")
            df["TOPIC_ID"] = -1
            df["TOPIC_LABEL"] = "Outlier / Unclustered"
            topic_model = _DummyModel()
        else:
            print(f"Clustering {len(rep_docs)} distinct stories...")
            topic_model = make_topic_model(len(rep_docs), embedding_model,
                                           nr_topics=nr_topics)
            rep_topics, _ = topic_model.fit_transform(rep_docs, embeddings=rep_embeddings)

            # Map group id -> topic, then propagate to every article.
            group_topic = {g: rep_topics[gi] for gi, g in enumerate(range(len(reps)))}
            topics = [int(group_topic[g]) for g in group_of]
            df["TOPIC_ID"] = topics
            df["TOPIC_LABEL"] = [topic_label(topic_model, t) for t in topics]

            try:
                rep_docs_map = topic_model.get_representative_docs()
            except Exception:
                rep_docs_map = {}

            n_topics = len({t for t in topics if t != -1})
            n_outliers = sum(1 for t in topics if t == -1)
            print(f"Found {n_topics} topic(s); {n_outliers} article(s) left as outliers.")

    # ---- per-topic summary (adds TOPIC_DESCRIPTION) ----
    summary = summarize_topics(df, topic_model, rep_docs_map=rep_docs_map,
                               risk_map=risk_map)

    # Attach the generated description back onto each article row too.
    desc_by_topic = dict(zip(summary["TOPIC_ID"], summary["TOPIC_DESCRIPTION"])) \
        if not summary.empty else {}
    df["TOPIC_DESCRIPTION"] = df["TOPIC_ID"].map(desc_by_topic).fillna("")

    # ---- write per-article output ----
    Path(articles_out).parent.mkdir(exist_ok=True)
    df.to_csv(articles_out, index=False, encoding="utf-8")
    print(f"Wrote per-article topics -> {articles_out}")

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

    def get_representative_docs(self, *_args, **_kwargs):
        return {}


def parse_args():
    p = argparse.ArgumentParser(
        description="Cluster risk news articles into topics with spaCy + BERTopic."
    )
    p.add_argument("--risk-type", choices=["enterprise", "emerging"])
    p.add_argument("--input", help="Path to a sentiment CSV (overrides --risk-type).")
    p.add_argument("--articles-out", help="Path for per-article topic CSV.")
    p.add_argument("--summary-out", help="Path for per-topic summary CSV.")
    p.add_argument("--days", type=int,
                   help="Only cluster articles from the last N days (anchored to "
                        "the most recent article date). Outputs get a _last<N>d suffix.")
    p.add_argument("--encoded", help="Encoded risk-terms CSV (for topic descriptions).")
    p.add_argument("--risk-id-col", dest="risk_id_col",
                   help="Risk ID column in the encoded CSV.")
    p.add_argument("--nr-topics", dest="nr_topics", default=None,
                   help="Reduce to this many topics after clustering: an integer "
                        "(e.g. 60) or 'auto' to let BERTopic merge similar topics. "
                        "Fewer, cleaner themes; omit to keep all topics.")
    return p.parse_args()


def main():
    setup_ssl_verification()
    args = parse_args()

    # Suffix for default output names when a date window is applied.
    suffix = f"_last{args.days}d" if args.days else ""

    encoded = risk_id_col = None
    if args.input:
        input_path = args.input
        default_articles = Path(input_path).with_name(
            Path(input_path).stem + f"_topics{suffix}.csv")
        default_summary = Path(input_path).with_name(
            Path(input_path).stem + f"_topic_summary{suffix}.csv")
        articles_out = args.articles_out or str(default_articles)
        summary_out = args.summary_out or str(default_summary)
        encoded = args.encoded
        risk_id_col = args.risk_id_col
    elif args.risk_type:
        cfg = RISK_TYPE_CONFIG[args.risk_type]
        input_path = cfg["input"]
        articles_out = args.articles_out or _with_suffix(cfg["articles_out"], suffix)
        summary_out = args.summary_out or _with_suffix(cfg["summary_out"], suffix)
        encoded = args.encoded or cfg["encoded"]
        risk_id_col = args.risk_id_col or cfg["risk_id_col"]
    else:
        print("ERROR: provide --risk-type {enterprise|emerging} or --input <csv>")
        sys.exit(1)

    # Parse --nr-topics: "auto", an integer, or None.
    nr_topics = args.nr_topics
    if nr_topics is not None and str(nr_topics).lower() != "auto":
        try:
            nr_topics = int(nr_topics)
        except ValueError:
            print(f"WARNING: --nr-topics '{nr_topics}' is not an int or 'auto'; ignoring.")
            nr_topics = None

    print("#" * 50)
    print("Risk News Topic Clustering (spaCy + sentence-transformers + BERTopic)")
    print(f"Input: {input_path}")
    if args.days:
        print(f"Window: last {args.days} days")
    if nr_topics:
        print(f"Topic reduction: nr_topics={nr_topics}")
    print("#" * 50)

    run(input_path, articles_out, summary_out, days=args.days,
        encoded_path=encoded, risk_id_col=risk_id_col, nr_topics=nr_topics)
    print("#" * 50)
    print("Done.")


if __name__ == "__main__":
    main()
