# RISK DISCOVERY SCORING  (Path B)
#
# Takes the broad, un-keyworded corpus from discover_fetch.py, clusters it, and
# ranks the resulting themes by how FAR they sit from every existing risk. A
# large, recent, good-quality cluster that is distant from all current risk
# definitions is a candidate risk the taxonomy does not yet cover.
#
# Distinct from topic_gaps.py: that tool measures how poorly a *risk-filtered*
# topic fits the risk that surfaced it. This tool operates on news that matched
# NO risk at all, and scores novelty as distance from the WHOLE taxonomy.
#
#   nearest_risk_sim = max cosine(cluster centroid, any risk's term-vector)
#   novelty          = 1 - nearest_risk_sim   (high = unlike every known risk)
#   discovery_score  = novelty * log1p(distinct_stories) * quality * recency
#
# Reuses the shared embedding cache, dedup, BERTopic model, title/description
# helpers, and the risk-term decoder from the existing pipeline.
#
# Usage:
#   python discover_score.py                       # output/discovery_news.csv
#   python discover_score.py --nr-topics 40 --min-quality 0.6
#   python discover_score.py --input path/to.csv --out output/discovery_ranked.csv

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from topic_clustering import (
    resolve_embedding_model, get_embedding_model, embed_documents,
    dedup_documents, make_topic_model, build_documents, enrich_with_spacy,
    topic_title, topic_description, load_risk_map, EMBEDDING_MODEL,
)

# All risk term files, so novelty is measured against BOTH taxonomies at once.
RISK_SOURCES = [
    ("data/EnterpriseRisksListEncoded.csv", "ENTERPRISE_RISK_ID", "ENT"),
    ("data/EmergingRisksListEncoded.csv", "EMERGING_RISK_ID", "EMG"),
]

NEW_WINDOW_MONTHS = 3
CENTROID_SAMPLE = 40


def load_all_risk_vectors(model):
    """Embed every risk's search terms from both taxonomies.

    Returns (risk_matrix, meta) where risk_matrix rows are per-risk mean term
    vectors and meta[i] = (namespace, risk_id, label).
    """
    from embedding_cache import encode_cached

    rows, meta = [], []
    for path, id_col, ns in RISK_SOURCES:
        rmap = load_risk_map(path, id_col)
        for rid, info in rmap.items():
            terms = info.get("terms") or []
            if not terms:
                continue
            vecs = encode_cached(model, terms, model_name=EMBEDDING_MODEL, verbose=False)
            mv = vecs.mean(axis=0)
            mv = mv / (np.linalg.norm(mv) + 1e-9)
            rows.append(mv)
            meta.append((ns, str(rid), info.get("label", "")))
    if not rows:
        return None, []
    return np.vstack(rows), meta


def run(input_path, out_path, nr_topics, min_quality, source="lexicon"):
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"ERROR: discovery corpus not found: {input_path}")
        print("Run discover_fetch.py first (needs NEWS_DATA_API_KEY).")
        sys.exit(1)

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} discovery articles from {input_path}")

    # Filter by fetch method. The risk-LEXICON stream (queries the language of
    # risk) yields far more risk-relevant themes than the broad CATEGORY sweep,
    # so it's the default. A source comparison showed category-only adds mostly
    # benign general news (sports, entertainment, weather).
    if source != "all" and "SEARCH_TERM_ID" in df.columns:
        tag = "LEXICON" if source == "lexicon" else "DISCOVERY"
        before = len(df)
        df = df[df["SEARCH_TERM_ID"] == tag].copy()
        print(f"Source filter '{source}': kept {len(df)} of {before} articles "
              f"(SEARCH_TERM_ID == {tag}).")
    elif source != "all":
        print(f"NOTE: no SEARCH_TERM_ID column; using all {len(df)} articles.")

    if len(df) < 5:
        print("Too few articles to cluster meaningfully.")
        sys.exit(0)

    # Quality (recompute if missing / placeholder).
    q = pd.to_numeric(df.get("QUALITY_SCORE"), errors="coerce") if "QUALITY_SCORE" in df else None
    if q is None or q.fillna(0).nunique() <= 1:
        try:
            from content_quality import score_frame
            df["QUALITY_SCORE"] = score_frame(df)
        except Exception:
            df["QUALITY_SCORE"] = 1.0
    df["QUALITY_SCORE"] = pd.to_numeric(df["QUALITY_SCORE"], errors="coerce").fillna(0.0)

    # spaCy enrichment (keeps titles/descriptions consistent with the pipeline).
    print("Enriching with spaCy...")
    df = enrich_with_spacy(df)

    docs = build_documents(df)
    model = get_embedding_model()
    print(f"Embedding {len(docs)} articles (cached)...")
    embeddings = embed_documents(docs, model)

    quality = df["QUALITY_SCORE"].tolist()
    group_of, reps = dedup_documents(docs, embeddings, quality=quality)
    df["DUP_GROUP"] = group_of
    n_stories = len(reps)
    print(f"Dedup: {len(docs)} articles -> {n_stories} distinct stories.")

    rep_docs = [docs[i] for i in reps]
    rep_emb = embeddings[reps]
    if len(rep_docs) < 5:
        print("Too few distinct stories to cluster.")
        sys.exit(0)

    print(f"Clustering {len(rep_docs)} stories...")
    topic_model = make_topic_model(len(rep_docs), model, nr_topics=nr_topics)
    rep_topics, _ = topic_model.fit_transform(rep_docs, embeddings=rep_emb)
    group_topic = {g: rep_topics[gi] for gi, g in enumerate(range(len(reps)))}
    df["TOPIC_ID"] = [int(group_topic[g]) for g in group_of]

    # Risk vectors for novelty scoring.
    risk_matrix, risk_meta = load_all_risk_vectors(model)
    if risk_matrix is None:
        print("ERROR: no risk term-vectors could be built.")
        sys.exit(1)
    print(f"Scoring novelty against {len(risk_meta)} risks from both taxonomies.")

    # Dates for recency.
    df["_date"] = pd.to_datetime(df.get("PUBLISHED_DATE"), errors="coerce")
    df["month"] = df["_date"].dt.to_period("M").astype(str)
    all_months = sorted(m for m in df["month"].dropna().unique())
    recent_cut = set(all_months[-NEW_WINDOW_MONTHS:]) if all_months else set()

    from embedding_cache import encode_cached

    rows = []
    used_titles = set()
    order = (df[df["TOPIC_ID"] != -1].groupby("TOPIC_ID").size()
             .sort_values(ascending=False).index.tolist())
    for tid in order:
        g = df[df["TOPIC_ID"] == tid]
        texts = [t for t in build_documents(g.head(CENTROID_SAMPLE)) if t]
        if not texts:
            continue
        cvecs = encode_cached(model, texts, model_name=EMBEDDING_MODEL, verbose=False)
        centroid = cvecs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)

        sims = centroid @ risk_matrix.T
        nearest_i = int(np.argmax(sims))
        nearest_sim = float(sims[nearest_i])
        novelty = round(1.0 - nearest_sim, 4)
        ns, rid, rlabel = risk_meta[nearest_i]

        distinct = int(g["DUP_GROUP"].nunique())
        avg_quality = round(float(g["QUALITY_SCORE"].mean()), 3)
        months_present = sorted(g["month"].dropna().unique())
        first_seen = months_present[0] if months_present else ""
        is_new = first_seen in recent_cut
        recency = 1.3 if is_new else 1.0

        score = round(novelty * float(np.log1p(distinct)) * avg_quality * recency, 4)

        rows.append({
            "TOPIC_ID": tid,
            "TOPIC_TITLE": topic_title(topic_model, tid, g, used=used_titles),
            "TOPIC_DESCRIPTION": topic_description(topic_model, tid, g),
            "DISTINCT_STORIES": distinct,
            "TOTAL_ARTICLES": len(g),
            "AVG_QUALITY": avg_quality,
            "FIRST_SEEN": first_seen,
            "IS_NEW": is_new,
            "NOVELTY": novelty,
            "NEAREST_RISK": f"{ns}:{rid}",
            "NEAREST_RISK_LABEL": rlabel,
            "NEAREST_RISK_SIM": round(nearest_sim, 4),
            "DISCOVERY_SCORE": score,
        })

    ranked = pd.DataFrame(rows).sort_values("DISCOVERY_SCORE", ascending=False)
    if min_quality > 0:
        before = len(ranked)
        ranked = ranked[ranked["AVG_QUALITY"] >= min_quality]
        print(f"Quality filter (>= {min_quality}): kept {len(ranked)} of {before} topics.")

    out_path = Path(out_path)
    out_path.parent.mkdir(exist_ok=True)
    ranked.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote ranked discovery topics -> {out_path}")

    print("\nTop candidate UNKNOWN risks (far from every existing risk):")
    for _, r in ranked.head(15).iterrows():
        nf = " [NEW]" if r["IS_NEW"] else ""
        print(f"  novelty {r['NOVELTY']:.2f} score {r['DISCOVERY_SCORE']:.2f}{nf}  "
              f"{str(r['TOPIC_TITLE'])[:46]}  "
              f"(nearest: {r['NEAREST_RISK']} {str(r['NEAREST_RISK_LABEL'])[:22]} "
              f"@ {r['NEAREST_RISK_SIM']:.2f})")


def parse_args():
    p = argparse.ArgumentParser(description="Rank broad-news themes by distance from all risks.")
    p.add_argument("--input", default="output/discovery_news.csv")
    p.add_argument("--out", default="output/discovery_ranked.csv")
    p.add_argument("--nr-topics", dest="nr_topics", default=40,
                   help="Reduce to this many topics (int or 'auto').")
    p.add_argument("--min-quality", type=float, default=0.6, dest="min_quality")
    p.add_argument("--source", choices=["lexicon", "category", "all"], default="lexicon",
                   help="Which fetched articles to score: 'lexicon' (risk-phrase "
                        "queries, default - most risk-relevant), 'category' (broad "
                        "sweep), or 'all'.")
    return p.parse_args()


def main():
    args = parse_args()
    nr_topics = args.nr_topics
    if nr_topics is not None and str(nr_topics).lower() != "auto":
        try:
            nr_topics = int(nr_topics)
        except ValueError:
            nr_topics = None
    print("#" * 60)
    print("RISK DISCOVERY SCORING (novelty vs. full taxonomy)")
    print(f"Source: {args.source}")
    print("#" * 60)
    run(args.input, args.out, nr_topics, args.min_quality, source=args.source)
    print("Done.")


if __name__ == "__main__":
    main()
