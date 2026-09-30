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
# Max distinct-story articles to export per topic for the report's drill-down.
ARTICLES_PER_TOPIC = 25

# FINRA-domain anchor: phrases describing FINRA's actual remit (securities
# regulation, brokers, markets, investors, fraud, oversight). A cluster's
# FINRA_RELEVANCE is its cosine similarity to the mean of these — high means the
# theme sits inside FINRA's world, regardless of whether it's novel. Used to
# separate genuinely on-domain emerging risks from novel-but-off-topic noise
# (e.g. sports/astrology, which score high novelty but low relevance).
FINRA_ANCHOR_PHRASES = [
    "securities regulation", "broker-dealer oversight", "investor protection",
    "market integrity", "financial fraud", "securities fraud",
    "market manipulation", "insider trading", "capital markets",
    "brokerage firm compliance", "financial regulation enforcement",
    "stock market volatility", "investment adviser misconduct",
    "anti-money laundering", "trading surveillance",
]


def load_finra_anchor(model):
    """Return a single unit vector representing FINRA's regulatory domain,
    or None if it can't be built."""
    from embedding_cache import encode_cached
    try:
        vecs = encode_cached(model, FINRA_ANCHOR_PHRASES, model_name=EMBEDDING_MODEL,
                             verbose=False)
    except Exception:
        return None
    if vecs is None or len(vecs) == 0:
        return None
    anchor = vecs.mean(axis=0)
    return anchor / (np.linalg.norm(anchor) + 1e-9)


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

    finra_anchor = load_finra_anchor(model)
    if finra_anchor is None:
        print("NOTE: FINRA anchor unavailable; FINRA_RELEVANCE will be 0.")
    else:
        print(f"Scoring FINRA-relevance against a {len(FINRA_ANCHOR_PHRASES)}-phrase domain anchor.")

    # Dates for recency.
    df["_date"] = pd.to_datetime(df.get("PUBLISHED_DATE"), errors="coerce")
    df["month"] = df["_date"].dt.to_period("M").astype(str)
    all_months = sorted(m for m in df["month"].dropna().unique())
    recent_cut = set(all_months[-NEW_WINDOW_MONTHS:]) if all_months else set()

    from embedding_cache import encode_cached

    rows = []
    article_rows = []  # per-article export so the report can list a topic's stories
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

        # FINRA-relevance: closeness to the regulatory-domain anchor. Clamp the
        # small-negative cosines that can occur to 0 so the axis reads 0..1.
        if finra_anchor is not None:
            finra_relevance = round(max(0.0, float(centroid @ finra_anchor)), 4)
        else:
            finra_relevance = 0.0

        distinct = int(g["DUP_GROUP"].nunique())
        avg_quality = round(float(g["QUALITY_SCORE"].mean()), 3)
        months_present = sorted(g["month"].dropna().unique())
        first_seen = months_present[0] if months_present else ""
        is_new = first_seen in recent_cut
        recency = 1.3 if is_new else 1.0

        score = round(novelty * float(np.log1p(distinct)) * avg_quality * recency, 4)

        # Per-article export: one row per DISTINCT story (best-quality
        # representative of each dup group), so the report can list a topic's
        # articles with links without showing syndicated reprints. Capped per
        # topic to keep the embedded report JSON manageable.
        seen_groups = set()
        art_sorted = g.sort_values("QUALITY_SCORE", ascending=False)
        for _, a in art_sorted.iterrows():
            grp = a.get("DUP_GROUP")
            if grp in seen_groups:
                continue
            seen_groups.add(grp)
            link = str(a.get("LINK", "") or "").strip()
            title = str(a.get("TITLE", "") or "").strip()
            if not title and not link:
                continue
            article_rows.append({
                "TOPIC_ID": tid,
                "TITLE": title,
                "LINK": link,
                "SOURCE": str(a.get("SOURCE", "") or "").strip(),
                "SENTIMENT": str(a.get("SENTIMENT", "") or "").strip(),
                "PUBLISHED_DATE": str(a.get("PUBLISHED_DATE", "") or "")[:10],
            })
            if len(seen_groups) >= ARTICLES_PER_TOPIC:
                break

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
            "FINRA_RELEVANCE": finra_relevance,
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

    # Per-article drill-down file, keyed by TOPIC_ID (only topics that survived
    # the quality filter, so the report doesn't reference dropped topics).
    kept_topics = set(ranked["TOPIC_ID"].tolist())
    art_df = pd.DataFrame(article_rows)
    if not art_df.empty:
        art_df = art_df[art_df["TOPIC_ID"].isin(kept_topics)]
    articles_path = out_path.with_name("discovery_topic_articles.csv")
    art_df.to_csv(articles_path, index=False, encoding="utf-8")
    print(f"Wrote per-topic articles ({len(art_df)} rows) -> {articles_path}")

    print("\nTop candidate UNKNOWN risks (far from every existing risk):")
    for _, r in ranked.head(15).iterrows():
        nf = " [NEW]" if r["IS_NEW"] else ""
        print(f"  novelty {r['NOVELTY']:.2f} finra {r['FINRA_RELEVANCE']:.2f} "
              f"score {r['DISCOVERY_SCORE']:.2f}{nf}  "
              f"{str(r['TOPIC_TITLE'])[:42]}  "
              f"(nearest: {r['NEAREST_RISK']} @ {r['NEAREST_RISK_SIM']:.2f})")


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
