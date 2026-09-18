# TOPIC GAPS - novelty / taxonomy-fit scoring  (PROTOTYPE)
#
# The ultimate goal of this pipeline is to surface *aspects of risk that the
# current taxonomy does not account for*. Intensity alone can't do that: every
# topic is tied to a risk simply because it was surfaced by that risk's search
# terms. A topic can be intense yet only loosely related to the risk that
# found it - that loose fit is the real signal of an uncovered risk aspect.
#
# This script scores each topic on:
#
#   fit    = max cosine similarity between the topic's centroid embedding and
#            the embeddings of its assigned risks' search terms (0..1).
#   gap    = 1 - fit   (high = topic doesn't semantically match its risk)
#   first_seen  = earliest month the topic appears
#   is_new      = first_seen falls within the last NEW_WINDOW_MONTHS of data
#   emerging_gap_score = gap * log1p(peak_intensity) * recency_boost
#
# The output ranks topics that are intense, recent, AND poorly explained by the
# existing risk definitions - candidate blind spots in the risk taxonomy.
#
# Usage:
#   python topic_gaps.py --risk-type enterprise
#   python topic_gaps.py --risk-type emerging

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the offline-aware model resolver and risk-term decoder from the
# clustering stage so embeddings live in the same vector space.
from topic_clustering import resolve_embedding_model, load_risk_map, EMBEDDING_MODEL

# Cache namespace - must match the name topic_clustering uses so both stages
# share the same on-disk embedding cache.
_EMB_NAME = EMBEDDING_MODEL

# A topic is "new" if it first appears within this many months of the data end.
NEW_WINDOW_MONTHS = 3
# Cap how many articles we embed per topic when computing a centroid (speed).
CENTROID_SAMPLE = 40

RISK_TYPE_CONFIG = {
    "enterprise": {
        "topics": "output/enterprise_risks_topics.csv",
        "encoded": "data/EnterpriseRisksListEncoded.csv",
        "risk_id_col": "ENTERPRISE_RISK_ID",
        "out": "output/enterprise_risks_gaps.csv",
    },
    "emerging": {
        "topics": "output/emerging_risks_topics.csv",
        "encoded": "data/EmergingRisksListEncoded.csv",
        "risk_id_col": "EMERGING_RISK_ID",
        "out": "output/emerging_risks_gaps.csv",
    },
}


def _doc_text(row):
    title = str(row.get("TITLE", "") or "")
    summary = str(row.get("SUMMARY", "") or "")
    return (title + ". " + summary).strip()


def _cos_sim_matrix(a, b):
    """Rows of a vs rows of b, assuming both are L2-normalized -> dot product."""
    return a @ b.T


def build_gaps(topics_df, risk_map, model, min_quality=0.0):
    """Compute per-topic fit / gap / novelty scores.

    Returns a DataFrame ranked by emerging_gap_score (desc). Topics whose mean
    QUALITY_SCORE is below `min_quality` are dropped so boilerplate (press
    releases, digests, transcripts) doesn't dominate the taxonomy-gap ranking.
    """
    df = topics_df.copy()
    df = df[df["TOPIC_ID"] != -1]  # ignore the outlier bucket

    # Ensure a usable quality column. Older topic CSVs either lack it or carry
    # the legacy all-zero placeholder from the original scraper, so recompute
    # whenever the column is missing or has no signal (constant / all zero).
    existing = pd.to_numeric(df.get("QUALITY_SCORE"), errors="coerce") if "QUALITY_SCORE" in df else None
    needs_scoring = existing is None or existing.fillna(0).nunique() <= 1
    if needs_scoring:
        try:
            from content_quality import score_frame
            df["QUALITY_SCORE"] = score_frame(df)
            print("Computed QUALITY_SCORE on the fly (CSV had none/placeholder).")
        except Exception:
            df["QUALITY_SCORE"] = 1.0
    df["QUALITY_SCORE"] = pd.to_numeric(df["QUALITY_SCORE"], errors="coerce").fillna(0.0)

    df["_date"] = pd.to_datetime(df.get("PUBLISHED_DATE"), errors="coerce")
    df["month"] = df["_date"].dt.to_period("M").astype(str)
    all_months = sorted(m for m in df["month"].dropna().unique())
    if not all_months:
        print("WARNING: no parseable dates; novelty scoring will be limited.")
    recent_cutoff = set(all_months[-NEW_WINDOW_MONTHS:]) if all_months else set()

    # ---- embed each risk's search terms (one per-risk representative vector) ----
    # We reduce each risk to a single mean term-vector so a topic's similarity
    # to every risk is directly comparable; this is what calibration needs.
    from embedding_cache import encode_cached

    risk_ids = sorted(risk_map.keys(), key=lambda v: (len(v), v))
    all_terms, term_owner = [], []
    for rid in risk_ids:
        for t in risk_map[rid]["terms"]:
            all_terms.append(t)
            term_owner.append(rid)
    term_vecs = encode_cached(model, all_terms, model_name=_EMB_NAME, verbose=False)

    risk_vec = {}          # rid -> mean term vector (normalized)
    risk_term_vecs = {}    # rid -> matrix of its term vectors (for best-term sim)
    for rid in risk_ids:
        idx = [i for i, o in enumerate(term_owner) if o == rid]
        if not idx:
            continue
        mat = term_vecs[idx]
        risk_term_vecs[rid] = mat
        mv = mat.mean(axis=0)
        risk_vec[rid] = mv / (np.linalg.norm(mv) + 1e-9)
    ordered_rids = [r for r in risk_ids if r in risk_vec]
    risk_matrix = np.vstack([risk_vec[r] for r in ordered_rids]) if ordered_rids else None
    print(f"Embedded search terms for {len(risk_vec)} risks.")

    # ---- topic centroids (embedded via the shared cache) ----
    topic_ids = sorted(df["TOPIC_ID"].unique())
    print(f"Scoring {len(topic_ids)} topics...")

    # Build all centroid docs first so the cache embeds them in one batch.
    centroid_docs, centroid_tids = [], []
    per_topic_texts = {}
    for tid in topic_ids:
        g = df[df["TOPIC_ID"] == tid]
        texts = [_doc_text(r) for _, r in g.head(CENTROID_SAMPLE).iterrows() if _doc_text(r)]
        if not texts:
            continue
        per_topic_texts[tid] = texts
        centroid_docs.extend(texts)
        centroid_tids.extend([tid] * len(texts))
    doc_vecs = encode_cached(model, centroid_docs, model_name=_EMB_NAME, verbose=True)

    # Aggregate per-topic centroids from the embedded docs.
    centroids = {}
    for tid in per_topic_texts:
        idx = [i for i, t in enumerate(centroid_tids) if t == tid]
        c = doc_vecs[idx].mean(axis=0)
        centroids[tid] = c / (np.linalg.norm(c) + 1e-9)

    rows = []
    for tid in topic_ids:
        if tid not in centroids:
            continue
        g = df[df["TOPIC_ID"] == tid]
        centroid = centroids[tid]

        topic_risks = sorted({str(r) for r in g["RISK_ID"].dropna()
                              .apply(lambda v: str(v).replace(".0", ""))})

        # Raw fit = best similarity to ANY term of ANY assigned risk.
        best_fit, best_risk = -1.0, ""
        for rid in topic_risks:
            tv = risk_term_vecs.get(rid)
            if tv is None:
                continue
            sim = float(np.max(centroid @ tv.T))
            if sim > best_fit:
                best_fit, best_risk = sim, rid
        best_fit = max(0.0, best_fit)
        gap_raw = round(1.0 - best_fit, 4)

        # --- CALIBRATION ---
        # Compare the topic's similarity to its assigned risk against its
        # similarity to ALL risks. If many risks match better, the topic fits
        # its taxonomy slot poorly *relative to the alternatives* - a stronger
        # signal than a low absolute cosine (which is inflated for every topic
        # because centroids and short phrases embed far apart).
        fit_percentile, gap_cal, better_risk, better_label = 1.0, 0.0, "", ""
        if risk_matrix is not None and topic_risks:
            all_sims = centroid @ risk_matrix.T           # sim to every risk
            assigned_idx = [i for i, r in enumerate(ordered_rids) if r in topic_risks]
            assigned_best = float(np.max(all_sims[assigned_idx])) if assigned_idx else 0.0
            # percentile of assigned-risk fit within the all-risk distribution
            fit_percentile = float(np.mean(all_sims <= assigned_best))
            gap_cal = round(1.0 - fit_percentile, 4)
            # which risk (if any) fits better than the assigned one
            top_i = int(np.argmax(all_sims))
            if ordered_rids[top_i] not in topic_risks:
                better_risk = ordered_rids[top_i]
                better_label = risk_map.get(better_risk, {}).get("label", "")

        # novelty
        months_present = sorted(g["month"].dropna().unique())
        first_seen = months_present[0] if months_present else ""
        is_new = first_seen in recent_cutoff
        recency_boost = 1.5 if is_new else 1.0

        # intensity proxy: peak monthly DISTINCT-STORY volume when available,
        # else peak monthly article volume.
        if "DUP_GROUP" in g and months_present:
            peak_volume = int(g.groupby("month")["DUP_GROUP"].nunique().max())
        elif months_present:
            peak_volume = int(g.groupby("month").size().max())
        else:
            peak_volume = len(g)
        avg_quality = round(float(g["QUALITY_SCORE"].mean()), 3)

        # Score now uses the CALIBRATED gap.
        emerging_gap = round(gap_cal * float(np.log1p(peak_volume)) * recency_boost, 4)

        rows.append({
            "TOPIC_ID": tid,
            "TOPIC_LABEL": _first(g, "TOPIC_LABEL"),
            "TOPIC_DESCRIPTION": _first(g, "TOPIC_DESCRIPTION"),
            "RISK_IDS": ", ".join(topic_risks),
            "BEST_FIT_RISK": best_risk,
            "BEST_FIT_RISK_LABEL": risk_map.get(best_risk, {}).get("label", ""),
            "FIT": round(best_fit, 4),
            "GAP_RAW": gap_raw,
            "FIT_PERCENTILE": round(fit_percentile, 4),
            "GAP": gap_cal,
            "BETTER_FIT_RISK": better_risk,
            "BETTER_FIT_RISK_LABEL": better_label,
            "FIRST_SEEN": first_seen,
            "IS_NEW": is_new,
            "PEAK_VOLUME": peak_volume,
            "TOTAL_ARTICLES": len(g),
            "AVG_QUALITY": avg_quality,
            "EMERGING_GAP_SCORE": emerging_gap,
        })

    out = pd.DataFrame(rows)
    n_before = len(out)
    if min_quality > 0:
        out = out[out["AVG_QUALITY"] >= min_quality]
        print(f"Quality filter (>= {min_quality}): kept {len(out)} of {n_before} topics.")
    out = out.sort_values("EMERGING_GAP_SCORE", ascending=False)
    return out.reset_index(drop=True)


def _first(group, col):
    if col not in group:
        return ""
    for v in group[col].dropna():
        if str(v).strip():
            return str(v)
    return ""


def run(topics_path, encoded_path, risk_id_col, out_path, min_quality=0.5):
    topics_path = Path(topics_path)
    if not topics_path.exists():
        print(f"ERROR: topics file not found: {topics_path}")
        print("Run topic_clustering.py (full history) first.")
        sys.exit(1)

    df = pd.read_csv(topics_path)
    print(f"Loaded {len(df)} articles from {topics_path}")

    risk_map = load_risk_map(encoded_path, risk_id_col)
    if not risk_map:
        print("ERROR: could not load risk terms; gap scoring needs them.")
        sys.exit(1)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(resolve_embedding_model())

    gaps = build_gaps(df, risk_map, model, min_quality=min_quality)
    if gaps.empty:
        print("No topics to score.")
        sys.exit(0)

    out_path = Path(out_path)
    out_path.parent.mkdir(exist_ok=True)
    gaps.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote gap scores -> {out_path}")

    top = gaps.head(12)
    print("\nTop candidate taxonomy gaps (calibrated gap: fits assigned risk poorly")
    print("relative to other risks; intense + recent weigh up the score):")
    for _, r in top.iterrows():
        newflag = " [NEW]" if r["IS_NEW"] else ""
        better = ""
        if str(r.get("BETTER_FIT_RISK", "")):
            better = f" -> fits R{r['BETTER_FIT_RISK']} ({str(r['BETTER_FIT_RISK_LABEL'])[:24]}) better"
        print(f"  gap {r['GAP']:.2f} score {r['EMERGING_GAP_SCORE']:.2f}{newflag}  "
              f"[{r['TOPIC_ID']}] {str(r['TOPIC_LABEL'])[:40]}{better}")


def parse_args():
    p = argparse.ArgumentParser(description="Score topics for taxonomy fit / novelty (prototype).")
    p.add_argument("--risk-type", choices=["enterprise", "emerging"])
    p.add_argument("--topics")
    p.add_argument("--encoded")
    p.add_argument("--risk-id-col", dest="risk_id_col")
    p.add_argument("--out")
    p.add_argument("--min-quality", type=float, default=0.5, dest="min_quality",
                   help="Drop topics whose mean QUALITY_SCORE is below this (default 0.5). "
                        "Set 0 to disable.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.risk_type:
        cfg = RISK_TYPE_CONFIG[args.risk_type]
        topics = args.topics or cfg["topics"]
        encoded = args.encoded or cfg["encoded"]
        risk_id_col = args.risk_id_col or cfg["risk_id_col"]
        out = args.out or cfg["out"]
    elif args.topics and args.encoded and args.risk_id_col:
        topics, encoded, risk_id_col = args.topics, args.encoded, args.risk_id_col
        out = args.out or str(Path(topics).with_name(Path(topics).stem + "_gaps.csv"))
    else:
        print("ERROR: provide --risk-type, or --topics + --encoded + --risk-id-col.")
        sys.exit(1)

    print("#" * 50)
    print("Risk Topic Gap / Novelty Scoring (prototype)")
    print(f"Topics: {topics}")
    print("#" * 50)
    run(topics, encoded, risk_id_col, out, min_quality=args.min_quality)
    print("Done.")


if __name__ == "__main__":
    main()
