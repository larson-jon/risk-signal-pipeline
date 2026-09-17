# TOPIC TRENDS - month-over-month risk signal intensity
#
# The pipeline clusters the *entire* article history once (topic_clustering.py)
# so topic IDs are stable across time. This script then bins each topic's
# articles by calendar month and computes a composite "signal intensity" score
# per topic per month, so we can track whether a risk-related theme is growing
# or fading - and surface emerging aspects of risk that the current taxonomy
# may not fully capture.
#
# Intensity score (per topic, per month):
#     volume       = number of articles for the topic that month
#     neg_share    = fraction of those articles labeled Negative
#     avg_compound = mean VADER compound score that month (-1..+1)
#     severity     = 1 + neg_share + max(0, -avg_compound)      (>= 1)
#     intensity    = volume * severity
# So a negative surge scores higher than a neutral one of equal volume, while
# intensity never falls below raw volume. We also emit raw volume and
# share-of-coverage so the score stays interpretable.
#
# Benign / low-signal topics can optionally be dropped (--drop-benign): the
# outlier cluster is always excluded, and topics that are overwhelmingly
# neutral and thinly spread are flagged as low relevance.
#
# Usage:
#   python topic_trends.py --risk-type enterprise
#   python topic_trends.py --topics output/enterprise_risks_topics.csv \
#                          --out output/enterprise_risks_trends.csv

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RISK_TYPE_CONFIG = {
    "enterprise": {
        "topics": "output/enterprise_risks_topics.csv",
        "out": "output/enterprise_risks_trends.csv",
    },
    "emerging": {
        "topics": "output/emerging_risks_topics.csv",
        "out": "output/emerging_risks_trends.csv",
    },
}


def compute_severity(neg_share, avg_compound):
    """Sentiment weight (>= 1). Grows with negative share and negative tone."""
    return 1.0 + float(neg_share) + max(0.0, -float(avg_compound))


def flag_benign(topic_rows, total_months):
    """Heuristic: is this topic low-signal / benign?

    A topic is flagged benign if it is almost never negative AND its coverage
    is thin and flat (low peak intensity). These are the boilerplate clusters
    (weather, earnings-transcript templates, state-name lists) that don't
    represent a real risk signal.
    """
    neg_share_overall = topic_rows["neg_articles"].sum() / max(1, topic_rows["volume"].sum())
    peak_volume = topic_rows["volume"].max()
    mostly_neutral = neg_share_overall < 0.08
    thin = peak_volume < 5
    return bool(mostly_neutral and thin)


def build_trends(articles_df, drop_benign=False):
    """Return (trends_df, topic_meta_df).

    trends_df: one row per (topic, month) with volume, sentiment, intensity.
    topic_meta_df: one row per topic with label/description, risks, benign flag,
    peak month, and the overall + recent trend direction.
    """
    df = articles_df.copy()

    # Parse months; drop rows we can't place in time.
    df["_date"] = pd.to_datetime(df.get("PUBLISHED_DATE"), errors="coerce")
    df = df.dropna(subset=["_date"])
    df["month"] = df["_date"].dt.to_period("M").astype(str)

    # Always exclude the outlier cluster from trend analysis.
    df = df[df["TOPIC_ID"] != -1]
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df["_neg"] = (df.get("SENTIMENT", "").astype(str) == "Negative").astype(int)
    df["_compound"] = pd.to_numeric(df.get("SENTIMENT_COMPOUND"), errors="coerce").fillna(0.0)

    # Total articles per month (for share-of-coverage).
    month_totals = df.groupby("month").size().rename("month_total")

    all_months = sorted(df["month"].unique())

    # ---- per (topic, month) aggregation ----
    grp = df.groupby(["TOPIC_ID", "month"])
    agg = grp.agg(
        volume=("TOPIC_ID", "size"),
        neg_articles=("_neg", "sum"),
        avg_compound=("_compound", "mean"),
    ).reset_index()

    agg = agg.merge(month_totals, on="month", how="left")
    agg["neg_share"] = agg["neg_articles"] / agg["volume"]
    agg["share_of_coverage"] = agg["volume"] / agg["month_total"]
    agg["severity"] = agg.apply(
        lambda r: compute_severity(r["neg_share"], r["avg_compound"]), axis=1
    )
    agg["intensity"] = (agg["volume"] * agg["severity"]).round(2)
    agg["avg_compound"] = agg["avg_compound"].round(4)
    agg["neg_share"] = agg["neg_share"].round(4)
    agg["share_of_coverage"] = agg["share_of_coverage"].round(5)

    # ---- per-topic metadata ----
    # First non-null label/description/risk set for each topic.
    meta_rows = []
    for tid, g in df.groupby("TOPIC_ID"):
        topic_series = agg[agg["TOPIC_ID"] == tid].set_index("month").reindex(all_months)
        intensity = topic_series["intensity"].fillna(0.0)
        volume = topic_series["volume"].fillna(0.0)

        # trend direction: slope of intensity over month index (least squares).
        x = np.arange(len(all_months))
        slope = float(np.polyfit(x, intensity.values, 1)[0]) if len(all_months) > 1 else 0.0

        # recent momentum: last 3 months vs prior 3 months.
        recent = intensity.values[-3:].mean() if len(intensity) >= 1 else 0.0
        prior = intensity.values[-6:-3].mean() if len(intensity) >= 6 else intensity.values[:-3].mean() if len(intensity) > 3 else 0.0
        momentum = float(recent - prior)

        peak_idx = int(np.argmax(intensity.values)) if len(intensity) else 0
        peak_month = all_months[peak_idx] if all_months else ""

        risk_ids = sorted({str(r) for r in g.get("RISK_ID", pd.Series(dtype=str)).dropna()},
                          key=lambda v: (len(v), v))

        benign = flag_benign(
            agg[agg["TOPIC_ID"] == tid][["volume", "neg_articles"]], len(all_months)
        )

        meta_rows.append({
            "TOPIC_ID": tid,
            "TOPIC_LABEL": _first(g, "TOPIC_LABEL"),
            "TOPIC_DESCRIPTION": _first(g, "TOPIC_DESCRIPTION"),
            "RISK_IDS": ", ".join(risk_ids),
            "TOTAL_ARTICLES": int(volume.sum()),
            "PEAK_MONTH": peak_month,
            "PEAK_INTENSITY": round(float(intensity.max()), 2) if len(intensity) else 0.0,
            "MEAN_INTENSITY": round(float(intensity.mean()), 2) if len(intensity) else 0.0,
            "TREND_SLOPE": round(slope, 3),
            "MOMENTUM": round(momentum, 2),
            "IS_BENIGN": benign,
        })

    trends_df = agg[[
        "TOPIC_ID", "month", "volume", "neg_articles", "neg_share",
        "avg_compound", "share_of_coverage", "severity", "intensity",
    ]].sort_values(["TOPIC_ID", "month"]).reset_index(drop=True)

    topic_meta_df = pd.DataFrame(meta_rows).sort_values(
        "PEAK_INTENSITY", ascending=False
    ).reset_index(drop=True)

    # ---- per (risk, topic, month) breakdown ----
    # Intensity computed from ONLY that risk's articles, so each risk's charts
    # reflect the topics that are most intense *for that risk specifically*.
    risk_topic_df = _risk_topic_breakdown(df)

    if drop_benign:
        keep = set(topic_meta_df[~topic_meta_df["IS_BENIGN"]]["TOPIC_ID"])
        trends_df = trends_df[trends_df["TOPIC_ID"].isin(keep)]
        topic_meta_df = topic_meta_df[topic_meta_df["TOPIC_ID"].isin(keep)]
        risk_topic_df = risk_topic_df[risk_topic_df["TOPIC_ID"].isin(keep)]

    return trends_df, topic_meta_df, risk_topic_df


def _risk_topic_breakdown(df):
    """Intensity per (RISK_ID, TOPIC_ID, month) using only that risk's articles.

    Returns a long-format frame plus per-(risk,topic) peak/momentum so the
    report can rank each risk's own most intense topics.
    """
    d = df.dropna(subset=["RISK_ID"]).copy()
    if d.empty:
        return pd.DataFrame(columns=[
            "RISK_ID", "TOPIC_ID", "month", "volume", "intensity",
            "peak_intensity", "momentum",
        ])
    d["RISK_ID"] = d["RISK_ID"].apply(
        lambda v: str(int(v)) if str(v).replace(".0", "").isdigit() else str(v)
    ).str.replace(r"\.0$", "", regex=True)

    all_months = sorted(d["month"].unique())
    grp = d.groupby(["RISK_ID", "TOPIC_ID", "month"])
    agg = grp.agg(
        volume=("TOPIC_ID", "size"),
        neg_articles=("_neg", "sum"),
        avg_compound=("_compound", "mean"),
    ).reset_index()
    agg["neg_share"] = agg["neg_articles"] / agg["volume"]
    agg["severity"] = agg.apply(
        lambda r: compute_severity(r["neg_share"], r["avg_compound"]), axis=1
    )
    agg["intensity"] = (agg["volume"] * agg["severity"]).round(2)

    # Per (risk, topic): peak intensity and momentum (last 3 vs prior 3 months).
    rows = []
    for (rid, tid), g in agg.groupby(["RISK_ID", "TOPIC_ID"]):
        series = g.set_index("month")["intensity"].reindex(all_months).fillna(0.0)
        vals = series.values
        recent = vals[-3:].mean() if len(vals) else 0.0
        prior = vals[-6:-3].mean() if len(vals) >= 6 else (vals[:-3].mean() if len(vals) > 3 else 0.0)
        rows.append({
            "RISK_ID": rid,
            "TOPIC_ID": tid,
            "peak_intensity": round(float(vals.max()), 2) if len(vals) else 0.0,
            "momentum": round(float(recent - prior), 2),
        })
    stat = pd.DataFrame(rows)

    out = agg[["RISK_ID", "TOPIC_ID", "month", "volume", "intensity"]].merge(
        stat, on=["RISK_ID", "TOPIC_ID"], how="left"
    )
    return out.sort_values(["RISK_ID", "TOPIC_ID", "month"]).reset_index(drop=True)


def _first(group, col):
    """First non-empty value of a column in a group, or ''."""
    if col not in group:
        return ""
    vals = group[col].dropna()
    for v in vals:
        if str(v).strip():
            return str(v)
    return ""


def run(topics_path, out_path, drop_benign=False):
    topics_path = Path(topics_path)
    if not topics_path.exists():
        print(f"ERROR: topics file not found: {topics_path}")
        print("Run topic_clustering.py (full history, no --days) first.")
        sys.exit(1)

    df = pd.read_csv(topics_path)
    print(f"Loaded {len(df)} articles from {topics_path}")

    trends_df, meta_df, risk_topic_df = build_trends(df, drop_benign=drop_benign)
    if trends_df.empty:
        print("No datable, non-outlier articles to build trends from.")
        sys.exit(0)

    out_path = Path(out_path)
    out_path.parent.mkdir(exist_ok=True)
    trends_df.to_csv(out_path, index=False, encoding="utf-8")

    meta_out = out_path.with_name(out_path.stem + "_topics.csv")
    meta_df.to_csv(meta_out, index=False, encoding="utf-8")

    risk_out = out_path.with_name(out_path.stem + "_by_risk.csv")
    risk_topic_df.to_csv(risk_out, index=False, encoding="utf-8")

    n_months = trends_df["month"].nunique()
    n_topics = trends_df["TOPIC_ID"].nunique()
    n_benign = int(meta_df["IS_BENIGN"].sum())
    print(f"Built trends: {n_topics} topics over {n_months} months.")
    print(f"  {n_benign} topic(s) flagged benign"
          f"{' (dropped)' if drop_benign else ' (kept; use --drop-benign to exclude)'}.")
    print(f"Wrote month-by-topic trends -> {out_path}")
    print(f"Wrote per-topic trend summary -> {meta_out}")
    print(f"Wrote per-(risk,topic) breakdown -> {risk_out}")

    # Show the topics with the strongest recent momentum (emerging signals).
    rising = meta_df[~meta_df["IS_BENIGN"]].sort_values("MOMENTUM", ascending=False).head(10)
    if not rising.empty:
        print("\nTop rising signals (recent momentum):")
        for _, r in rising.iterrows():
            label = (r["TOPIC_LABEL"] or f"Topic {r['TOPIC_ID']}")[:50]
            print(f"  +{r['MOMENTUM']:.1f}  [{r['TOPIC_ID']}] {label} "
                  f"(peak {r['PEAK_MONTH']}, risks {r['RISK_IDS']})")


def parse_args():
    p = argparse.ArgumentParser(description="Month-over-month risk topic intensity trends.")
    p.add_argument("--risk-type", choices=["enterprise", "emerging"])
    p.add_argument("--topics", help="Per-article topics CSV (full history).")
    p.add_argument("--out", help="Output trends CSV path.")
    p.add_argument("--drop-benign", action="store_true",
                   help="Exclude topics flagged as benign / low-signal.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.risk_type:
        cfg = RISK_TYPE_CONFIG[args.risk_type]
        topics = args.topics or cfg["topics"]
        out = args.out or cfg["out"]
    elif args.topics:
        topics = args.topics
        out = args.out or str(Path(topics).with_name(Path(topics).stem + "_trends.csv"))
    else:
        print("ERROR: provide --risk-type or --topics.")
        sys.exit(1)

    print("#" * 50)
    print("Risk Topic Trends (monthly signal intensity)")
    print(f"Topics: {topics}")
    print("#" * 50)
    run(topics, out, drop_benign=args.drop_benign)
    print("Done.")


if __name__ == "__main__":
    main()
