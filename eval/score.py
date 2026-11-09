"""Step 3: score dense / BM25 / hybrid against your blind relevance labels.

    ./.venv/bin/python eval/score.py

Judgments use the TREC Deep Learning 0-3 passage scale (eval/ANNOTATION_GUIDELINES.md).
Metrics at depth 5 (every top-5 result is judged, so nothing is guessed):
MRR@5, Hit@5, Precision@5, Recall@5 with TREC binarization (3,2 relevant),
graded nDCG@5 (linear gain), plus a lenient (>=1) sensitivity check.
Reports overall, by question type, and bootstrap 95% CIs.
Writes eval/v2_retrieval_results.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_DIR = Path(__file__).resolve().parent
K = 5
# Guidelines v1.1 recency rule: for questions about "recent" periods, a passage
# filed before this cutoff (12 months before the corpus end, 2026-11-04) can be
# at most grade 2. Applied programmatically to every judgment.
RECENCY_CUTOFF = "2025-11-04"
METHODS = ("dense", "bm25", "hybrid_rrf")


def per_query_metrics(ranking, grades, threshold, k=K):
    """grades: chunk_id -> 0..3 (TREC DL). Binary metrics count grade >= threshold;
    nDCG uses linear graded gain (trec_eval convention) against the ideal ordering."""
    relevant = {c for c, g in grades.items() if g >= threshold}
    top = ranking[:k]
    hits = [1 if c in relevant else 0 for c in top]
    first = next((i for i, h in enumerate(hits) if h), None)
    dcg = sum(grades.get(c, 0) / math.log2(i + 2) for i, c in enumerate(top))
    ideal_gains = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))
    return {
        "mrr": 0.0 if first is None else 1.0 / (first + 1),
        "hit": float(first is not None),
        "precision": sum(hits) / k,
        "recall": (sum(hits) / len(relevant)) if relevant else 0.0,
        "ndcg": (dcg / idcg) if idcg else 0.0,
    }


def bootstrap_ci(values, n=10000, seed=7):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 3), round(float(np.percentile(means, 97.5)), 3)]


def main():
    questions = {json.loads(l)["qid"]: json.loads(l)
                 for l in (EVAL_DIR / "v2_questions.jsonl").read_text().splitlines() if l.strip()}
    runs = json.loads((EVAL_DIR / "v2_runs.json").read_text())
    pool = pd.read_csv(EVAL_DIR / "v2_label_pool.csv", dtype=str)
    labels = pd.read_csv(EVAL_DIR / "v2_labels.csv", dtype={"qid": str, "chunk_id": str})

    missing = set(zip(pool.qid, pool.chunk_id)) - set(zip(labels.qid, labels.chunk_id))
    if missing:
        print(f"WARNING: {len(missing)} pool items are unlabeled; they count as not relevant.")

    adjudicated_path = EVAL_DIR / "v2_adjudicated.csv"
    adjudicated = 0
    if adjudicated_path.is_file():
        adj = pd.read_csv(adjudicated_path, dtype={"qid": str, "chunk_id": str})
        final = {(r.qid, r.chunk_id): int(r.final) for r in adj.itertuples()}
        labels = labels.copy()
        keys = list(zip(labels.qid, labels.chunk_id))
        labels["relevant"] = [final.get(k, v) for k, v in zip(keys, labels.relevant)]
        adjudicated = sum(k in final for k in keys)
        print(f"Adjudication: {adjudicated} first-pass labels replaced by adjudicated grades.")
    grades = {q: dict(zip(g.chunk_id, g.relevant.astype(int))) for q, g in labels.groupby("qid")}
    filing_date = dict(zip(pool.chunk_id, pool.filing_date))
    recency_qids = {q for q, meta in questions.items() if "recent" in meta.get("question", "").lower()}
    capped = 0
    for q, g in grades.items():
        if q in recency_qids:
            for chunk, grade in g.items():
                if grade == 3 and str(filing_date.get(chunk, "")) < RECENCY_CUTOFF:
                    g[chunk] = 2
                    capped += 1
    print(f"Recency rule (guidelines v1.1): {capped} grade-3 judgments capped to 2 "
          f"across {len(recency_qids)} 'recent' questions (filed before {RECENCY_CUTOFF}).")

    no_relevant = sorted(q for q in pool.qid.unique()
                         if not any(v >= 2 for v in grades.get(q, {}).values()))
    if no_relevant:
        print(f"Questions with no grade-2/3 passage in the pool (kept, score 0 on binary metrics): {no_relevant}")

    metrics = ["mrr", "hit", "precision", "recall", "ndcg"]
    results = {}
    for mode, threshold in (("official", 2), ("lenient", 1)):
        rows = []
        for run in runs:
            m = per_query_metrics(run["ranking"], grades.get(run["qid"], {}), threshold)
            rows.append({"qid": run["qid"], "method": run["method"],
                         "type": questions[run["qid"]]["type"], **m})
        df = pd.DataFrame(rows)
        overall = df.groupby("method")[metrics].mean().reindex(METHODS)
        by_type = df.pivot_table(index="type", columns="method", values="mrr", aggfunc="mean")[list(METHODS)]
        ci = {m: bootstrap_ci(df[df.method == m].sort_values("qid")["mrr"]) for m in METHODS}
        pivot = df.pivot(index="qid", columns="method", values="mrr")
        d, b, h = (overall.loc[m, "mrr"] for m in ("dense", "bm25", "hybrid_rrf"))
        results[mode] = {
            "overall": overall, "by_type": by_type, "ci": ci,
            "lift_dense": (100 * (h - d) / d) if d else None,
            "lift_bm25": (100 * (h - b) / b) if b else None,
            "lift_dense_ci": bootstrap_ci(pivot["hybrid_rrf"] - pivot["dense"]),
            "lift_bm25_ci": bootstrap_ci(pivot["hybrid_rrf"] - pivot["bm25"]),
        }

    # Both-sides coverage: for comparison questions, does the top-k contain a
    # relevant (grade >= 2) passage for EVERY company in the question?
    chunk_ticker = dict(zip(pool.chunk_id, pool.ticker))
    coverage_rows = []
    for run in runs:
        q = questions[run["qid"]]
        if len(q["tickers"]) < 2:
            continue
        g = grades.get(run["qid"], {})
        covered = {chunk_ticker.get(c) for c in run["ranking"][:K] if g.get(c, 0) >= 2}
        coverage_rows.append({"qid": run["qid"], "method": run["method"],
                              "covered": all(t in covered for t in q["tickers"]),
                              "companies_covered": sum(t in covered for t in q["tickers"]),
                              "companies": len(q["tickers"])})
    coverage = pd.DataFrame(coverage_rows)

    latency = pd.read_csv(EVAL_DIR / "v2_retrieval_latency.csv")
    lat = latency.groupby("method")["seconds"].quantile([0.5, 0.95]).unstack().reindex(METHODS)

    counts = labels.relevant.value_counts().to_dict()
    print(f"\n{labels.qid.nunique()} questions judged | grades: 3={counts.get(3, 0)} 2={counts.get(2, 0)} 1={counts.get(1, 0)} 0={counts.get(0, 0)}")
    for mode, r in results.items():
        label = ("TREC DL binarization: grades 3,2 relevant" if mode == "official"
                 else "sensitivity check: grades 3,2,1 relevant")
        print(f"\n=== {mode.upper()} @ {K}: {label}; nDCG is graded in both ===")
        print(r["overall"].round(3).to_string())
        print("MRR 95% bootstrap CI:", r["ci"])
        if r["lift_dense"] is not None:
            print(f"hybrid MRR vs dense: {r['lift_dense']:+.1f}%  (per-question diff CI {r['lift_dense_ci']})")
        if r["lift_bm25"] is not None:
            print(f"hybrid MRR vs BM25:  {r['lift_bm25']:+.1f}%  (per-question diff CI {r['lift_bm25_ci']})")
        print(f"MRR@{K} by question type:")
        print(r["by_type"].round(3).to_string())
    if len(coverage):
        print(f"\n=== Both-sides coverage @ {K} (comparison questions: a grade-2/3 passage for every company) ===")
        summary = coverage.groupby("method").agg(
            questions_fully_covered=("covered", "sum"), questions=("covered", "size"))
        print(summary.reindex(METHODS).to_string())
        print("per question:")
        print(coverage.pivot(index="qid", columns="method", values="companies_covered")
              .reindex(columns=list(METHODS)).astype(str).add("/2").to_string())
    print("\n=== Retrieval latency (s, in-process) ===")
    print(lat.rename(columns={0.5: "p50", 0.95: "p95"}).round(3).to_string())

    out = {
        "k": K, "questions": int(labels.qid.nunique()), "judged": int(len(labels)),
        "grade_counts": {str(k): int(v) for k, v in counts.items()},
        "latency_seconds": lat.round(4).to_dict(orient="index"),
        "zero_relevant_questions": no_relevant,
        "guidelines_version": "1.1",
        "recency_cutoff": RECENCY_CUTOFF,
        "recency_capped_judgments": capped,
        "adjudicated_judgments": adjudicated,
    }
    if len(coverage):
        out["both_sides_coverage"] = {
            m: {"fully_covered": int(coverage[coverage.method == m].covered.sum()),
                "questions": int((coverage.method == m).sum())} for m in METHODS}
    for mode, r in results.items():
        out[mode] = {
            "overall": r["overall"].round(4).to_dict(orient="index"),
            "mrr_ci95": r["ci"],
            "hybrid_lift_vs_dense_pct": None if r["lift_dense"] is None else round(r["lift_dense"], 2),
            "hybrid_lift_vs_bm25_pct": None if r["lift_bm25"] is None else round(r["lift_bm25"], 2),
            "mrr_by_type": r["by_type"].round(4).to_dict(orient="index"),
        }
    (EVAL_DIR / "v2_retrieval_results.json").write_text(json.dumps(out, indent=2))
    print("\nSaved eval/v2_retrieval_results.json")


if __name__ == "__main__":
    main()
