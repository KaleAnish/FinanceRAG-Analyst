"""Step 1 of the v2 evaluation: run every answerable question through dense,
BM25 and hybrid retrieval, save the rankings, and build a blind labeling pool.

Pool = union of each method's top-5 per question. Method names are NOT stored
in the pool file and items are shuffled, so labeling is blind to the method.

Run with the server STOPPED (it needs the memory):
    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
    FINANCE_RAG_RUNTIME_DIR=artifacts/builds/runtime-v2-mvp50/runtime \
    ./.venv/bin/python eval/build_pool.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from finance_rag.retrieval import FinanceRAGRetriever  # noqa: E402

METHODS = ("dense", "bm25", "hybrid_rrf")
RANK_DEPTH = 10   # rankings saved
POOL_DEPTH = 5    # judged depth
EVAL_DIR = ROOT / "eval"


def main():
    questions = [json.loads(l) for l in (EVAL_DIR / "v2_questions.jsonl").read_text().splitlines() if l.strip()]
    answerable = [q for q in questions if q["answerable"]]

    t = time.perf_counter()
    retriever = FinanceRAGRetriever()
    print(f"retriever loaded in {time.perf_counter() - t:.1f}s "
          f"({len(retriever.chunks_df):,} chunks)", flush=True)

    # warm-up so the first timed query is not penalized
    retriever.search("warm up", top_k=5)

    runs, pool_rows, latencies = [], {}, []
    for q in answerable:
        ticker = q["tickers"] if len(q["tickers"]) > 1 else q["tickers"][0]
        for method in METHODS:
            t0 = time.perf_counter()
            hits = retriever.search(q["question"], method=method, ticker=ticker, top_k=RANK_DEPTH)
            elapsed = time.perf_counter() - t0
            latencies.append({"qid": q["qid"], "method": method, "seconds": elapsed})
            ranking = hits["chunk_id"].tolist()
            runs.append({"qid": q["qid"], "method": method, "ranking": ranking})
            for _, row in hits.head(POOL_DEPTH).iterrows():
                key = (q["qid"], row["chunk_id"])
                if key not in pool_rows:
                    pool_rows[key] = {
                        "qid": q["qid"], "type": q["type"], "question": q["question"],
                        "chunk_id": row["chunk_id"], "ticker": row["ticker"],
                        "form_type": row["form_type"], "filing_date": row["filing_date"],
                        "source_url": row.get("source_url", ""), "text": row["text"],
                    }
        print(f"{q['qid']} pooled", flush=True)

    (EVAL_DIR / "v2_runs.json").write_text(json.dumps(runs, indent=1))
    pd.DataFrame(latencies).to_csv(EVAL_DIR / "v2_retrieval_latency.csv", index=False)

    pool = list(pool_rows.values())
    rng = random.Random(20261107)
    by_q = {}
    for row in pool:
        by_q.setdefault(row["qid"], []).append(row)
    ordered = []
    for q in answerable:  # keep question order, shuffle within a question (blind)
        items = by_q.get(q["qid"], [])
        rng.shuffle(items)
        ordered.extend(items)
    pd.DataFrame(ordered).to_csv(EVAL_DIR / "v2_label_pool.csv", index=False)

    lat = pd.DataFrame(latencies).groupby("method")["seconds"].describe(percentiles=[0.5, 0.95])
    print(f"\n{len(answerable)} questions | {len(ordered)} chunks to label "
          f"(avg {len(ordered) / len(answerable):.1f} per question)")
    print(lat[["50%", "95%", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
