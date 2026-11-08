"""Step 2 (runs in the background while you label): answer all 50 questions
end to end and measure citation validity, abstention behaviour and latency.

Paced for Groq's free tier (~8k tokens/min): one question every ~35 s, with a
longer back-off after rate limits. Resumable: finished questions are skipped.

    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
    FINANCE_RAG_RUNTIME_DIR=artifacts/builds/runtime-v2-mvp50/runtime \
    ./.venv/bin/python eval/run_generation.py

Writes eval/v2_generation.jsonl (one line per question) and prints a summary.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from finance_rag.app_service import FinanceRAGService  # noqa: E402

EVAL_DIR = ROOT / "eval"
OUT = EVAL_DIR / "v2_generation.jsonl"
PACE_SECONDS = float(os.getenv("EVAL_PACE_SECONDS", "35"))


def summarize():
    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    answered = [r for r in answerable if r["status"] == "ok" and not r["abstained"]]
    ok = [r for r in rows if r["status"] == "ok"]
    lat = sorted(r["seconds"] for r in ok)

    def pct(n, d):
        return f"{100 * n / d:.1f}% ({n}/{d})" if d else "n/a"

    print("\n=== Generation summary ===")
    print("completed without error:      ", pct(len(ok), len(rows)))
    print("answerable -> answered:        ", pct(len(answered), len(answerable)))
    print("citation validity (answered):  ", pct(sum(r["citations_valid"] for r in answered), len(answered)))
    print("unanswerable -> abstained:     ", pct(sum(r["status"] == "ok" and r["abstained"] for r in unanswerable), len(unanswerable)))
    print("answerable -> wrongly abstained:", pct(sum(r["status"] == "ok" and r["abstained"] for r in answerable), len(answerable)))
    if lat:
        print(f"end-to-end latency p50 {lat[len(lat) // 2]:.1f}s | p95 {lat[int(0.95 * (len(lat) - 1))]:.1f}s")


def main():
    questions = [json.loads(l) for l in (EVAL_DIR / "v2_questions.jsonl").read_text().splitlines() if l.strip()]
    done = set()
    if OUT.is_file():
        done = {json.loads(l)["qid"] for l in OUT.read_text().splitlines()
                if l.strip() and json.loads(l)["status"] == "ok"}

    service = FinanceRAGService(project_dir=ROOT)
    todo = [q for q in questions if q["qid"] not in done]
    print(f"{len(done)} done, {len(todo)} to go (~{len(todo) * PACE_SECONDS / 60:.0f} min)", flush=True)

    for i, q in enumerate(todo, 1):
        ticker = q["tickers"] if len(q["tickers"]) > 1 else q["tickers"][0]
        record = {"qid": q["qid"], "type": q["type"], "answerable": q["answerable"]}
        t0 = time.perf_counter()
        try:
            result = service.search(q["question"], ticker=ticker, top_k=5).to_dict()
            cv = result.get("citation_validation") or {}
            record.update({
                "status": "ok" if not result.get("error") else "error",
                "error": result.get("error"),
                "abstained": bool(result.get("abstained")),
                "citations_valid": bool(cv.get("citations_valid")),
                "unique_cited": cv.get("unique_cited_labels", []),
                "answer": result.get("answer", ""),
                "abstention_reason": result.get("abstention_reason"),
                "sources": [s.get("chunk_id") for s in result.get("sources", [])],
            })
        except Exception as exc:  # recorded, then retried on the next run
            record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"[:300]})
        record["seconds"] = round(time.perf_counter() - t0, 2)

        with OUT.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"[{i}/{len(todo)}] {q['qid']} {record['status']}"
              f"{' abstained' if record.get('abstained') else ''} ({record['seconds']}s)", flush=True)

        wait = PACE_SECONDS * (3 if "RateLimit" in str(record.get("error")) else 1)
        if i < len(todo):
            time.sleep(wait)

    # keep only the latest record per question
    latest = {}
    for line in OUT.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            latest[r["qid"]] = r
    OUT.write_text("".join(json.dumps(r) + "\n" for r in latest.values()))
    summarize()


if __name__ == "__main__":
    main()
