"""Adjudicate first-pass vs. audit disagreements (standard annotation practice).

    ./.venv/bin/streamlit run eval/adjudicate_app.py

Shows each audit passage where the two grades differ, with BOTH grades visible
(adjudication is deliberately not blind). Pick the final grade per
eval/ANNOTATION_GUIDELINES.md and optionally note why. Saved to
eval/v2_adjudicated.csv; eval/score.py uses these as the final labels.
Kappa (eval/agreement.py) is still computed on the original two passes.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

EVAL_DIR = Path(__file__).resolve().parent
OUT = EVAL_DIR / "v2_adjudicated.csv"
NAMES = {3: "3 · perfectly relevant", 2: "2 · highly relevant", 1: "1 · related", 0: "0 · irrelevant"}

st.set_page_config(page_title="FinanceRAG v2 adjudication", layout="wide")

first = pd.read_csv(EVAL_DIR / "v2_labels.csv", dtype={"qid": str, "chunk_id": str})
audit = pd.read_csv(EVAL_DIR / "v2_labels_audit.csv", dtype={"qid": str, "chunk_id": str})
pool = pd.read_csv(EVAL_DIR / "v2_label_pool.csv", dtype=str).fillna("")
merged = first.merge(audit, on=["qid", "chunk_id"], suffixes=("_first", "_audit"))
disagree = merged[merged.relevant_first != merged.relevant_audit].merge(pool, on=["qid", "chunk_id"])
disagree = disagree.sort_values(["qid", "chunk_id"]).reset_index(drop=True)

done = pd.read_csv(OUT, dtype={"qid": str, "chunk_id": str}) if OUT.is_file() else \
    pd.DataFrame(columns=["qid", "chunk_id", "first", "audit", "final", "note"])
done_keys = set(zip(done.qid, done.chunk_id))
todo = [i for i, r in disagree.iterrows() if (r.qid, r.chunk_id) not in done_keys]

st.progress(len(done_keys) / max(len(disagree), 1),
            text=f"Adjudication — {len(done_keys)} / {len(disagree)} disagreements resolved")
if not todo:
    st.success("All disagreements adjudicated. Next: ./.venv/bin/python eval/score.py")
    st.dataframe(done, use_container_width=True)
    st.stop()

row = disagree.loc[todo[0]]
st.markdown(f"### {row.qid} · {row.question}")
st.caption(f"question type: {row.type}")
st.markdown(f"**Passage:** {row.ticker} · {row.form_type} · filed {row.filing_date} · [SEC filing]({row.source_url})")
st.code(row.text, language=None, wrap_lines=True, height=320)
st.markdown(f"**First pass:** {NAMES[int(row.relevant_first)]} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"**Audit:** {NAMES[int(row.relevant_audit)]}")

note = st.text_input("Reason (optional, one line)", key=f"note-{row.qid}-{row.chunk_id}")
cols = st.columns(4)
for col, grade in zip(cols, (3, 2, 1, 0)):
    if col.button(f"final: {NAMES[grade]}", use_container_width=True):
        new = pd.DataFrame([{"qid": row.qid, "chunk_id": row.chunk_id, "first": int(row.relevant_first),
                             "audit": int(row.relevant_audit), "final": grade, "note": note}])
        pd.concat([done, new]).to_csv(OUT, index=False)
        st.rerun()
st.caption("apply the written guidelines (v1.1), including the recency rule and 'when torn, choose the lower grade'.")
