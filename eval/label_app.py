"""Blind relevance-labeling app (TREC Deep Learning 0–3 scale).

    ./.venv/bin/streamlit run eval/label_app.py            # main labeling
    ./.venv/bin/streamlit run eval/label_app.py -- audit   # 10% blind re-label for kappa

Every click is saved immediately, so you can stop and resume at any time.
The retrieval method behind each passage is hidden. Guidelines:
eval/ANNOTATION_GUIDELINES.md
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

EVAL_DIR = Path(__file__).resolve().parent
POOL = EVAL_DIR / "v2_label_pool.csv"
AUDIT_MODE = "audit" in sys.argv[1:]
LABELS = EVAL_DIR / ("v2_labels_audit.csv" if AUDIT_MODE else "v2_labels.csv")
AUDIT_FRACTION, AUDIT_SEED = 0.10, 20261108

GRADES = [
    (3, "3 · perfectly relevant", "dedicated to the question and states the exact answer"),
    (2, "2 · highly relevant", "contains an answer, but partial, buried, or mixed with other content"),
    (1, "1 · related", "on topic, but does not answer it (passing mention, cross-reference, heading)"),
    (0, "0 · irrelevant", "off topic, wrong company, boilerplate, cover page, table of contents"),
]

st.set_page_config(page_title="FinanceRAG v2 labeling", layout="wide")


@st.cache_data
def load_pool():
    pool = pd.read_csv(POOL, dtype=str).fillna("")
    if AUDIT_MODE:
        pool = pool.sample(frac=AUDIT_FRACTION, random_state=AUDIT_SEED).reset_index(drop=True)
    return pool


def load_labels():
    if LABELS.is_file():
        return pd.read_csv(LABELS, dtype={"qid": str, "chunk_id": str, "relevant": int})
    return pd.DataFrame(columns=["qid", "chunk_id", "relevant"])


def save_label(qid, chunk_id, grade):
    labels = load_labels()
    labels = labels[~((labels.qid == qid) & (labels.chunk_id == chunk_id))]
    labels = pd.concat([labels, pd.DataFrame([{"qid": qid, "chunk_id": chunk_id, "relevant": grade}])])
    labels.to_csv(LABELS, index=False)


pool = load_pool()
labels = load_labels()
if len(labels) and labels.relevant.max() > 3:
    st.error("Label file contains grades above 3 — check eval/v2_labels.csv")
    st.stop()
done = set(zip(labels.qid, labels.chunk_id))
remaining = [i for i, r in pool.iterrows() if (r.qid, r.chunk_id) not in done]

title = "AUDIT re-label (blind, 10% sample)" if AUDIT_MODE else "Relevance labeling"
st.progress(len(done) / len(pool), text=f"{title} — {len(done)} / {len(pool)} done ({len(pool) - len(done)} left)")

if not remaining:
    nxt = "./.venv/bin/python eval/score.py" if not AUDIT_MODE else "./.venv/bin/python eval/agreement.py"
    st.success(f"All done! Next: {nxt}")
    st.stop()

row = pool.loc[remaining[0]]
st.markdown(f"### {row.qid} · {row.question}")
st.caption(f"question type: {row.type}")
st.markdown(f"**Passage:** {row.ticker} · {row.form_type} · filed {row.filing_date} · "
            f"[SEC filing]({row.source_url})")
st.code(row.text, language=None, wrap_lines=True, height=340)

cols = st.columns(4)
for col, (grade, label, help_text) in zip(cols, GRADES):
    if col.button(label, use_container_width=True, type="primary" if grade == 3 else "secondary", help=help_text):
        save_label(row.qid, row.chunk_id, grade)
        st.rerun()
    col.caption(help_text)

st.caption("when torn between two grades, choose the lower one · comparisons: judge for either company · "
           "binary metrics count 3 and 2 as relevant (TREC DL)")
if st.button("↩ undo last label") and len(labels):
    labels.iloc[:-1].to_csv(LABELS, index=False)
    st.rerun()
