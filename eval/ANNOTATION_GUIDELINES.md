# FinanceRAG v2 — Relevance Annotation Guidelines

Version 1.1 · v1.0 fixed 2026-11-07 before labeling began; v1.1 rule added 2026-11-08 (see change log).

## Scale

Adopted from the TREC Deep Learning track passage-judgment scale (NIST, 2019–2023).

| Grade | Label | Definition (TREC DL) | Applied to SEC filings |
|---|---|---|---|
| **3** | Perfectly relevant | The passage is dedicated to the query and contains the exact answer. | The passage is *about* what was asked and states it directly: the risk factor itself, the named executive change, the requested figure for the requested company. |
| **2** | Highly relevant | The passage has some answer for the query, but the answer may be unclear or hidden amongst extraneous information. | The answer is present but partial, buried in a longer section, mixed with other topics, or requires reading across a table. |
| **1** | Related | The passage seems related to the query but does not answer it. | Same company and topic area, but no answer: a passing mention, a cross-reference ("see Item 1A"), a summary heading, the right metric for an irrelevant segment. |
| **0** | Irrelevant | The passage has nothing to do with the query. | Off topic, wrong company, cover page, boilerplate (forward-looking-statement warnings, signatures, exhibit indexes), table of contents. |

## Decision rules

1. Judge the passage **on its own**. Do not use outside knowledge to fill gaps.
2. **Comparison questions** (two companies): a passage that answers the question for *either* company is judged as if the question were about that company alone.
3. **"Recent" questions** ("recent quarters", "recent results"): any reporting period is acceptable for grades 1–2, but **grade 3 requires a filing dated within the 12 months before the corpus end** (on or after 2025-11-04). An older passage that otherwise answers the question is at most grade 2. *(v1.1)*
4. **8-K event questions**: grade 3 requires naming the event specifics (who left / who was appointed / what was completed). A passage that only lists the 8-K item number without specifics is at most 1.
5. **Tables**: a table row containing the requested figure counts as an answer (grade 2 or 3 depending on clarity).
6. **When torn between two grades, choose the lower one.**
7. Labels are given **blind**: the retrieval method that produced each passage is hidden and passages are shuffled within each question.

## Binarization (TREC DL convention)

- Binary metrics (MRR, Hit, Precision, Recall): grades **3 and 2 = relevant**, grades 1 and 0 = not relevant.
- nDCG uses the full 0–3 graded scale with linear gain (trec_eval convention).
- A lenient variant (grades ≥ 1 relevant) is reported only as a sensitivity check.

## Quality control

After all labels are collected, a random 10% sample (seed fixed) is re-labeled blind without access to the first labels. Agreement is reported as **linearly weighted Cohen's kappa** on the 0–3 scale and unweighted kappa on the binarized labels.

## Adjudication

Audit passages whose first-pass and audit grades differ are resolved in an unblinded adjudication pass (`eval/adjudicate_app.py`), with both grades visible, applying these guidelines. Adjudicated grades (`eval/v2_adjudicated.csv`, with a one-line reason) replace the first-pass grade for scoring. Agreement statistics are always reported on the original, pre-adjudication passes.

## Pooling

Top-5 results from each of three retrieval systems (dense BGE, BM25, hybrid RRF) per question, deduplicated. All pooled passages are judged; metrics are computed at depth 5, so no unjudged passages are scored.

## Change log

- **v1.1 (2026-11-08)** — Added the recency rule (decision rule 3). It was introduced after first-pass labeling, so it is applied **programmatically** to all judgments in `eval/score.py` (grade 3 → 2 for passages filed before 2025-11-04 on questions containing "recent"), identically for every retrieval method. Binary metrics are unaffected (grades 2 and 3 are both relevant); only graded nDCG changes. The audit kappa in `v2_agreement.json` was computed on raw labels.
- **v1.0 (2026-11-07)** — Initial guidelines, fixed before labeling.
