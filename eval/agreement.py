"""Annotator consistency: compare the blind 10% audit re-labels with the originals.

    ./.venv/bin/python eval/agreement.py

Reports linearly weighted Cohen's kappa on the 0–3 scale and unweighted kappa
on TREC-binarized labels (3,2 = relevant). Writes eval/v2_agreement.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_DIR = Path(__file__).resolve().parent


def cohen_kappa(a, b, categories, weights=None):
    a, b = np.asarray(a), np.asarray(b)
    k = len(categories)
    index = {c: i for i, c in enumerate(categories)}
    observed = np.zeros((k, k))
    for x, y in zip(a, b):
        observed[index[x], index[y]] += 1
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    if weights == "linear":
        w = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
    else:
        w = 1 - np.eye(k)
    denominator = (w * expected).sum()
    return 1.0 if denominator == 0 else 1 - (w * observed).sum() / denominator


def interpret(kappa):
    # Landis & Koch (1977)
    for bound, word in ((0.8, "almost perfect"), (0.6, "substantial"), (0.4, "moderate"), (0.2, "fair"), (0.0, "slight")):
        if kappa > bound:
            return word
    return "poor"


def main():
    first = pd.read_csv(EVAL_DIR / "v2_labels.csv", dtype={"qid": str, "chunk_id": str})
    audit = pd.read_csv(EVAL_DIR / "v2_labels_audit.csv", dtype={"qid": str, "chunk_id": str})
    merged = first.merge(audit, on=["qid", "chunk_id"], suffixes=("_first", "_audit"))
    if merged.empty:
        raise SystemExit("No overlapping labels between v2_labels.csv and v2_labels_audit.csv")

    graded = cohen_kappa(merged.relevant_first, merged.relevant_audit, [0, 1, 2, 3], weights="linear")
    binary = cohen_kappa(merged.relevant_first >= 2, merged.relevant_audit >= 2, [False, True])
    exact = float((merged.relevant_first == merged.relevant_audit).mean())

    print(f"audit items: {len(merged)}")
    print(f"exact grade agreement:          {100 * exact:.1f}%")
    print(f"weighted kappa (0–3, linear):   {graded:.3f}  ({interpret(graded)})")
    print(f"kappa on binarized (3,2 vs 1,0): {binary:.3f}  ({interpret(binary)})")
    print("\nconfusion (rows = first pass, cols = audit):")
    print(pd.crosstab(merged.relevant_first, merged.relevant_audit).to_string())

    (EVAL_DIR / "v2_agreement.json").write_text(json.dumps({
        "audit_items": int(len(merged)), "exact_agreement": round(exact, 4),
        "weighted_kappa_graded": round(float(graded), 4), "kappa_binary": round(float(binary), 4),
        "interpretation_graded": interpret(graded), "interpretation_binary": interpret(binary),
    }, indent=2))
    print("\nSaved eval/v2_agreement.json")


if __name__ == "__main__":
    main()
