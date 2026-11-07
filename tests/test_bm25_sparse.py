import random

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from finance_rag.bm25_sparse import SparseBM25Okapi
from finance_rag.retrieval import tokenize_for_bm25


def _corpus(seed=7, docs=400):
    rng = random.Random(seed)
    common = ["the", "company", "revenue", "risk", "and", "of"]
    rare = [f"term{i}" for i in range(300)]
    out = []
    for _ in range(docs):
        n = rng.randint(1, 60)
        out.append([rng.choice(common) if rng.random() < 0.5 else rng.choice(rare)
                    for _ in range(n)])
    return out


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_scores_are_bit_identical_to_rank_bm25(seed):
    corpus = _corpus(seed)
    reference = BM25Okapi(corpus)
    sparse = SparseBM25Okapi.build(corpus)
    queries = [
        ["revenue", "risk"], ["the", "the", "term5"], ["missing_token"],
        ["term1", "company", "term299", "of"], [],
    ]
    for query in queries:
        np.testing.assert_array_equal(sparse.get_scores(query), reference.get_scores(query))


def test_round_trip_save_load(tmp_path):
    corpus = [tokenize_for_bm25(t) for t in [
        "Apple revenue grew in fiscal 2025.",
        "NVIDIA data-center revenue and supply risk.",
        "Risk factors include export controls.",
    ]]
    built = SparseBM25Okapi.build(corpus)
    built.save(tmp_path)
    assert SparseBM25Okapi.exists(tmp_path)
    loaded = SparseBM25Okapi.load(tmp_path)
    query = tokenize_for_bm25("revenue risk")
    np.testing.assert_array_equal(loaded.get_scores(query), built.get_scores(query))
    np.testing.assert_array_equal(loaded.get_scores(query), BM25Okapi(corpus).get_scores(query))


def test_empty_corpus_rejected():
    with pytest.raises(ValueError):
        SparseBM25Okapi.build([])
