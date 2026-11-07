"""Prebuilt, sparse BM25Okapi with exact score parity to ``rank_bm25.BM25Okapi``.

Why: ``rank_bm25`` keeps one Python dict per document and scores every
document on every query in Python. That is fine for 4k chunks and unusable at
100k+. This implementation stores term postings as a CSR matrix
(term -> doc ids + term frequencies), so a query only touches the postings of
its own terms, and the index is built once by the processing worker and
loaded from disk by the API.

Parity: idf (including the epsilon floor and its insertion-order average),
document lengths, avgdl and the per-term score expression are computed exactly
as in ``rank_bm25`` 0.2.2, in float64, so scores are bit-identical.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

BM25_FORMAT_VERSION = "csr_bm25okapi_v1"


class SparseBM25Okapi:
    def __init__(self, *, vocab, idf, indptr, doc_ids, term_freqs, doc_len,
                 k1=1.5, b=0.75, epsilon=0.25):
        self.vocab = vocab                      # term -> term id
        self.idf = np.asarray(idf, dtype=np.float64)
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.doc_ids = np.asarray(doc_ids, dtype=np.int32)
        self.term_freqs = np.asarray(term_freqs)
        self.doc_len = np.asarray(doc_len, dtype=np.int64)
        self.k1, self.b, self.epsilon = float(k1), float(b), float(epsilon)
        self.corpus_size = int(len(self.doc_len))
        self.avgdl = float(self.doc_len.sum()) / self.corpus_size

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, tokenized_corpus, k1=1.5, b=0.75, epsilon=0.25):
        vocab: dict[str, int] = {}
        doc_freq: list[int] = []
        rows, cols, tfs, doc_len = [], [], [], []

        for doc_index, document in enumerate(tokenized_corpus):
            doc_len.append(len(document))
            frequencies: dict[str, int] = {}
            for word in document:
                frequencies[word] = frequencies.get(word, 0) + 1
            for word, freq in frequencies.items():
                term_id = vocab.get(word)
                if term_id is None:
                    term_id = len(vocab)
                    vocab[word] = term_id
                    doc_freq.append(0)
                doc_freq[term_id] += 1
                rows.append(term_id)
                cols.append(doc_index)
                tfs.append(freq)

        corpus_size = len(doc_len)
        if corpus_size == 0:
            raise ValueError("Cannot build BM25 over an empty corpus.")

        # rank_bm25 _calc_idf, in vocabulary insertion order.
        idf = np.empty(len(vocab), dtype=np.float64)
        idf_sum = 0
        negative = []
        for term_id, freq in enumerate(doc_freq):
            value = math.log(corpus_size - freq + 0.5) - math.log(freq + 0.5)
            idf[term_id] = value
            idf_sum += value
            if value < 0:
                negative.append(term_id)
        average_idf = idf_sum / len(vocab)
        idf[negative] = epsilon * average_idf

        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int32)
        max_tf = max(tfs) if tfs else 0
        tf_dtype = np.uint16 if max_tf < 2**16 else np.uint32
        tfs = np.asarray(tfs, dtype=tf_dtype)

        order = np.argsort(rows, kind="stable")  # keeps doc order inside a term
        counts = np.bincount(rows, minlength=len(vocab))
        indptr = np.zeros(len(vocab) + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])

        return cls(vocab=vocab, idf=idf, indptr=indptr, doc_ids=cols[order],
                   term_freqs=tfs[order], doc_len=doc_len, k1=k1, b=b, epsilon=epsilon)

    # ------------------------------------------------------------------ query
    def get_scores(self, query_tokens) -> np.ndarray:
        score = np.zeros(self.corpus_size)
        k1, b, avgdl = self.k1, self.b, self.avgdl
        for token in query_tokens:
            term_id = self.vocab.get(token)
            if term_id is None:
                continue  # rank_bm25 adds an exact 0.0 here
            lo, hi = self.indptr[term_id], self.indptr[term_id + 1]
            docs = self.doc_ids[lo:hi]
            tf = self.term_freqs[lo:hi].astype(np.float64)
            dl = self.doc_len[docs]
            score[docs] += self.idf[term_id] * (
                tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
            )
        return score

    # ------------------------------------------------------------ persistence
    def save(self, directory: Path) -> dict:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        terms = [None] * len(self.vocab)
        for term, term_id in self.vocab.items():
            terms[term_id] = term
        np.savez(
            directory / "bm25_index.npz",
            idf=self.idf, indptr=self.indptr, doc_ids=self.doc_ids,
            term_freqs=self.term_freqs, doc_len=self.doc_len,
        )
        (directory / "bm25_vocab.json").write_text(
            json.dumps(terms, ensure_ascii=False), encoding="utf-8"
        )
        meta = {
            "format": BM25_FORMAT_VERSION, "k1": self.k1, "b": self.b,
            "epsilon": self.epsilon, "corpus_size": self.corpus_size,
            "vocab_size": len(self.vocab), "postings": int(len(self.doc_ids)),
        }
        (directory / "bm25_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    @classmethod
    def load(cls, directory: Path) -> "SparseBM25Okapi":
        directory = Path(directory)
        meta = json.loads((directory / "bm25_meta.json").read_text(encoding="utf-8"))
        if meta.get("format") != BM25_FORMAT_VERSION:
            raise ValueError(f"Unsupported BM25 index format: {meta.get('format')!r}")
        terms = json.loads((directory / "bm25_vocab.json").read_text(encoding="utf-8"))
        arrays = np.load(directory / "bm25_index.npz")
        index = cls(
            vocab={term: i for i, term in enumerate(terms)},
            idf=arrays["idf"], indptr=arrays["indptr"], doc_ids=arrays["doc_ids"],
            term_freqs=arrays["term_freqs"], doc_len=arrays["doc_len"],
            k1=meta["k1"], b=meta["b"], epsilon=meta["epsilon"],
        )
        if index.corpus_size != meta["corpus_size"] or len(terms) != meta["vocab_size"]:
            raise ValueError("BM25 index files are inconsistent with bm25_meta.json.")
        return index

    @staticmethod
    def exists(directory: Path) -> bool:
        directory = Path(directory)
        return all((directory / n).is_file()
                   for n in ("bm25_index.npz", "bm25_vocab.json", "bm25_meta.json"))
