"""FinanceRAG retrieval runtime aligned with frozen held-out v2."""

from pathlib import Path
import json
import math
import os
import re

import faiss
import numpy as np
import pandas as pd

from .bm25_sparse import SparseBM25Okapi


# Default to the checked-out project directory so local and hosted deployments work.
PROJECT_DIR = Path(__file__).resolve().parents[2]
RUNTIME_DIR = PROJECT_DIR / "artifacts" / "runtime"
EVALUATION_DIR = PROJECT_DIR / "artifacts" / "evaluation"

CHUNKS_PATH = RUNTIME_DIR / "chunks_df.parquet"
EMBEDDINGS_PATH = RUNTIME_DIR / "chunk_embeddings.npy"
METADATA_PATH = RUNTIME_DIR / "runtime_metadata.json"
CONFIG_PATH = EVALUATION_DIR / "heldout_retrieval_config_v2.json"

DEFAULT_DENSE_CANDIDATE_K = 30
DEFAULT_BM25_CANDIDATE_K = 20
DEFAULT_RRF_K = 30
DEFAULT_FINAL_TOP_K = 10
EXPECTED_MODEL = "BAAI/bge-small-en-v1.5"

RUNTIME_DIR_ENV = "FINANCE_RAG_RUNTIME_DIR"


def resolve_runtime_dir(project_dir, runtime_dir=None):
    """Explicit argument > FINANCE_RAG_RUNTIME_DIR > artifacts/runtime (v1)."""
    chosen = runtime_dir or os.getenv(RUNTIME_DIR_ENV) or "artifacts/runtime"
    chosen = Path(chosen)
    return chosen if chosen.is_absolute() else Path(project_dir) / chosen


def _as_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    cleaned = [str(v).strip().upper() for v in value if v is not None and str(v).strip()]
    return cleaned or None


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['.-][A-Za-z0-9]+)*")


def tokenize_for_bm25(text):
    """Match the deterministic lexical tokenizer used by v2."""
    return TOKEN_PATTERN.findall(str(text).lower())


class FinanceRAGRetriever:
    """Load the frozen corpus artifacts and expose dense, BM25, and RRF retrieval."""

    def __init__(self, project_dir=PROJECT_DIR, runtime_dir=None, embedding_model=None):
        self.project_dir = Path(project_dir)
        runtime_dir = resolve_runtime_dir(self.project_dir, runtime_dir)
        self.runtime_dir = runtime_dir
        evaluation_dir = self.project_dir / "artifacts" / "evaluation"

        chunks_path = runtime_dir / "chunks_df.parquet"
        embeddings_path = runtime_dir / "chunk_embeddings.npy"
        metadata_path = runtime_dir / "runtime_metadata.json"
        runtime_config_path = self.project_dir / "config" / "retrieval_config.json"
        config_path = (
            runtime_config_path
            if runtime_config_path.is_file()
            else evaluation_dir / "heldout_retrieval_config_v2.json"
        )

        for path in (chunks_path, embeddings_path, metadata_path, config_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required FinanceRAG artifact missing: {path}")

        self.chunks_df = pd.read_parquet(chunks_path).reset_index(drop=True)
        self.embeddings = np.load(embeddings_path).astype(np.float32, copy=False)
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.config = json.loads(config_path.read_text(encoding="utf-8"))

        self._validate_artifacts()

        self.expected_rows, self.embedding_dimensions = self.embeddings.shape

        self.corpus_tickers = (
            self.chunks_df["ticker"].astype(str).str.strip().str.upper()
        )
        self.corpus_form_types = (
            self.chunks_df["form_type"].astype(str).str.strip().str.upper()
        )
        self.corpus_filing_dates = (
            pd.to_datetime(self.chunks_df["filing_date"], errors="raise")
            .dt.strftime("%Y-%m-%d")
        )

        # v2 bundles ship a prebuilt sparse BM25 index; v1 bundles do not, so
        # it is built in memory. Scores are bit-identical to rank_bm25 either way.
        if SparseBM25Okapi.exists(runtime_dir):
            self.bm25_index = SparseBM25Okapi.load(runtime_dir)
            if self.bm25_index.corpus_size != len(self.chunks_df):
                raise ValueError("BM25 index and chunk table row counts do not match.")
            self.bm25_source = "prebuilt"
        else:
            tokenized_corpus = [
                tokenize_for_bm25(text) for text in self.chunks_df["text"].tolist()
            ]
            self.bm25_index = SparseBM25Okapi.build(tokenized_corpus)
            self.bm25_source = "rebuilt_in_memory"

        self.faiss_index = faiss.IndexFlatIP(self.embedding_dimensions)
        self.faiss_index.add(np.ascontiguousarray(self.embeddings, dtype=np.float32))

        if embedding_model is None:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer(self.metadata["embedding_model_name"])
        self.embedding_model = embedding_model

    @property
    def available_tickers(self):
        return sorted(self.corpus_tickers.unique().tolist())

    @property
    def available_form_types(self):
        return sorted(self.corpus_form_types.unique().tolist())

    def _validate_artifacts(self):
        required_columns = {
            "chunk_id", "text", "ticker", "form_type", "filing_date",
            "company", "form_type", "source_url",
        }
        missing = sorted(required_columns - set(self.chunks_df.columns))
        if missing:
            raise ValueError(f"Corpus is missing required columns: {missing}")

        if self.metadata.get("embedding_model_name") != EXPECTED_MODEL:
            raise ValueError(
                "Unexpected embedding model in runtime metadata: "
                f"{self.metadata.get('embedding_model_name')!r}"
            )

        if self.chunks_df["chunk_id"].isna().any() or not self.chunks_df["chunk_id"].is_unique:
            raise ValueError("Corpus chunk_id values must be non-null and unique.")

        if self.chunks_df["text"].isna().any():
            raise ValueError("Corpus contains null chunk text.")

        if self.embeddings.ndim != 2:
            raise ValueError("Embedding matrix must be two-dimensional.")

        if len(self.chunks_df) != self.embeddings.shape[0]:
            raise ValueError("Corpus and embedding row counts do not match.")

        if self.embeddings.shape[1] != self.metadata.get("embedding_dimensions"):
            raise ValueError("Embedding dimensions disagree with runtime metadata.")

        if not np.isfinite(self.embeddings).all():
            raise ValueError("Embedding matrix contains non-finite values.")

        if not np.allclose(
            np.linalg.norm(self.embeddings, axis=1), 1.0, atol=1e-4
        ):
            raise ValueError("Corpus embeddings are not unit-normalized.")

        config_checks = {
            "dense_candidate_k": DEFAULT_DENSE_CANDIDATE_K,
            "bm25_candidate_k": DEFAULT_BM25_CANDIDATE_K,
            "rrf_k": DEFAULT_RRF_K,
            "final_top_k": DEFAULT_FINAL_TOP_K,
        }
        for key, expected in config_checks.items():
            if self.config.get(key) != expected:
                raise ValueError(
                    f"Frozen v2 setting {key!r} should be {expected}, "
                    f"found {self.config.get(key)!r}."
                )

    def encode_query(self, query):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        vector = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)

        if vector.shape != (1, self.embeddings.shape[1]):
            raise ValueError(f"Unexpected query embedding shape: {vector.shape}")
        if not np.isfinite(vector).all():
            raise ValueError("Query embedding contains non-finite values.")

        return vector

    def resolve_candidate_rows(self, *, ticker=None, form_type=None, filing_date=None,
                               start_date=None, end_date=None):
        mask = np.ones(len(self.chunks_df), dtype=bool)

        tickers = _as_list(ticker)
        if tickers is not None:
            mask &= self.corpus_tickers.isin(tickers).to_numpy()

        form_types = _as_list(form_type)
        if form_types is not None:
            mask &= self.corpus_form_types.isin(form_types).to_numpy()

        if filing_date is not None:
            normalized_date = pd.to_datetime(
                filing_date, errors="raise"
            ).strftime("%Y-%m-%d")
            mask &= self.corpus_filing_dates.eq(normalized_date).to_numpy()

        # ISO date strings compare correctly as strings.
        if start_date is not None:
            start = pd.to_datetime(start_date, errors="raise").strftime("%Y-%m-%d")
            mask &= (self.corpus_filing_dates >= start).to_numpy()
        if end_date is not None:
            end = pd.to_datetime(end_date, errors="raise").strftime("%Y-%m-%d")
            mask &= (self.corpus_filing_dates <= end).to_numpy()

        rows = np.flatnonzero(mask).astype(np.int64)
        if len(rows) == 0:
            raise ValueError(
                "No corpus chunks matched the supplied metadata filters. "
                "Check ticker, form type, and filing dates."
            )
        return rows

    def _construct_results(
        self, rows, scores, *, method, bm25_ranks=None, dense_ranks=None
    ):
        rows = np.asarray(rows, dtype=np.int64)
        scores = np.asarray(scores, dtype=np.float64)
        result = self.chunks_df.iloc[rows].copy()

        result.insert(0, "corpus_row", rows)
        result.insert(
            0, "dense_rank",
            dense_ranks if dense_ranks is not None
            else pd.array([pd.NA] * len(result), dtype="Int64"),
        )
        result.insert(
            0, "bm25_rank",
            bm25_ranks if bm25_ranks is not None
            else pd.array([pd.NA] * len(result), dtype="Int64"),
        )
        result.insert(0, "retrieval_score", scores)
        result.insert(0, "retrieval_method", method)
        result.insert(0, "rank", np.arange(1, len(result) + 1, dtype=np.int64))
        return result.reset_index(drop=True)

    def retrieve_dense(
        self, query, *, top_k=10, ticker=None, form_type=None, filing_date=None,
        start_date=None, end_date=None,
    ):
        if not 1 <= top_k <= len(self.chunks_df):
            raise ValueError(f"top_k must be between 1 and {len(self.chunks_df):,}.")

        query_vector = self.encode_query(query)
        filters_supplied = any(
            value is not None
            for value in (ticker, form_type, filing_date, start_date, end_date)
        )

        if filters_supplied:
            rows = self.resolve_candidate_rows(
                ticker=ticker, form_type=form_type, filing_date=filing_date,
                start_date=start_date, end_date=end_date,
            )
            scores = self.embeddings[rows] @ query_vector[0]
            positions = np.argsort(-scores, kind="stable")[:min(top_k, len(rows))]
            selected_rows = rows[positions]
            selected_scores = scores[positions]
            method = "dense_filtered"
        else:
            scores, rows = self.faiss_index.search(query_vector, min(top_k, len(self.chunks_df)))
            selected_rows = rows[0].astype(np.int64)
            selected_scores = scores[0].astype(np.float64)
            method = "dense"

        return self._construct_results(selected_rows, selected_scores, method=method)

    def retrieve_bm25(
        self, query, *, top_k=10, ticker=None, form_type=None, filing_date=None,
        start_date=None, end_date=None,
    ):
        if not 1 <= top_k <= len(self.chunks_df):
            raise ValueError(f"top_k must be between 1 and {len(self.chunks_df):,}.")

        rows = self.resolve_candidate_rows(
            ticker=ticker, form_type=form_type, filing_date=filing_date,
            start_date=start_date, end_date=end_date,
        )
        scores_all = np.asarray(
            self.bm25_index.get_scores(tokenize_for_bm25(query)),
            dtype=np.float64,
        )
        scores = scores_all[rows]
        positions = np.argsort(-scores, kind="stable")[:min(top_k, len(rows))]
        selected_rows = rows[positions]
        selected_scores = scores[positions]

        filtered = any(
            value is not None
            for value in (ticker, form_type, filing_date, start_date, end_date)
        )
        method = "bm25_filtered" if filtered else "bm25"
        return self._construct_results(selected_rows, selected_scores, method=method)

    def retrieve_hybrid_rrf(
        self,
        query,
        *,
        top_k=DEFAULT_FINAL_TOP_K,
        dense_candidate_k=DEFAULT_DENSE_CANDIDATE_K,
        bm25_candidate_k=DEFAULT_BM25_CANDIDATE_K,
        rrf_k=DEFAULT_RRF_K,
        ticker=None,
        form_type=None,
        filing_date=None,
        start_date=None,
        end_date=None,
    ):
        if not 1 <= top_k <= len(self.chunks_df):
            raise ValueError(f"top_k must be between 1 and {len(self.chunks_df):,}.")
        if dense_candidate_k < top_k or bm25_candidate_k < top_k:
            raise ValueError("Both candidate depths must be greater than or equal to top_k.")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive.")

        filters = dict(ticker=ticker, form_type=form_type, filing_date=filing_date,
                       start_date=start_date, end_date=end_date)
        dense = self.retrieve_dense(query, top_k=dense_candidate_k, **filters)
        bm25 = self.retrieve_bm25(query, top_k=bm25_candidate_k, **filters)

        dense_rank_map = {
            int(row.corpus_row): int(row.rank)
            for row in dense.itertuples(index=False)
        }
        bm25_rank_map = {
            int(row.corpus_row): int(row.rank)
            for row in bm25.itertuples(index=False)
        }

        candidate_rows = sorted(set(dense_rank_map) | set(bm25_rank_map))
        fused_records = []

        for row in candidate_rows:
            dense_rank = dense_rank_map.get(row)
            bm25_rank = bm25_rank_map.get(row)
            score = 0.0
            if dense_rank is not None:
                score += 1.0 / (rrf_k + dense_rank)
            if bm25_rank is not None:
                score += 1.0 / (rrf_k + bm25_rank)

            fused_records.append({
                "corpus_row": row,
                "retrieval_score": score,
                "dense_rank": dense_rank,
                "bm25_rank": bm25_rank,
            })

        # Preserve v2's deterministic tie-break order.
        fused_records.sort(
            key=lambda record: (
                -record["retrieval_score"],
                record["bm25_rank"] if record["bm25_rank"] is not None else np.inf,
                record["dense_rank"] if record["dense_rank"] is not None else np.inf,
                record["corpus_row"],
            )
        )
        fused_records = fused_records[:top_k]

        return self._construct_results(
            [record["corpus_row"] for record in fused_records],
            [record["retrieval_score"] for record in fused_records],
            method="hybrid_rrf_asymmetric_filtered",
            bm25_ranks=pd.array(
                [record["bm25_rank"] for record in fused_records], dtype="Int64"
            ),
            dense_ranks=pd.array(
                [record["dense_rank"] for record in fused_records], dtype="Int64"
            ),
        )

    def _search_one(self, query, *, method, top_k, **filters):
        if method == "dense":
            return self.retrieve_dense(query, top_k=top_k, **filters)
        if method == "bm25":
            return self.retrieve_bm25(query, top_k=top_k, **filters)
        if method == "hybrid_rrf":
            return self.retrieve_hybrid_rrf(
                query,
                top_k=top_k,
                dense_candidate_k=max(self.config["dense_candidate_k"], top_k),
                bm25_candidate_k=max(self.config["bm25_candidate_k"], top_k),
                rrf_k=self.config["rrf_k"],
                **filters,
            )
        raise ValueError("method must be 'dense', 'bm25', or 'hybrid_rrf'")

    def search(
        self,
        query,
        *,
        method="hybrid_rrf",
        ticker=None,
        form_type=None,
        filing_date=None,
        start_date=None,
        end_date=None,
        top_k=DEFAULT_FINAL_TOP_K,
    ):
        """App-facing retrieval entry point.

        ``ticker`` may be one ticker or a list. With several tickers, evidence is
        retrieved per company and interleaved, so a comparison question
        ("NVDA vs AMD margins") gets evidence for every company instead of
        whichever one dominates the global ranking.
        """
        filters = dict(form_type=form_type, filing_date=filing_date,
                       start_date=start_date, end_date=end_date)
        tickers = _as_list(ticker)

        if tickers is None or len(tickers) == 1:
            return self._search_one(query, method=method, top_k=top_k,
                                    ticker=tickers, **filters)

        per_ticker_k = max(1, math.ceil(top_k / len(tickers)))
        per_ticker = []
        for name in tickers:
            try:
                per_ticker.append(self._search_one(
                    query, method=method, top_k=per_ticker_k, ticker=name, **filters))
            except ValueError as exc:
                if "No corpus chunks matched" not in str(exc):
                    raise
        if not per_ticker:
            raise ValueError(
                "No corpus chunks matched the supplied metadata filters. "
                "Check ticker, form type, and filing dates."
            )

        # Round-robin by within-company rank: A1, B1, A2, B2, ...
        frames = []
        for position in range(per_ticker_k):
            for result in per_ticker:
                if position < len(result):
                    frames.append(result.iloc[[position]])
        merged = pd.concat(frames, ignore_index=True).head(top_k)
        merged["rank"] = np.arange(1, len(merged) + 1, dtype=np.int64)
        merged["retrieval_method"] = merged["retrieval_method"].astype(str) + "_multi_ticker"
        return merged
