"""Build a versioned runtime bundle from the frozen raw SEC corpus.

Stages (each resumable, each safe to re-run):

1. ``select``   choose filings from the frozen manifest for a company list
2. ``fetch``    pull raw HTML from R2 (or local storage) into a local cache,
                verifying every object against the manifest SHA-256
3. ``chunk``    HTML -> text -> canonical chunks, one parquet shard per filing
4. ``embed``    BGE embeddings into a checkpointed memmap (survives Colab resets)
5. ``finalize`` validate, build sparse BM25, write the runtime contract that
                ``FinanceRAGRetriever`` loads, plus a SHA-256 build manifest

Output layout (``<out>/<version>/``)::

    runtime/chunks_df.parquet          <- same contract as runtime-v1
    runtime/chunk_embeddings.npy
    runtime/runtime_metadata.json
    runtime/bm25_index.npz             <- new: prebuilt sparse BM25
    runtime/bm25_vocab.json
    runtime/bm25_meta.json
    build_manifest.json                <- inputs, versions, counts, file hashes
    work/                              <- resumable intermediates (not shipped)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..bm25_sparse import SparseBM25Okapi
from .chunking import (
    CHUNKING_VERSION,
    MAX_CHUNK_TOKENS,
    PROCESSING_VERSION,
    TOKENIZER_NAME,
    DocumentChunker,
    make_document_id,
    normalize_filing_text,
    sha256_text,
)
from .html_text import EXTRACTION_VERSION, extract_filing_text

log = logging.getLogger("finance_rag.build")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSIONS = 384
SUPPORTED_FORMS = ("10-K", "10-Q", "8-K")

CHUNK_COLUMNS = [
    "chunk_id", "document_id", "chunk_index", "start_span_index", "end_span_index",
    "start_char", "end_char", "token_count", "below_minimum_target", "text",
    "ticker", "company", "cik", "form_type", "filing_date", "report_date",
    "accession_number", "primary_document", "source_url",
    "raw_content_sha256", "processed_content_sha256",
    "processing_version", "corpus_version",
]


@dataclass
class BuildConfig:
    version: str
    tickers: list[str]
    forms: tuple[str, ...] = SUPPORTED_FORMS
    since: str | None = None
    until: str | None = None
    limit_per_company: int | None = None
    corpus_version: str = "raw-corpus-v1"
    manifest_path: Path = Path("artifacts/releases/raw-corpus-v1/download_manifest.csv.gz")
    inventory_path: Path = Path("artifacts/releases/raw-corpus-v1/filing_inventory.csv.gz")
    securities_path: Path = Path("data/universe/securities.csv")
    out_dir: Path = Path("artifacts/builds")
    cache_dir: Path = Path("data/cache/raw")
    fetch_workers: int = 8
    chunk_workers: int = max(1, min(4, (os.cpu_count() or 2) - 1))
    embed_batch_size: int = 32
    embed_checkpoint_rows: int = 4096
    device: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def build_dir(self) -> Path:
        return Path(self.out_dir) / self.version

    @property
    def work_dir(self) -> Path:
        return self.build_dir / "work"

    @property
    def runtime_dir(self) -> Path:
        return self.build_dir / "runtime"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ticker_file(path: Path) -> list[str]:
    tickers = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip().upper()
        if line and line not in tickers:
            tickers.append(line)
    return tickers


# --------------------------------------------------------------------- select
def select_filings(cfg: BuildConfig) -> pd.DataFrame:
    manifest = pd.read_csv(cfg.manifest_path, dtype=str)
    inventory = pd.read_csv(cfg.inventory_path, dtype=str)

    if not manifest["status"].eq("downloaded").all():
        raise ValueError("Manifest contains filings that were not downloaded.")

    # Resolve requested tickers to CIKs (handles GOOG/GOOGL style aliases).
    ticker_to_cik = dict(zip(manifest["ticker"].str.upper(), manifest["cik"]))
    if Path(cfg.securities_path).is_file():
        securities = pd.read_csv(cfg.securities_path, dtype=str)
        ticker_to_cik.update(zip(securities["ticker"].str.upper(), securities["cik"]))

    unknown = [t for t in cfg.tickers if t not in ticker_to_cik]
    if unknown:
        raise ValueError(f"Tickers not found in the frozen corpus: {unknown}")

    cik_to_ticker = {}
    for ticker in cfg.tickers:
        cik = ticker_to_cik[ticker]
        if cik in cik_to_ticker:
            raise ValueError(f"{ticker} and {cik_to_ticker[cik]} are the same issuer ({cik}).")
        cik_to_ticker[cik] = ticker

    selected = manifest[manifest["cik"].isin(cik_to_ticker)].copy()
    selected = selected[selected["form_type"].isin(cfg.forms)]
    if cfg.since:
        selected = selected[selected["filing_date"] >= cfg.since]
    if cfg.until:
        selected = selected[selected["filing_date"] <= cfg.until]

    selected["ticker"] = selected["cik"].map(cik_to_ticker)
    extra_cols = ["accession_number", "company", "report_date", "primary_document"]
    selected = selected.merge(
        inventory[extra_cols].drop_duplicates("accession_number"),
        on="accession_number", how="left", validate="one_to_one",
    )
    if selected["primary_document"].isna().any():
        raise ValueError("Some selected filings are missing from the filing inventory.")

    selected = selected.sort_values(
        ["ticker", "filing_date", "form_type", "accession_number"],
        kind="stable",
    )
    if cfg.limit_per_company:
        # Keep the most recent filings of each form for sample builds.
        selected = (
            selected.sort_values("filing_date", ascending=False, kind="stable")
            .groupby(["ticker", "form_type"], sort=False)
            .head(cfg.limit_per_company)
            .sort_values(["ticker", "filing_date", "form_type", "accession_number"], kind="stable")
        )

    selected["document_id"] = [
        make_document_id(t, f, a)
        for t, f, a in zip(selected["ticker"], selected["form_type"], selected["accession_number"])
    ]
    if not selected["document_id"].is_unique:
        raise ValueError("Duplicate document_id values in selection.")
    selected = selected.reset_index(drop=True)

    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(cfg.work_dir / "selected_filings.csv", index=False)
    missing = sorted(set(cfg.tickers) - set(selected["ticker"]))
    log.info("Selected %d filings for %d companies (%s). No filings for: %s",
             len(selected), selected["ticker"].nunique(),
             selected["form_type"].value_counts().to_dict(), missing or "none")
    return selected


# ---------------------------------------------------------------------- fetch
def cache_path(cfg: BuildConfig, object_key: str) -> Path:
    return Path(cfg.cache_dir) / object_key


def fetch_raw(cfg: BuildConfig, selected: pd.DataFrame, storage) -> dict:
    """Download missing raw objects into the cache, verifying SHA-256."""

    def one(row):
        target = cache_path(cfg, row["object_key"])
        if target.is_file() and file_sha256(target) == row["sha256"]:
            return "cached"
        payload = storage.get_bytes(row["object_key"])
        digest = hashlib.sha256(payload).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {row['object_key']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        tmp.write_bytes(payload)
        tmp.replace(target)
        return "fetched"

    counts = {"cached": 0, "fetched": 0}
    rows = selected.to_dict("records")
    with ThreadPoolExecutor(max_workers=cfg.fetch_workers) as pool:
        futures = {pool.submit(one, row): row for row in rows}
        for done, future in enumerate(as_completed(futures), 1):
            counts[future.result()] += 1  # any failure stops the build loudly
            if done % 100 == 0 or done == len(rows):
                log.info("fetch %d/%d %s", done, len(rows), counts)
    return counts


# ---------------------------------------------------------------------- chunk
_WORKER_CHUNKER = None


def _worker_chunker():
    global _WORKER_CHUNKER
    if _WORKER_CHUNKER is None:
        from transformers import AutoTokenizer
        _WORKER_CHUNKER = DocumentChunker(AutoTokenizer.from_pretrained(TOKENIZER_NAME))
    return _WORKER_CHUNKER


def chunk_filing(row: dict, raw_path: str, shard_path: str, corpus_version: str,
                 chunker: DocumentChunker | None = None) -> dict:
    """Process one filing into a parquet shard. Runs inside worker processes."""
    chunker = chunker or _worker_chunker()
    raw = Path(raw_path).read_bytes()
    text = normalize_filing_text(extract_filing_text(raw))
    chunks = chunker.chunk_document(row["document_id"], text)

    meta = {
        "ticker": row["ticker"], "company": row["company"],
        "cik": str(row["cik"]).zfill(10), "form_type": row["form_type"],
        "filing_date": row["filing_date"], "report_date": row.get("report_date"),
        "accession_number": row["accession_number"],
        "primary_document": row["primary_document"], "source_url": row["source_url"],
        "raw_content_sha256": row["sha256"],
        "processed_content_sha256": sha256_text(text),
        "processing_version": f"{EXTRACTION_VERSION}+{PROCESSING_VERSION}",
        "corpus_version": corpus_version,
    }
    frame = pd.DataFrame([{**chunk, **meta} for chunk in chunks], columns=CHUNK_COLUMNS)
    tmp = Path(shard_path).with_suffix(".parquet.part")
    frame.to_parquet(tmp, index=False)
    tmp.replace(shard_path)
    return {"document_id": row["document_id"], "chars": len(text), "chunks": len(chunks)}


def chunk_all(cfg: BuildConfig, selected: pd.DataFrame, chunker=None) -> pd.DataFrame:
    shard_dir = cfg.work_dir / "chunks"
    shard_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for row in selected.to_dict("records"):
        shard = shard_dir / f"{row['document_id']}.parquet"
        if not shard.is_file():
            todo.append((row, str(cache_path(cfg, row["object_key"])), str(shard)))
    log.info("chunk: %d to do, %d already done", len(todo), len(selected) - len(todo))

    stats = []
    if chunker is not None or cfg.chunk_workers <= 1:
        for row, raw, shard in todo:
            stats.append(chunk_filing(row, raw, shard, cfg.corpus_version, chunker))
    else:
        with ProcessPoolExecutor(max_workers=cfg.chunk_workers) as pool:
            futures = [pool.submit(chunk_filing, row, raw, shard, cfg.corpus_version)
                       for row, raw, shard in todo]
            for done, future in enumerate(as_completed(futures), 1):
                stats.append(future.result())
                if done % 50 == 0 or done == len(futures):
                    log.info("chunk %d/%d", done, len(futures))
    return pd.DataFrame(stats)


def load_chunks(cfg: BuildConfig, selected: pd.DataFrame) -> pd.DataFrame:
    shard_dir = cfg.work_dir / "chunks"
    frames = []
    for document_id in selected["document_id"]:  # deterministic selection order
        frame = pd.read_parquet(shard_dir / f"{document_id}.parquet")
        if len(frame):
            frames.append(frame)
    if not frames:
        raise ValueError("No chunks were produced.")
    chunks = pd.concat(frames, ignore_index=True)
    chunks = chunks.sort_values(["ticker", "filing_date", "document_id", "chunk_index"],
                                kind="stable").reset_index(drop=True)
    return chunks


# ---------------------------------------------------------------------- embed
def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def embed_chunks(cfg: BuildConfig, chunks: pd.DataFrame, model=None) -> np.ndarray:
    """Encode in checkpointed slices; a crash resumes from the last slice."""
    rows = len(chunks)
    ids_hash = hashlib.sha256("\n".join(chunks["chunk_id"]).encode()).hexdigest()
    mm_path = cfg.work_dir / "embeddings.f32"
    progress_path = cfg.work_dir / "embeddings_progress.json"

    progress = {}
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        if progress.get("chunk_ids_sha256") != ids_hash or progress.get("rows") != rows:
            log.warning("Chunk set changed since last embedding run; restarting embeddings.")
            progress = {}
    done = int(progress.get("done", 0))

    mode = "r+" if done and mm_path.is_file() else "w+"
    matrix = np.memmap(mm_path, dtype=np.float32, mode=mode, shape=(rows, EMBEDDING_DIMENSIONS))

    if done < rows:
        if model is None:
            from sentence_transformers import SentenceTransformer
            device = pick_device(cfg.device)
            log.info("Loading %s on %s", EMBEDDING_MODEL, device)
            model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        texts = chunks["text"].tolist()
        step = cfg.embed_checkpoint_rows
        for start in range(done, rows, step):
            end = min(start + step, rows)
            vectors = model.encode(
                texts[start:end], batch_size=cfg.embed_batch_size,
                convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
            ).astype(np.float32, copy=False)
            if vectors.shape != (end - start, EMBEDDING_DIMENSIONS):
                raise ValueError(f"Unexpected embedding shape {vectors.shape}")
            matrix[start:end] = vectors
            matrix.flush()
            progress_path.write_text(json.dumps(
                {"rows": rows, "done": end, "chunk_ids_sha256": ids_hash}))
            log.info("embed %d/%d", end, rows)

    return np.asarray(matrix)


# ------------------------------------------------------------------- finalize
def validate_bundle(chunks: pd.DataFrame, embeddings: np.ndarray) -> dict:
    checks = {
        "rows_match": len(chunks) == embeddings.shape[0],
        "dimensions": embeddings.shape[1] == EMBEDDING_DIMENSIONS,
        "chunk_ids_unique": bool(chunks["chunk_id"].is_unique),
        "no_null_text": bool(chunks["text"].notna().all() and chunks["text"].str.len().gt(0).all()),
        "token_ceiling": int(chunks["token_count"].max()) <= MAX_CHUNK_TOKENS,
        "finite": bool(np.isfinite(embeddings).all()),
        "unit_normalized": bool(np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4)),
    }
    # Every sampled vector must retrieve itself first (Notebook 02 check).
    rng = np.random.default_rng(20260806)
    sample = rng.choice(len(chunks), size=min(64, len(chunks)), replace=False)
    sims = embeddings[sample] @ embeddings.T
    top = sims.argmax(axis=1)
    checks["self_retrieval"] = bool(
        all(t == s or np.isclose(sims[i, t], sims[i, s]) for i, (t, s) in enumerate(zip(top, sample)))
    )
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"Bundle validation failed: {failed}")
    return checks


def finalize(cfg: BuildConfig, selected: pd.DataFrame, chunks: pd.DataFrame,
             embeddings: np.ndarray) -> dict:
    from ..retrieval import tokenize_for_bm25

    checks = validate_bundle(chunks, embeddings)
    runtime = cfg.runtime_dir
    runtime.mkdir(parents=True, exist_ok=True)

    chunks.to_parquet(runtime / "chunks_df.parquet", index=False, compression="zstd")
    np.save(runtime / "chunk_embeddings.npy", np.ascontiguousarray(embeddings, dtype=np.float32))

    bm25 = SparseBM25Okapi.build([tokenize_for_bm25(t) for t in chunks["text"]])
    bm25_meta = bm25.save(runtime)

    metadata = {
        "runtime_version": cfg.version,
        "chunk_ids_unique": True,
        "corpus_rows": int(len(chunks)),
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "embedding_dtype": "float32",
        "embedding_model_name": EMBEDDING_MODEL,
        "embedding_rows": int(embeddings.shape[0]),
        "embeddings_normalized": True,
        "documents": int(chunks["document_id"].nunique()),
        "companies": sorted(chunks["ticker"].unique().tolist()),
        "form_types": sorted(chunks["form_type"].unique().tolist()),
        "filing_date_min": str(chunks["filing_date"].min()),
        "filing_date_max": str(chunks["filing_date"].max()),
        "chunking_version": CHUNKING_VERSION,
        "extraction_version": EXTRACTION_VERSION,
        "processing_version": PROCESSING_VERSION,
        "corpus_version": cfg.corpus_version,
        "bm25": bm25_meta,
    }
    (runtime / "runtime_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    produced = set(chunks["document_id"])
    empty_docs = sorted(set(selected["document_id"]) - produced)
    files = {p.name: {"sha256": file_sha256(p), "bytes": p.stat().st_size}
             for p in sorted(runtime.iterdir()) if p.is_file()}
    manifest = {
        "runtime_version": cfg.version,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        "inputs": {
            "manifest_sha256": file_sha256(Path(cfg.manifest_path)),
            "selected_filings": int(len(selected)),
            "documents_with_chunks": int(len(produced)),
            "documents_without_text": empty_docs,
        },
        "counts": {
            "chunks": int(len(chunks)),
            "by_form": chunks.groupby("form_type")["document_id"].nunique().to_dict(),
            "chunks_by_form": chunks["form_type"].value_counts().to_dict(),
            "mean_tokens": round(float(chunks["token_count"].mean()), 1),
        },
        "validation": checks,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "files": files,
    }
    (cfg.build_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Runtime bundle ready: %s (%d chunks, %d documents)",
             runtime, len(chunks), metadata["documents"])
    return manifest
