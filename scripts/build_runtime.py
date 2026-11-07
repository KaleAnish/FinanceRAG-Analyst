"""Build a FinanceRAG runtime bundle from the frozen raw SEC corpus in R2.

Examples
--------
# 1) Sample: 3 companies, 2 most recent filings per form. Run this first.
python scripts/build_runtime.py --version runtime-v2-sample \
    --tickers AAPL,JPM,XOM --limit-per-company 2

# 2) The MVP: 50 companies, all 10-K / 10-Q / 8-K in the frozen corpus.
python scripts/build_runtime.py --version runtime-v2-mvp50 \
    --companies config/universe/mvp50.txt

# Run stages separately (e.g. chunk on the Mac, embed on a Colab GPU):
python scripts/build_runtime.py ... --stage fetch
python scripts/build_runtime.py ... --stage chunk
python scripts/build_runtime.py ... --stage embed   # resumable
python scripts/build_runtime.py ... --stage finalize

Every stage skips work that is already done, so re-running after a crash is safe.
Storage credentials come from .env (STORAGE_BACKEND=r2, R2_*), exactly as for
the downloader. Nothing is written to R2.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from finance_rag.processing.runtime_build import (  # noqa: E402
    SUPPORTED_FORMS,
    BuildConfig,
    chunk_all,
    embed_chunks,
    fetch_raw,
    finalize,
    load_chunks,
    read_ticker_file,
    select_filings,
)

STAGES = ("select", "fetch", "chunk", "embed", "finalize")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True, help="Bundle name, e.g. runtime-v2-mvp50")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--companies", type=Path, help="File with one ticker per line")
    group.add_argument("--tickers", help="Comma-separated tickers")
    parser.add_argument("--forms", default=",".join(SUPPORTED_FORMS))
    parser.add_argument("--since", help="Earliest filing_date (YYYY-MM-DD)")
    parser.add_argument("--until", help="Latest filing_date (YYYY-MM-DD)")
    parser.add_argument("--limit-per-company", type=int,
                        help="Most recent N filings per company and form (sample builds)")
    parser.add_argument("--stage", choices=("all",) + STAGES, default="all")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "artifacts" / "builds")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_DIR / "data" / "cache" / "raw")
    parser.add_argument("--fetch-workers", type=int, default=8)
    parser.add_argument("--chunk-workers", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", help="cuda / mps / cpu (default: auto)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tickers = (read_ticker_file(args.companies) if args.companies
               else [t.strip().upper() for t in args.tickers.split(",") if t.strip()])
    forms = tuple(f.strip().upper() for f in args.forms.split(",") if f.strip())
    bad = [f for f in forms if f not in SUPPORTED_FORMS]
    if bad:
        raise SystemExit(f"Unsupported forms: {bad}")

    releases = PROJECT_DIR / "artifacts" / "releases" / "raw-corpus-v1"
    cfg = BuildConfig(
        version=args.version, tickers=tickers, forms=forms,
        since=args.since, until=args.until, limit_per_company=args.limit_per_company,
        manifest_path=releases / "download_manifest.csv.gz",
        inventory_path=releases / "filing_inventory.csv.gz",
        securities_path=PROJECT_DIR / "data" / "universe" / "securities.csv",
        out_dir=args.out_dir, cache_dir=args.cache_dir,
        fetch_workers=args.fetch_workers, embed_batch_size=args.batch_size, device=args.device,
    )
    if args.chunk_workers:
        cfg.chunk_workers = args.chunk_workers

    run = (lambda stage: args.stage in ("all", stage))
    selected = select_filings(cfg)

    if run("fetch"):
        from finance_rag.storage import create_storage
        fetch_raw(cfg, selected, create_storage())

    if run("chunk"):
        stats = chunk_all(cfg, selected)
        if len(stats):
            empty = stats[stats["chunks"] == 0]
            logging.info("chunked %d filings; %d produced no text", len(stats), len(empty))

    if run("embed") or run("finalize"):
        chunks = load_chunks(cfg, selected)
        logging.info("%d chunks across %d filings", len(chunks), chunks["document_id"].nunique())
        embeddings = embed_chunks(cfg, chunks) if run("embed") else None
        if run("finalize"):
            if embeddings is None:
                embeddings = embed_chunks(cfg, chunks)  # loads the finished memmap
            manifest = finalize(cfg, selected, chunks, embeddings)
            print(json.dumps({"runtime_dir": str(cfg.runtime_dir), **manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
