from __future__ import annotations

import argparse
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finance_rag.ingestion.sec_client import SECClient
from finance_rag.ingestion.sec_downloader import SECFilingDownloader
from finance_rag.storage import StorageConfig, create_storage


def atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        frame.to_csv(handle, index=False)

    temp_path.replace(destination)


def main() -> None:
    import time
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(
        description="Download SEC filing documents into configured storage."
    )
    parser.add_argument("--inventory", default="data/ingestion/filing_inventory.csv")
    parser.add_argument("--manifest", default="data/ingestion/download_manifest.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ticker", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    if not args.dry_run and not os.getenv("SEC_USER_AGENT"):
        parser.error("Set SEC_USER_AGENT before downloading.")

    inventory_path = Path(args.inventory)
    manifest_path = Path(args.manifest)

    if not inventory_path.exists():
        parser.error(f"Inventory not found: {inventory_path}")

    inventory = pd.read_csv(inventory_path, dtype=str).fillna("")

    required = {
        "cik", "ticker", "form_type", "accession_number",
        "filing_date", "primary_document", "source_url",
    }
    missing = required - set(inventory.columns)
    if missing:
        parser.error(f"Inventory missing columns: {sorted(missing)}")

    if args.ticker:
        selected = {ticker.upper() for ticker in args.ticker}
        inventory = inventory[
            inventory["ticker"].str.upper().isin(selected)
        ]

    inventory = inventory.drop_duplicates(
        ["cik", "accession_number"]
    ).reset_index(drop=True)

    if args.limit:
        inventory = inventory.head(args.limit)

    previous_frame = (
        pd.read_csv(manifest_path, dtype=str).fillna("")
        if manifest_path.exists()
        else pd.DataFrame()
    )

    previous = {}
    if not previous_frame.empty:
        previous = {
            (row["cik"], row["accession_number"]): row
            for row in previous_frame.to_dict("records")
        }

    if args.dry_run:
        print(f"Dry run: {len(inventory):,} filings selected.")
        print(f"Existing manifest records: {len(previous):,}")
        print(inventory[["ticker", "form_type", "accession_number"]].head(10).to_string(index=False))
        return

    config = StorageConfig.from_env()
    storage = create_storage(config)
    downloader = SECFilingDownloader(SECClient(), storage)

    results = dict(previous)
    downloaded = skipped = failed = 0
    total = len(inventory)
    start_time = time.monotonic()
    started_at = datetime.now(timezone.utc)

    print("\\n========== SEC DOWNLOAD RUN ==========")
    print(f"Started: {started_at.isoformat(timespec='seconds')}")
    print(f"Inventory: {inventory_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Selected filings: {total:,}")
    print(f"Existing manifest records: {len(previous):,}")
    print(f"Progress interval: {args.progress_every:,}")
    print("Workers: 1 (SEC request pacing preserved)")
    print("======================================\\n", flush=True)

    last_checkpoint = 0

    for index, row in inventory.iterrows():
        record = row.to_dict()
        identity = (
            str(record["cik"]).zfill(10),
            record["accession_number"],
        )
        prior = previous.get(identity)

        if not args.force and downloader.should_skip(record, prior):
            skipped += 1
        else:
            outcome = downloader.download_one(record)
            results[identity] = outcome

            if outcome["status"] == "downloaded":
                downloaded += 1
            else:
                failed += 1

        processed = index + 1

        if processed % 10 == 0 or processed == total:
            atomic_csv(pd.DataFrame(results.values()), manifest_path)
            last_checkpoint = processed

        if processed % args.progress_every == 0 or processed == total:
            elapsed = time.monotonic() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = total - processed
            eta_seconds = remaining / rate if rate > 0 else 0

            print(
                f"[{processed:,}/{total:,}] "
                f"downloaded={downloaded:,}, "
                f"skipped={skipped:,}, "
                f"failed={failed:,} | "
                f"elapsed={elapsed / 60:.1f} min | "
                f"rate={rate * 60:.1f} filings/min | "
                f"ETA={eta_seconds / 60:.1f} min",
                flush=True,
            )

    if last_checkpoint != total:
        atomic_csv(pd.DataFrame(results.values()), manifest_path)

    elapsed = time.monotonic() - start_time
    finished_at = datetime.now(timezone.utc)

    print("\\n========== DOWNLOAD COMPLETE ==========")
    print(f"Started: {started_at.isoformat(timespec='seconds')}")
    print(f"Finished: {finished_at.isoformat(timespec='seconds')}")
    print(f"Elapsed: {elapsed / 60:.2f} minutes ({elapsed / 3600:.2f} hours)")
    print(f"Downloaded this run: {downloaded:,}")
    print(f"Skipped: {skipped:,}")
    print(f"Failed: {failed:,}")
    print(f"Average throughput: {total / elapsed * 60:.2f} filings/min" if elapsed else "Average throughput: N/A")
    print(f"Manifest: {manifest_path}")
    print(f"Total manifest records: {len(results):,}")
    print("=======================================")


if __name__ == "__main__":
    main()
