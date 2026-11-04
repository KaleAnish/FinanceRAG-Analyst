from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FILES = {
    "filing_inventory": ROOT / "data/ingestion/filing_inventory_sp500.csv",
    "download_manifest": ROOT / "data/ingestion/download_manifest.csv",
    "sha256_verification": ROOT / "data/ingestion/audit/sha256_verification.csv",
    "audit_summary": ROOT / "data/ingestion/audit/corpus_audit_summary.json",
    "missing_objects": ROOT / "data/ingestion/audit/missing_r2_objects.csv",
    "orphan_objects": ROOT / "data/ingestion/audit/orphan_r2_objects.csv",
    "size_mismatches": ROOT / "data/ingestion/audit/size_mismatches.csv",
}

EXPECTED_FILINGS = 22431


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def write_deterministic_gzip(source: Path, destination: Path) -> None:
    with source.open("rb") as src:
        with destination.open("wb") as raw_dst:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_dst,
                compresslevel=9,
                mtime=0,
            ) as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the verified SEC raw corpus as an immutable release artifact."
    )
    parser.add_argument("--release-id", default="raw-corpus-v1")
    parser.add_argument("--expected-filings", type=int, default=EXPECTED_FILINGS)
    parser.add_argument(
        "--output",
        default="artifacts/releases/raw-corpus-v1",
    )
    args = parser.parse_args()

    output = (ROOT / args.output).resolve()
    if output.exists():
        raise SystemExit(
            f"Release directory already exists: {output}\n"
            "Do not overwrite a frozen release. Choose a new release ID/output."
        )

    missing = [str(path) for path in DEFAULT_FILES.values() if not path.is_file()]
    if missing:
        raise SystemExit("Required source files are missing:\n" + "\n".join(missing))

    inventory_rows = csv_row_count(DEFAULT_FILES["filing_inventory"])
    manifest_rows = csv_row_count(DEFAULT_FILES["download_manifest"])
    checksum_rows = csv_row_count(DEFAULT_FILES["sha256_verification"])

    checks = {
        "inventory_rows": inventory_rows,
        "manifest_rows": manifest_rows,
        "sha256_evidence_rows": checksum_rows,
    }

    for name, count in checks.items():
        if count != args.expected_filings:
            raise SystemExit(
                f"Validation failed: {name}={count}; "
                f"expected {args.expected_filings}. No release created."
            )

    with DEFAULT_FILES["audit_summary"].open("r", encoding="utf-8") as handle:
        audit_summary = json.load(handle)

    if not isinstance(audit_summary, dict):
        raise SystemExit("Audit summary must contain a JSON object.")

    output.mkdir(parents=True)

    compressed_sources = [
        ("filing_inventory.csv.gz", "filing_inventory"),
        ("download_manifest.csv.gz", "download_manifest"),
        ("sha256_verification.csv.gz", "sha256_verification"),
        ("missing_r2_objects.csv.gz", "missing_objects"),
        ("orphan_r2_objects.csv.gz", "orphan_objects"),
        ("size_mismatches.csv.gz", "size_mismatches"),
    ]

    release_files = []

    for destination_name, source_name in compressed_sources:
        destination = output / destination_name
        write_deterministic_gzip(DEFAULT_FILES[source_name], destination)
        release_files.append(destination)

    summary_destination = output / "corpus_audit_summary.json"
    shutil.copy2(DEFAULT_FILES["audit_summary"], summary_destination)
    release_files.append(summary_destination)

    source_evidence = {}
    for name, path in DEFAULT_FILES.items():
        source_evidence[name] = {
            "source_path": str(path.relative_to(ROOT)),
            "source_size_bytes": path.stat().st_size,
            "source_sha256": sha256_file(path),
            "csv_rows": csv_row_count(path) if path.suffix == ".csv" else None,
        }

    metadata = {
        "release_id": args.release_id,
        "release_created_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_status": "FROZEN",
        "corpus": {
            "description": "SEC EDGAR filing corpus for the S&P 500 issuer universe",
            "expected_filing_count": args.expected_filings,
            "inventory_row_count": inventory_rows,
            "download_manifest_row_count": manifest_rows,
            "sha256_verification_row_count": checksum_rows,
            "raw_object_bytes": 32506926931,
            "raw_object_decimal_gb": round(32506926931 / 1_000_000_000, 4),
            "raw_object_gib": round(32506926931 / (1024 ** 3), 4),
            "r2_bucket": "finance-rag-corpus",
            "r2_prefix": "raw/sec/",
        },
        "integrity": {
            "audit_result": "PASS",
            "objects_checked": 22431,
            "sha256_passed": 22431,
            "sha256_mismatched": 0,
            "sha256_errors": 0,
            "missing_objects": 0,
            "orphan_objects": 0,
            "size_mismatches": 0,
            "inventory_manifest_pair_mismatches": 0,
            "note": (
                "Values reflect the completed full-corpus R2 audit. "
                "This freeze operation packages existing evidence; "
                "it does not rerun the remote checksum audit."
            ),
        },
        "provenance": {
            "filing_source": "U.S. Securities and Exchange Commission EDGAR",
            "universe": "S&P 500 constituent input resolved against SEC ticker/CIK mapping",
            "membership_source_caveat": (
                "The constituent input was sourced from Wikipedia; "
                "do not describe membership as independently verified by S&P."
            ),
            "forms": ["10-K", "10-Q", "8-K"],
            "discovery_windows": {
                "10-K": "5 years",
                "10-Q": "5 years",
                "8-K": "2 years",
            },
        },
        "source_evidence": source_evidence,
        "audit_summary_snapshot": audit_summary,
    }

    metadata_path = output / "release_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    release_files.append(metadata_path)

    readme = f"""# {args.release_id}

Status: FROZEN

This release records the verified raw SEC EDGAR filing corpus used by
FinanceRAG-Analyst.

## Corpus

- Filing inventory rows: {inventory_rows:,}
- Download manifest rows: {manifest_rows:,}
- SHA-256 evidence rows: {checksum_rows:,}
- Raw R2 object bytes: 32,506,926,931
- Raw corpus size: 32.51 GB decimal (30.27 GiB)
- R2 bucket: finance-rag-corpus
- R2 prefix: raw/sec/

## Integrity

The completed R2 audit reported:

- Objects checked: 22,431
- SHA-256 matches: 22,431
- SHA-256 mismatches: 0
- Errors: 0
- Missing objects: 0
- Orphan objects: 0
- Size mismatches: 0

The audit evidence is included in this release. This packaging operation
does not re-download or re-hash the remote corpus.

## Provenance caveat

The S&P 500 constituent input was sourced from Wikipedia and resolved
against the SEC ticker/CIK mapping. It is not an independently verified
official S&P constituent history.

## Files

- `filing_inventory.csv.gz`
- `download_manifest.csv.gz`
- `sha256_verification.csv.gz`
- `corpus_audit_summary.json`
- `missing_r2_objects.csv.gz`
- `orphan_r2_objects.csv.gz`
- `size_mismatches.csv.gz`
- `release_metadata.json`
- `RELEASE_SHA256SUMS`

Raw filing documents and R2 credentials are not included.
"""
    readme_path = output / "README.md"
    readme_path.write_text(readme, encoding="utf-8")
    release_files.append(readme_path)

    checksum_lines = []
    for path in sorted(release_files):
        checksum_lines.append(f"{sha256_file(path)}  {path.name}")

    checksum_path = output / "RELEASE_SHA256SUMS"
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    print("=" * 72)
    print(f"FROZEN RELEASE CREATED: {args.release_id}")
    print("=" * 72)
    print(f"Location: {output}")
    print(f"Inventory rows: {inventory_rows:,}")
    print(f"Manifest rows: {manifest_rows:,}")
    print(f"SHA evidence rows: {checksum_rows:,}")
    print(f"Release files: {len(release_files) + 1}")
    print(f"Release checksum index: {checksum_path.name}")
    print("\nNo R2 objects were modified.")
    print("No raw filing documents or credentials were copied.")
    print("\nVerify the release with:")
    print(
        f"cd '{output}' && "
        "shasum -a 256 -c RELEASE_SHA256SUMS"
    )


if __name__ == "__main__":
    main()
