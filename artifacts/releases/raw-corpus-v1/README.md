# raw-corpus-v1

Status: FROZEN

This release records the verified raw SEC EDGAR filing corpus used by
FinanceRAG-Analyst.

## Corpus

- Filing inventory rows: 22,431
- Download manifest rows: 22,431
- SHA-256 evidence rows: 22,431
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
