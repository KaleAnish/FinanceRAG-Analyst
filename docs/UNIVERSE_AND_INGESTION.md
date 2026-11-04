# Company universe and SEC filing discovery (v1)

This increment adds two separate capabilities:

1. Resolve a sourced constituent list to SEC issuer CIKs and create issuer/security/index-membership outputs.
2. Discover SEC 10-K, 10-Q and 8-K filing metadata for resolved issuers.

It does **not** download filing bodies, parse filings, chunk text, create embeddings or rebuild FAISS/BM25 indexes. Those are the next pipeline stage.

## 1. Prepare a sourced constituent list

Copy `data/universe/constituents_input_template.csv` to
`data/universe/constituents_input.csv` and populate it from a source you are
entitled to use. Required columns:

- `ticker`
- `company`
- `exchange`
- `index_name`

Optional: `share_class`, `source_url`, `membership_start`, `membership_end`,
`as_of_date`.

Use one row per security/index membership. A company in two indexes has two
membership rows. Multiple share classes may have different ticker rows but can
resolve to the same CIK.

**Important:** the SEC ticker map resolves ticker to issuer; it does not establish
S&P 500, Dow Jones or any other index membership. Keep the source URL and
as-of date. Do not infer historical membership from today's constituents.

For an NYSE-listed S&P 500 subset, include S&P constituent rows with `exchange=NYSE`.
This is not the same thing as an independently defined "NYSE 100" index.

## 2. Configure SEC identification

Set an identifying User-Agent with a real contact email:

```bash
export SEC_USER_AGENT="FinanceRAG-Analyst your.name@example.com"
```

Do not commit real secrets. The SEC's public data endpoints require a descriptive
User-Agent; follow SEC fair-access guidance and avoid aggressive concurrency.

## 3. Download official SEC ticker/CIK/exchange mapping

```bash
python scripts/download_sec_ticker_map.py
```

The result is stored at `data/reference/sec_company_tickers_exchange.json`.

## 4. Build the universe

```bash
python scripts/build_company_universe.py \
  --constituents data/universe/constituents_input.csv \
  --sec-map data/reference/sec_company_tickers_exchange.json \
  --output data/universe \
  --as-of-date 2026-11-04
```

Outputs:
- `companies.csv`: one row per resolved issuer CIK.
- `securities.csv`: ticker/exchange/share-class records.
- `index_memberships.csv`: source-backed membership records.
- `sec_issuer_registry.csv`: issuer registry for filing discovery.
- `unresolved_constituents.csv`: tickers not resolved by SEC mapping.
- `universe_build_metadata.json`: counts, inputs and caveats.

## 5. Discover filing inventory

The initial operational target is five fiscal years of 10-Ks, 12-20 quarters
of 10-Qs and 12-24 months of 8-Ks. This script accepts date bounds, so run
separate windows by form when you want different coverage depths.

Example: five-year discovery window across all three forms:

```bash
python scripts/discover_sec_filings.py \
  --universe data/universe/sec_issuer_registry.csv \
  --start-date 2021-01-01 \
  --end-date 2026-11-04 \
  --forms 10-K 10-Q 8-K
```

For a 12-24 month 8-K window, run:

```bash
python scripts/discover_sec_filings.py \
  --universe data/universe/sec_issuer_registry.csv \
  --start-date 2024-11-04 \
  --end-date 2026-11-04 \
  --forms 8-K
```

For a 12-20 quarter 10-Q window, choose a start date about three to five years
back and run `--forms 10-Q`. The exact inventory is determined by filings
actually present at EDGAR, not a fixed count per issuer.

Use `--limit-companies 5` for a small smoke test before running the full universe.

Outputs:
- `data/ingestion/filing_inventory.csv`
- `data/ingestion/filing_discovery_failures.csv`

SEC submissions JSON contains recent filings plus pointers to older submission
shards. The discovery module follows those shards when they overlap the
requested filing-date window and deduplicates by accession number.

## 6. Next implementation stage

After reviewing the inventory:
1. Add idempotent filing-body download by accession number.
2. Store raw filings and normalized text in versioned Cloudflare R2 prefixes.
3. Add parser adapters by form type and exhibit handling.
4. Extend chunk metadata and retrieval filters to support all forms.
5. Build fresh indexes as a new corpus version; retain the frozen five-company
   evaluation corpus and benchmark unchanged.
6. Add Company Facts/XBRL structured financial metrics as a separate data path.

Do not label a company "covered" merely because it exists in `companies.csv`.
Coverage must be measured from successfully downloaded, parsed and indexed
filings.
