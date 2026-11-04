from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from datetime import datetime, timezone
import os
import tempfile

import pandas as pd

from finance_rag.ingestion.sec_client import SECClient
from finance_rag.ingestion.filing_discovery import discover_company_filings, SUPPORTED_FORMS


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", suffix=".tmp", dir=destination.parent, delete=False) as handle:
        temp_path = Path(handle.name)
        frame.to_csv(handle, index=False)
    temp_path.replace(destination)


def _merge_inventory(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    if existing.empty and incoming.empty:
        return pd.DataFrame(columns=[
            "cik", "ticker", "company", "form_type", "accession_number", "filing_date",
            "report_date", "acceptance_datetime", "primary_document",
            "primary_document_description", "is_xbrl", "is_inline_xbrl", "source_url", "ingestion_status",
        ])
    combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
    if not {"cik", "accession_number"}.issubset(combined.columns):
        raise ValueError("Existing filing inventory lacks CIK/accession_number keys; back it up and inspect it before continuing.")
    combined = combined.drop_duplicates(["cik", "accession_number"], keep="last")
    if "filing_date" in combined.columns:
        combined = combined.sort_values("filing_date", ascending=False, na_position="last")
    return combined.reset_index(drop=True)


def _merge_status(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
    if combined.empty:
        return pd.DataFrame(columns=["cik", "ticker", "company", "attempted_at_utc", "status", "filing_count", "error_type", "error"])
    combined = combined.drop_duplicates(["cik"], keep="last")
    return combined.sort_values(["ticker", "cik"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover SEC filing inventory with per-form windows and idempotent output merging.")
    parser.add_argument("--universe", default="data/universe/sec_issuer_registry.csv")
    parser.add_argument("--output", default="data/ingestion/filing_inventory.csv")
    parser.add_argument("--start-date", default=None, help="Fallback YYYY-MM-DD start date for forms without a form-specific start.")
    parser.add_argument("--end-date", default=None, help="Fallback YYYY-MM-DD end date for forms without a form-specific end.")
    for form, flag in (("10-K", "10k"), ("10-Q", "10q"), ("8-K", "8k")):
        parser.add_argument(f"--{flag}-start-date", dest={"10k": "tenk_start_date", "10q": "tenq_start_date", "8k": "eightk_start_date"}[flag], default=None, help=f"{form} filing start date YYYY-MM-DD")
        parser.add_argument(f"--{flag}-end-date", dest={"10k": "tenk_end_date", "10q": "tenq_end_date", "8k": "eightk_end_date"}[flag], default=None, help=f"{form} filing end date YYYY-MM-DD")
    parser.add_argument("--forms", nargs="+", default=["10-K", "10-Q", "8-K"])
    parser.add_argument("--limit-companies", type=int, default=None)
    args = parser.parse_args()

    if not os.getenv("SEC_USER_AGENT"):
        raise SystemExit("Set SEC_USER_AGENT='YourApp your.email@example.com' before running.")
    forms = {form.strip().upper() for form in args.forms}
    if not forms or forms - SUPPORTED_FORMS:
        raise SystemExit(f"--forms must contain one or more of: {', '.join(sorted(SUPPORTED_FORMS))}")
    if args.limit_companies is not None and args.limit_companies < 1:
        raise SystemExit("--limit-companies must be a positive integer.")

    form_windows = {
        "10-K": (args.tenk_start_date, args.tenk_end_date),
        "10-Q": (args.tenq_start_date, args.tenq_end_date),
        "8-K": (args.eightk_start_date, args.eightk_end_date),
    }
    form_windows = {form: window for form, window in form_windows.items() if form in forms and any(window)}

    universe = pd.read_csv(args.universe, dtype=str).fillna("")
    required = {"cik", "ticker", "company"}
    missing = required - set(universe.columns)
    if missing:
        raise SystemExit(f"Universe file missing required columns: {sorted(missing)}")
    if args.limit_companies is not None:
        universe = universe.head(args.limit_companies)

    client = SECClient()
    rows: list[dict] = []
    statuses: list[dict] = []
    for index, row in universe.reset_index(drop=True).iterrows():
        cik = str(row["cik"]).strip().zfill(10)
        ticker = row["ticker"]
        company = row["company"]
        attempted_at = datetime.now(timezone.utc).isoformat()
        try:
            found = discover_company_filings(
                client, cik, ticker, company,
                forms=forms,
                start_date=args.start_date,
                end_date=args.end_date,
                form_windows=form_windows,
            )
            rows.extend(found)
            statuses.append({
                "cik": cik, "ticker": ticker, "company": company,
                "attempted_at_utc": attempted_at, "status": "success",
                "filing_count": len(found), "error_type": "", "error": "",
            })
            print(f"[{index + 1}/{len(universe)}] {ticker}: {len(found)} filings")
        except Exception as exc:
            statuses.append({
                "cik": cik, "ticker": ticker, "company": company,
                "attempted_at_utc": attempted_at, "status": "failed",
                "filing_count": 0, "error_type": type(exc).__name__, "error": str(exc),
            })
            print(f"[{index + 1}/{len(universe)}] {ticker}: FAILED - {exc}")

    output = Path(args.output)
    existing_inventory = pd.read_csv(output, dtype=str).fillna("") if output.exists() else pd.DataFrame()
    merged_inventory = _merge_inventory(existing_inventory, pd.DataFrame(rows))
    _atomic_csv(merged_inventory, output)

    status_path = output.with_name("filing_discovery_status.csv")
    existing_status = pd.read_csv(status_path, dtype=str).fillna("") if status_path.exists() else pd.DataFrame()
    merged_status = _merge_status(existing_status, pd.DataFrame(statuses))
    _atomic_csv(merged_status, status_path)

    failed = sum(status["status"] == "failed" for status in statuses)
    print(f"\nThis run: {len(rows)} filings across {len(universe)} issuers; failures: {failed}.")
    print(f"Cumulative inventory: {len(merged_inventory)} unique issuer/accession records: {output}")
    print(f"Latest issuer status: {status_path}")


if __name__ == "__main__":
    main()
