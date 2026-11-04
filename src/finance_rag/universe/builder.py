from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
REQUIRED_INPUT_COLUMNS = {"ticker", "company", "exchange", "index_name"}


def normalize_ticker(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().upper())


def load_sec_ticker_map(records: dict | list) -> dict[str, dict]:
    """Convert SEC company_tickers_exchange.json into a ticker-keyed lookup."""
    if isinstance(records, dict) and "data" in records:
        headers = records.get("fields", [])
        rows = records["data"]
        raw = [dict(zip(headers, row)) for row in rows]
    elif isinstance(records, list):
        raw = records
    else:
        raise ValueError("Unexpected SEC ticker-map JSON structure.")

    result: dict[str, dict] = {}
    for row in raw:
        ticker = normalize_ticker(row.get("ticker")).replace("-", ".")
        if not ticker:
            continue
        cik = str(row.get("cik", "")).strip()
        if cik.isdigit():
            cik = cik.zfill(10)
        result[ticker] = {
            "cik": cik,
            "sec_title": row.get("name") or row.get("title"),
            "sec_exchange": row.get("exchange"),
        }
    return result


def build_company_universe(
    constituents_csv: str | Path,
    sec_ticker_map_json: str | Path,
    output_dir: str | Path,
    *,
    as_of_date: str | None = None,
) -> dict[str, int]:
    """
    Build issuer, security, index-membership, SEC registry and unresolved outputs.

    Input must be a sourced constituent file. This function deliberately does
    not scrape or invent current index membership.
    """
    source = Path(constituents_csv)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source, dtype=str).fillna("")
    missing = REQUIRED_INPUT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Constituent CSV missing columns: {sorted(missing)}")

    with Path(sec_ticker_map_json).open(encoding="utf-8") as f:
        sec_map = load_sec_ticker_map(json.load(f))

    df["ticker"] = df["ticker"].map(normalize_ticker)
    df["exchange"] = df["exchange"].str.strip().str.upper()
    df["index_name"] = df["index_name"].str.strip()
    df["company"] = df["company"].str.strip()
    if "share_class" not in df:
        df["share_class"] = ""
    if "source_url" not in df:
        df["source_url"] = ""
    if "membership_start" not in df:
        df["membership_start"] = ""
    if "membership_end" not in df:
        df["membership_end"] = ""
    if "as_of_date" not in df:
        df["as_of_date"] = as_of_date or datetime.now(timezone.utc).date().isoformat()
    elif as_of_date:
        df["as_of_date"] = as_of_date

    if df["ticker"].eq("").any():
        raise ValueError("Constituent input contains blank tickers.")
    if df.duplicated(["ticker", "exchange", "index_name"]).any():
        raise ValueError("Duplicate ticker/exchange/index membership rows found.")

    mapped = df["ticker"].map(sec_map)
    df["cik"] = mapped.map(lambda x: x["cik"] if isinstance(x, dict) else "")
    df["sec_title"] = mapped.map(lambda x: x["sec_title"] if isinstance(x, dict) else "")
    df["sec_exchange"] = mapped.map(lambda x: x["sec_exchange"] if isinstance(x, dict) else "")

    unresolved = df[df["cik"].eq("")].copy()
    unresolved["resolution_status"] = "unresolved_ticker"
    resolved = df[~df["cik"].eq("")].copy()

    companies = (
        resolved.sort_values(["cik", "ticker"])
        .groupby("cik", as_index=False)
        .agg(
            company=("company", "first"),
            sec_title=("sec_title", "first"),
            primary_ticker=("ticker", "first"),
            issuer_exchanges=("exchange", lambda s: "|".join(sorted(set(x for x in s if x)))),
        )
    )

    securities = (
        resolved[["cik", "ticker", "exchange", "share_class"]]
        .drop_duplicates()
        .sort_values(["cik", "ticker"])
    )

    memberships = resolved[
        ["cik", "ticker", "exchange", "index_name", "membership_start",
         "membership_end", "as_of_date", "source_url"]
    ].drop_duplicates()

    registry = (
        resolved.sort_values(["cik", "ticker"])
        .groupby("cik", as_index=False)
        .agg(
            company=("company", "first"),
            sec_title=("sec_title", "first"),
            ticker=("ticker", "first"),
            sec_exchange=("sec_exchange", "first"),
        )
        .sort_values("cik")
    )

    outputs = {
        "companies.csv": companies,
        "securities.csv": securities,
        "index_memberships.csv": memberships,
        "sec_issuer_registry.csv": registry,
        "unresolved_constituents.csv": unresolved,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output / filename, index=False)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "as_of_date": as_of_date,
        "input_file": source.name,
        "sec_ticker_map_file": Path(sec_ticker_map_json).name,
        "input_rows": int(len(df)),
        "resolved_membership_rows": int(len(resolved)),
        "unresolved_membership_rows": int(len(unresolved)),
        "unique_issuers": int(companies["cik"].nunique()),
        "unique_securities": int(securities[["ticker", "exchange"]].drop_duplicates().shape[0]),
        "indexes": sorted(df["index_name"].unique().tolist()),
        "membership_source_warning": (
            "Membership is only as authoritative as the supplied constituent CSV. "
            "SEC ticker mapping resolves issuer identity; it does not establish index membership."
        ),
    }
    (output / "universe_build_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata
