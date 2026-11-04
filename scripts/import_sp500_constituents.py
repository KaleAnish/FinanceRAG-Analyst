from __future__ import annotations

import json
import re
from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
import os
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SEC_MAP_PATH = ROOT / "data/reference/sec_company_tickers_exchange.json"
OUTPUT_PATH = ROOT / "data/universe/sp500_constituents.csv"


def normalize_cik(value):
    if pd.isna(value):
        return None
    digits = re.sub(r"\D", "", str(value).split(".")[0])
    return digits.zfill(10) if digits else None


def normalize_ticker(value):
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def load_sec_map():
    payload = json.loads(SEC_MAP_PATH.read_text())

    if isinstance(payload, dict) and "fields" in payload and "data" in payload:
        records = [
            dict(zip(payload["fields"], row))
            for row in payload["data"]
        ]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("Unrecognized SEC ticker-map structure")

    by_cik = {}

    for record in records:
        cik = normalize_cik(record.get("cik"))
        if cik:
            by_cik.setdefault(cik, []).append(record)

    return by_cik


def main():
    if not SEC_MAP_PATH.exists():
        raise FileNotFoundError(
            f"SEC ticker map missing: {SEC_MAP_PATH}"
        )

    print("Fetching S&P 500 constituent table...")

    response = requests.get(
        SOURCE_URL,
        headers={"User-Agent": (load_dotenv(), os.environ["SEC_USER_AGENT"])[1]},
        timeout=45,
    )
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))

    constituent_table = None

    for table in tables:
        columns = {str(column).strip().lower() for column in table.columns}
        if {"symbol", "security", "cik"}.issubset(columns):
            constituent_table = table.copy()
            break

    if constituent_table is None:
        raise RuntimeError(
            "Could not identify the S&P 500 table with Symbol, Security and CIK columns."
        )

    sec_by_cik = load_sec_map()

    rows = []
    unresolved = []

    for _, record in constituent_table.iterrows():
        ticker = str(record["Symbol"]).strip()
        company = str(record["Security"]).strip()
        cik = normalize_cik(record["CIK"])

        candidates = sec_by_cik.get(cik, []) if cik else []

        matching = [
            candidate for candidate in candidates
            if normalize_ticker(candidate.get("ticker", "")) == normalize_ticker(ticker)
        ]

        if len(matching) == 1:
            sec_record = matching[0]
        elif len(candidates) == 1:
            sec_record = candidates[0]
        else:
            sec_record = None

        exchange = (
            str(sec_record.get("exchange", "")).strip()
            if sec_record else ""
        )

        if not sec_record:
            unresolved.append({
                "ticker": ticker,
                "company": company,
                "cik": cik or "",
                "reason": "No unique SEC ticker/exchange match",
            })

        rows.append({
            "ticker": ticker,
            "company": company,
            "exchange": exchange,
            "index_name": "S&P 500",
            "source_url": SOURCE_URL,
            "as_of_date": date.today().isoformat(),
        })

    output = pd.DataFrame(rows)
    output.to_csv(OUTPUT_PATH, index=False)

    unresolved_path = ROOT / "data/universe/sp500_unresolved_sec_matches.csv"
    pd.DataFrame(
        unresolved,
        columns=["ticker", "company", "cik", "reason"],
    ).to_csv(unresolved_path, index=False)

    print(f"\nConstituent rows: {len(output)}")
    print(f"Unique tickers: {output['ticker'].nunique()}")
    print(f"Resolved SEC exchange: {(output['exchange'] != '').sum()}")
    print(f"Unresolved SEC matches: {len(unresolved)}")
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Saved: {unresolved_path}")

    if len(output) < 490 or len(output) > 510:
        raise RuntimeError(
            f"Unexpected constituent count: {len(output)}. Review source table."
        )


if __name__ == "__main__":
    main()
