from __future__ import annotations

from datetime import date
from itertools import zip_longest
from typing import Any, Mapping

SUPPORTED_FORMS = {"10-K", "10-Q", "8-K"}


def _rows_from_recent(recent: Mapping[str, list]) -> list[dict[str, Any]]:
    """Convert SEC parallel arrays to records without silently misaligning them."""
    keys = (
        "accessionNumber", "filingDate", "reportDate", "acceptanceDateTime",
        "form", "primaryDocument", "primaryDocDescription", "isXBRL", "isInlineXBRL",
    )
    arrays = [recent.get(key) or [] for key in keys]
    count = max((len(values) for values in arrays), default=0)
    padded = [list(values) + [None] * (count - len(values)) for values in arrays]
    return [dict(zip(keys, values)) for values in zip(*padded)]


def _parse_window(start_date: str | None, end_date: str | None) -> tuple[date | None, date | None]:
    start = date.fromisoformat(start_date) if start_date else None
    end = date.fromisoformat(end_date) if end_date else None
    if start and end and start > end:
        raise ValueError(f"start_date ({start}) must not be after end_date ({end}).")
    return start, end


def discover_company_filings(
    client,
    cik: str,
    ticker: str,
    company: str,
    *,
    forms: set[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    form_windows: Mapping[str, tuple[str | None, str | None]] | None = None,
) -> list[dict[str, Any]]:
    """Discover SEC filing metadata, including applicable historical submission shards.

    `form_windows` maps each form to its own (start_date, end_date). The common
    start/end arguments remain supported for backwards compatibility and act as
    defaults for forms without an explicit window.
    """
    selected_forms = set(SUPPORTED_FORMS if forms is None else forms)
    selected_forms = {str(form).strip().upper() for form in selected_forms}
    if not selected_forms:
        raise ValueError("At least one filing form must be selected.")
    unknown = selected_forms - SUPPORTED_FORMS
    if unknown:
        raise ValueError(f"Unsupported forms: {sorted(unknown)}")

    default_start, default_end = _parse_window(start_date, end_date)
    windows: dict[str, tuple[date | None, date | None]] = {}
    supplied_windows = form_windows or {}
    unknown_windows = set(supplied_windows) - SUPPORTED_FORMS
    if unknown_windows:
        raise ValueError(f"Unsupported form windows: {sorted(unknown_windows)}")
    for form in selected_forms:
        raw_start, raw_end = supplied_windows.get(form, (start_date, end_date))
        start, end = _parse_window(raw_start, raw_end)
        if raw_start is None:
            start = default_start
        if raw_end is None:
            end = default_end
        windows[form] = (start, end)

    submissions = client.get_submissions(cik)
    filings = submissions.get("filings", {})
    records = _rows_from_recent(filings.get("recent", {}))

    # A shard is needed if it overlaps at least one selected form's date window.
    for shard in filings.get("files", []):
        shard_name = shard.get("name")
        if not shard_name:
            continue
        filing_from_raw = shard.get("filingFrom")
        filing_to_raw = shard.get("filingTo")
        filing_from = date.fromisoformat(filing_from_raw) if filing_from_raw else None
        filing_to = date.fromisoformat(filing_to_raw) if filing_to_raw else None
        overlaps = False
        for start, end in windows.values():
            if start and filing_to and filing_to < start:
                continue
            if end and filing_from and filing_from > end:
                continue
            overlaps = True
            break
        if overlaps:
            records.extend(_rows_from_recent(client.get_submission_history(shard_name)))

    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in records:
        form = str(row.get("form") or "").upper()
        filing_date = row.get("filingDate")
        accession = row.get("accessionNumber")
        primary = row.get("primaryDocument")
        if form not in selected_forms or not filing_date or not accession or not primary:
            continue
        filing_day = date.fromisoformat(filing_date)
        start, end = windows[form]
        if start and filing_day < start:
            continue
        if end and filing_day > end:
            continue
        accession = str(accession)
        if accession in seen:
            continue
        seen.add(accession)
        output.append({
            "cik": str(cik).zfill(10),
            "ticker": ticker,
            "company": company,
            "form_type": form,
            "accession_number": accession,
            "filing_date": filing_date,
            "report_date": row.get("reportDate"),
            "acceptance_datetime": row.get("acceptanceDateTime"),
            "primary_document": str(primary),
            "primary_document_description": row.get("primaryDocDescription"),
            "is_xbrl": row.get("isXBRL"),
            "is_inline_xbrl": row.get("isInlineXBRL"),
            "source_url": client.filing_url(cik, accession, str(primary)),
            "ingestion_status": "discovered",
        })
    return sorted(output, key=lambda row: (row["filing_date"], row["ticker"], row["form_type"]), reverse=True)
