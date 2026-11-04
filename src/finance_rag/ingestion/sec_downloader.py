from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import PurePosixPath
from urllib.parse import urlparse


class SECFilingDownloader:
    def __init__(self, client, storage):
        self.client = client
        self.storage = storage

    @staticmethod
    def object_key(row: dict) -> str:
        cik = str(row["cik"]).lstrip("0") or "0"
        accession = str(row["accession_number"]).replace("-", "")
        filename = str(row["primary_document"])

        if not re.fullmatch(r"\d+", cik):
            raise ValueError("Invalid CIK")

        if not re.fullmatch(r"\d{18}", accession):
            raise ValueError("Invalid accession number")

        path = PurePosixPath(filename)
        if (
            not filename
            or path.is_absolute()
            or len(path.parts) != 1
            or filename in {".", ".."}
            or "\\" in filename
        ):
            raise ValueError("Invalid primary document filename")

        return f"raw/sec/{cik}/{accession}/{filename}"

    @staticmethod
    def _validate_url(url: str) -> None:
        parsed = urlparse(url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "www.sec.gov"
            or not parsed.path.startswith("/Archives/edgar/data/")
        ):
            raise ValueError("Filing URL is not an SEC EDGAR archive URL")

    def download_one(self, row: dict) -> dict:
        started = datetime.now(timezone.utc).isoformat()
        key = self.object_key(row)

        result = {
            "cik": str(row["cik"]).zfill(10),
            "ticker": row["ticker"],
            "form_type": row["form_type"],
            "accession_number": row["accession_number"],
            "filing_date": row["filing_date"],
            "source_url": row["source_url"],
            "object_key": key,
            "sha256": "",
            "size_bytes": 0,
            "downloaded_at_utc": started,
            "status": "failed",
            "error_type": "",
            "error": "",
        }

        response = None

        try:
            self._validate_url(row["source_url"])

            response = self.client.get_archive_response(
                row["source_url"],
                stream=True,
            )

            content = bytearray()
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    content.extend(chunk)

            if not content:
                raise ValueError("SEC returned an empty filing document")

            payload = bytes(content)
            digest = hashlib.sha256(payload).hexdigest()

            self.storage.put_bytes(
                key,
                payload,
                content_type=response.headers.get(
                    "Content-Type",
                    "application/octet-stream",
                ),
            )

            result.update({
                "sha256": digest,
                "size_bytes": len(payload),
                "status": "downloaded",
            })

        except Exception as exc:
            result.update({
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            })

        finally:
            if response is not None:
                response.close()

        return result

    def should_skip(self, row: dict, previous: dict | None) -> bool:
        if not previous or previous.get("status") != "downloaded":
            return False

        key = previous.get("object_key")
        if not key:
            return False

        return self.storage.exists(key)
