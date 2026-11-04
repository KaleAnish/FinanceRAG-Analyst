from __future__ import annotations

import os
import time
from typing import Any
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class SECClient:
    """SEC submissions/archive HTTP client with shared pacing and retry policy."""

    BASE = "https://data.sec.gov"
    ARCHIVES = "https://www.sec.gov/Archives/edgar/data"

    def __init__(self, user_agent: str | None = None, min_interval: float = 0.12):
        self.user_agent = user_agent or os.getenv("SEC_USER_AGENT")
        if not self.user_agent or "@" not in self.user_agent:
            raise ValueError("Set SEC_USER_AGENT to identify your application and include a contact email.")
        self.min_interval = max(float(min_interval), 0.11)
        self._last_request = 0.0
        self.submissions_session = self._make_session()
        self.archive_session = self._make_session()
        # Backwards-compatible alias for code that used client.session.
        self.session = self.submissions_session

    def _make_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
        retry = Retry(
            total=5, connect=5, read=5, status=5,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            respect_retry_after_header=True,
        )
        session.mount("https://", HTTPAdapter(max_retries=retry))
        return session

    def _get(self, session: requests.Session, url: str, *, stream: bool = False):
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            response = session.get(url, timeout=60, stream=stream)
            response.raise_for_status()
            return response
        finally:
            self._last_request = time.monotonic()

    def get_json(self, url: str) -> dict[str, Any]:
        response = self._get(self.submissions_session, url)
        try:
            return response.json()
        finally:
            response.close()

    def get_submissions(self, cik: str) -> dict[str, Any]:
        cik10 = str(cik).zfill(10)
        return self.get_json(f"{self.BASE}/submissions/CIK{cik10}.json")

    def get_submission_history(self, relative_path: str) -> dict[str, Any]:
        parts = relative_path.replace("\\", "/").split("/")
        if not relative_path or relative_path.startswith("/") or any(part in {"", ".", ".."} for part in parts):
            raise ValueError("Invalid SEC submissions history path.")
        return self.get_json(f"{self.BASE}/submissions/{quote(relative_path, safe='')}")

    def get_archive_response(self, url: str, *, stream: bool = True):
        """Return a successful archive response; caller must close it."""
        if not url.startswith(self.ARCHIVES + "/"):
            raise ValueError("Archive URL must use the SEC EDGAR Archives host and path.")
        return self._get(self.archive_session, url, stream=stream)

    @classmethod
    def filing_url(cls, cik: str, accession_number: str, primary_document: str) -> str:
        accession = accession_number.replace("-", "")
        cik_int = str(int(cik))
        return f"{cls.ARCHIVES}/{cik_int}/{accession}/{quote(primary_document, safe='')}"
