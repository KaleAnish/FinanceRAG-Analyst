import unittest

from finance_rag.ingestion.filing_discovery import discover_company_filings
from finance_rag.ingestion.sec_client import SECClient
from scripts.discover_sec_filings import _merge_inventory, _merge_status
import pandas as pd


class ShardClient:
    def __init__(self):
        self.shard_calls = []

    def get_submissions(self, cik):
        return {"filings": {
            "recent": {
                "accessionNumber": ["0001-25-000001"],
                "filingDate": ["2025-05-01"],
                "form": ["10-Q"],
                "primaryDocument": ["q.htm"],
            },
            "files": [
                {"name": "old.json", "filingFrom": "2021-01-01", "filingTo": "2021-12-31"},
                {"name": "inrange.json", "filingFrom": "2022-01-01", "filingTo": "2022-12-31"},
            ],
        }}

    def get_submission_history(self, name):
        self.shard_calls.append(name)
        return {
            "accessionNumber": ["0001-22-000001", "0001-25-000001"],
            "filingDate": ["2022-03-01", "2025-05-01"],
            "form": ["10-K", "10-Q"],
            "primaryDocument": ["k.htm", "q.htm"],
        }

    @staticmethod
    def filing_url(cik, accession, primary):
        return f"https://example.test/{accession}/{primary}"


class DiscoveryUpgradeTests(unittest.TestCase):
    def test_per_form_windows_and_shard_deduplication(self):
        client = ShardClient()
        results = discover_company_filings(
            client, "1234", "TEST", "Test Co",
            forms={"10-K", "10-Q"},
            form_windows={"10-K": ("2022-01-01", "2022-12-31"), "10-Q": ("2025-01-01", "2025-12-31")},
        )
        self.assertEqual({row["accession_number"] for row in results}, {"0001-22-000001", "0001-25-000001"})
        self.assertEqual(client.shard_calls, ["inrange.json"])

    def test_empty_forms_rejected(self):
        with self.assertRaises(ValueError):
            discover_company_filings(ShardClient(), "1", "T", "Co", forms=set())

    def test_inverted_date_range_rejected(self):
        with self.assertRaises(ValueError):
            discover_company_filings(ShardClient(), "1", "T", "Co", start_date="2025-12-31", end_date="2025-01-01")

    def test_accession_inventory_merge_is_idempotent(self):
        old = pd.DataFrame([{"cik": "1", "accession_number": "A", "filing_date": "2025-01-01", "ticker": "OLD"}])
        new = pd.DataFrame([{"cik": "1", "accession_number": "A", "filing_date": "2025-01-01", "ticker": "NEW"}, {"cik": "2", "accession_number": "B", "filing_date": "2025-02-01", "ticker": "TWO"}])
        merged = _merge_inventory(old, new)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged.loc[merged["accession_number"] == "A", "ticker"].iloc[0], "NEW")

    def test_latest_issuer_status_replaces_previous_status(self):
        old = pd.DataFrame([{"cik": "1", "status": "failed", "ticker": "ONE"}])
        new = pd.DataFrame([{"cik": "1", "status": "success", "ticker": "ONE"}, {"cik": "2", "status": "failed", "ticker": "TWO"}])
        merged = _merge_status(old, new)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged.loc[merged["cik"] == "1", "status"].iloc[0], "success")

    def test_sec_sessions_have_no_forced_host_header(self):
        client = SECClient(user_agent="Test contact@example.com")
        self.assertIsNot(client.submissions_session, client.archive_session)
        self.assertNotIn("Host", client.submissions_session.headers)
        self.assertNotIn("Host", client.archive_session.headers)
        client.submissions_session.close()
        client.archive_session.close()


if __name__ == "__main__":
    unittest.main()
