import unittest

from finance_rag.ingestion.filing_discovery import (
    discover_company_filings,
)


class FakeSECClient:
    def get_submissions(self, cik):
        return {
            "filings": {
                "recent": {
                    "accessionNumber": [
                        "0000000001-25-000001",
                        "0000000001-25-000002",
                        "0000000001-25-000003",
                    ],
                    "filingDate": [
                        "2025-02-01",
                        "2025-05-01",
                        "2025-06-01",
                    ],
                    "reportDate": [
                        "2024-12-31",
                        "2025-03-31",
                        "2025-05-20",
                    ],
                    "acceptanceDateTime": [None, None, None],
                    "form": ["10-K", "10-Q", "8-K"],
                    "primaryDocument": [
                        "annual.htm",
                        "quarterly.htm",
                        "event.htm",
                    ],
                    "primaryDocDescription": [None, None, None],
                    "isXBRL": [1, 1, 1],
                    "isInlineXBRL": [1, 1, 1],
                },
                "files": [],
            }
        }

    @staticmethod
    def filing_url(cik, accession, primary):
        return f"https://example.com/{accession}/{primary}"


class FilingDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeSECClient()

    def test_discovers_all_supported_forms(self):
        results = discover_company_filings(
            self.client,
            "1234",
            "TEST",
            "Test Company",
        )
        self.assertEqual(len(results), 3)
        self.assertEqual(
            {row["form_type"] for row in results},
            {"10-K", "10-Q", "8-K"},
        )

    def test_date_filter(self):
        results = discover_company_filings(
            self.client,
            "1234",
            "TEST",
            "Test Company",
            start_date="2025-03-01",
            end_date="2025-05-31",
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["form_type"], "10-Q")

    def test_rejects_unsupported_form(self):
        with self.assertRaises(ValueError):
            discover_company_filings(
                self.client,
                "1234",
                "TEST",
                "Test Company",
                forms={"S-1"},
            )


if __name__ == "__main__":
    unittest.main()
