import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from finance_rag.universe.builder import build_company_universe


class UniverseBuilderTests(unittest.TestCase):

    def test_builds_issuer_registry_and_preserves_securities(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            constituents = pd.DataFrame([
                {
                    "ticker": "AAA",
                    "company": "Alpha Inc.",
                    "exchange": "NYSE",
                    "index_name": "S&P 500",
                    "source_url": "https://example.com",
                },
                {
                    "ticker": "AAB",
                    "company": "Alpha Inc.",
                    "exchange": "NYSE",
                    "index_name": "S&P 500",
                    "source_url": "https://example.com",
                },
                {
                    "ticker": "BBB",
                    "company": "Beta Inc.",
                    "exchange": "Nasdaq",
                    "index_name": "S&P 500",
                    "source_url": "https://example.com",
                },
            ])

            constituents_path = root / "constituents.csv"
            constituents.to_csv(constituents_path, index=False)

            sec_map = {
                "fields": ["cik", "name", "ticker", "exchange"],
                "data": [
                    [1001, "Alpha Inc.", "AAA", "NYSE"],
                    [1001, "Alpha Inc.", "AAB", "NYSE"],
                    [2002, "Beta Inc.", "BBB", "Nasdaq"],
                ],
            }

            sec_map_path = root / "sec_map.json"
            sec_map_path.write_text(json.dumps(sec_map))

            output = root / "output"

            metadata = build_company_universe(
                constituents_path,
                sec_map_path,
                output,
                as_of_date="2026-11-04",
            )

            companies = pd.read_csv(output / "companies.csv")
            securities = pd.read_csv(output / "securities.csv")
            registry = pd.read_csv(output / "sec_issuer_registry.csv")
            memberships = pd.read_csv(output / "index_memberships.csv")

            self.assertEqual(len(companies), 2)
            self.assertEqual(len(securities), 3)
            self.assertEqual(len(registry), 2)
            self.assertEqual(len(memberships), 3)
            self.assertEqual(metadata["unique_issuers"], 2)
            self.assertEqual(metadata["unresolved_membership_rows"], 0)

            self.assertIn("ticker", registry.columns)
            self.assertIn("sec_exchange", registry.columns)
            self.assertNotIn("primary_ticker", registry.columns)

            alpha = registry[registry["cik"] == 1001].iloc[0]
            self.assertEqual(alpha["ticker"], "AAA")

    def test_retains_unresolved_constituents(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            pd.DataFrame([{
                "ticker": "UNKNOWN",
                "company": "Unknown Inc.",
                "exchange": "NYSE",
                "index_name": "S&P 500",
                "source_url": "https://example.com",
            }]).to_csv(root / "constituents.csv", index=False)

            (root / "sec_map.json").write_text(json.dumps({
                "fields": ["cik", "name", "ticker", "exchange"],
                "data": [],
            }))

            metadata = build_company_universe(
                root / "constituents.csv",
                root / "sec_map.json",
                root / "output",
                as_of_date="2026-11-04",
            )

            unresolved = pd.read_csv(
                root / "output" / "unresolved_constituents.csv"
            )

            self.assertEqual(len(unresolved), 1)
            self.assertEqual(metadata["unresolved_membership_rows"], 1)


if __name__ == "__main__":
    unittest.main()
