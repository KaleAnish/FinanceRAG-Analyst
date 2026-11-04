

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import argparse
import json

from finance_rag.universe.builder import build_company_universe


def main():
    parser = argparse.ArgumentParser(description="Build a sourced company/security universe.")
    parser.add_argument("--constituents", default="data/universe/constituents_input.csv")
    parser.add_argument("--sec-map", default="data/reference/sec_company_tickers_exchange.json")
    parser.add_argument("--output", default="data/universe")
    parser.add_argument("--as-of-date", default=None)
    args = parser.parse_args()

    result = build_company_universe(
        args.constituents, args.sec_map, args.output, as_of_date=args.as_of_date
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
