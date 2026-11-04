from __future__ import annotations

import json
from pathlib import Path

import requests

URL = "https://www.sec.gov/files/company_tickers_exchange.json"


def main():
    out = Path("data/reference/sec_company_tickers_exchange.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    user_agent = __import__("os").getenv("SEC_USER_AGENT")
    if not user_agent or "@" not in user_agent:
        raise SystemExit("Set SEC_USER_AGENT to your application name and real contact email.")
    response = requests.get(
        URL,
        headers={"User-Agent": user_agent},
        timeout=45,
    )
    response.raise_for_status()
    out.write_text(json.dumps(response.json(), indent=2), encoding="utf-8")
    print(f"Saved SEC ticker mapping to {out}")


if __name__ == "__main__":
    main()
