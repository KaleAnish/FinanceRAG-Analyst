"""FinanceRAG MCP server.

Exposes the live FinanceRAG API (https://api.financerag.anishkale.dev) as MCP
tools, so Claude Desktop or any MCP client can search SEC filings and get
citation-checked answers.

Tools
  list_companies   companies, forms and date range covered
  search_filings   ranked passages only (fast, no LLM call)
  ask_filings      a dated, cited answer plus its sources

Run (stdio):  python mcp_server/financerag_mcp.py
Point it at another deployment with FINANCERAG_API_URL.
"""

from __future__ import annotations

import os
import re
from typing import Literal, Optional

import httpx

try:  # MCP Python SDK 2.x
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # SDK 1.x
    from mcp.server.fastmcp import FastMCP as _Server

API_URL = os.environ.get("FINANCERAG_API_URL", "https://api.financerag.anishkale.dev").rstrip("/")
TIMEOUT = httpx.Timeout(90.0, connect=10.0)

mcp = _Server(
    "financerag",
    instructions=(
        "Research tools over SEC filings (10-K, 10-Q, 8-K) from 50 large US companies. "
        "Answers cite filing passages as [n]; keep those citations and filing dates when you relay them."
    ),
)

FormType = Optional[Literal["10-K", "10-Q", "8-K"]]


def _request_body(query: str, tickers: Optional[list[str]], form_type: FormType,
                  start_date: Optional[str], end_date: Optional[str], top_k: int) -> dict:
    body: dict = {"query": query, "top_k": max(1, min(int(top_k), 10))}
    if tickers:
        body["tickers"] = [t.strip().upper() for t in tickers if t and t.strip()][:5]
    if form_type:
        body["form_type"] = form_type
    if start_date:
        body["start_date"] = start_date
    if end_date:
        body["end_date"] = end_date
    return body


def _call(method: str, path: str, body: Optional[dict] = None) -> dict:
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.request(method, f"{API_URL}{path}", json=body)
    except httpx.HTTPError as exc:
        return {"error": f"Could not reach FinanceRAG at {API_URL}: {exc}"}
    if response.status_code == 429:
        wait = response.headers.get("Retry-After", "30")
        return {"error": f"Rate limited by FinanceRAG (free LLM tier). Try again in about {wait} seconds."}
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return {"error": f"FinanceRAG returned {response.status_code}: {detail}"}
    return response.json()


def _format_sources(sources: list[dict], with_text: bool, max_chars: int = 900) -> str:
    lines = []
    for i, src in enumerate(sources, start=1):
        head = f"[{i}] {src.get('ticker')} {src.get('form_type')} filed {src.get('filing_date')}"
        if src.get("company"):
            head += f" ({src['company']})"
        if src.get("source_url"):
            head += f"\n    {src['source_url']}"
        lines.append(head)
        if with_text and src.get("text"):
            text = " ".join(str(src["text"]).split())
            lines.append("    " + (text[:max_chars] + "…" if len(text) > max_chars else text))
    return "\n".join(lines)


@mcp.tool()
def list_companies() -> str:
    """List the companies, filing types and date range FinanceRAG covers.
    Call this first if you're unsure whether a company is covered."""
    data = _call("GET", "/v1/corpus")
    if "error" in data:
        return data["error"]
    lines = []
    for key in ("companies", "filings", "passages", "date_range", "form_types"):
        if key in data and not isinstance(data[key], (list, dict)):
            lines.append(f"{key}: {data[key]}")
    for key in ("tickers", "companies_list", "available_tickers"):
        if isinstance(data.get(key), list):
            lines.append(f"{key}: " + ", ".join(map(str, data[key])))
    if isinstance(data.get("form_types"), list):
        lines.append("form types: " + ", ".join(data["form_types"]))
    extra = {k: v for k, v in data.items() if k not in {
        "companies", "filings", "passages", "date_range", "form_types",
        "tickers", "companies_list", "available_tickers"}}
    if extra:
        lines.append(f"other: {extra}")
    return "\n".join(lines) if lines else str(data)


@mcp.tool()
def search_filings(query: str, tickers: Optional[list[str]] = None, form_type: FormType = None,
                   start_date: Optional[str] = None, end_date: Optional[str] = None,
                   top_k: int = 5) -> str:
    """Search SEC filings (10-K, 10-Q, 8-K) from 50 large US companies and return
    the most relevant passages with their filing dates and links. No answer is
    generated, so this is fast and not rate limited by the LLM.

    Args:
        query: what to look for, in plain English.
        tickers: up to 5 tickers, e.g. ["NVDA", "AMD"]. Leave empty to let the
            system detect companies named in the query.
        form_type: "10-K", "10-Q" or "8-K".
        start_date / end_date: filing-date bounds as YYYY-MM-DD.
        top_k: number of passages, 1-10.
    """
    data = _call("POST", "/v1/search", _request_body(query, tickers, form_type, start_date, end_date, top_k))
    if "error" in data:
        return data["error"]
    sources = data.get("sources") or data.get("results") or data.get("evidence") or []
    if not sources:
        return "No matching passages found."
    return _format_sources(sources, with_text=True)


@mcp.tool()
def ask_filings(question: str, tickers: Optional[list[str]] = None, form_type: FormType = None,
                start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """Ask a research question and get an answer written only from SEC filing
    passages, with every claim cited as [n] and dated to its filing. The system
    declines when the filings don't support an answer.

    Companies and time phrases in the question ("this year", "last quarter")
    are detected automatically; pass tickers or dates only to override.
    The public API allows about 3 questions per minute.

    Args:
        question: the research question.
        tickers: up to 5 tickers to restrict to, e.g. ["JPM", "BAC"].
        form_type: "10-K", "10-Q" or "8-K".
        start_date / end_date: filing-date bounds as YYYY-MM-DD.
    """
    data = _call("POST", "/v1/ask", _request_body(question, tickers, form_type, start_date, end_date, 5))
    if "error" in data:
        return data["error"]
    sources = data.get("sources") or []
    if data.get("abstained"):
        reason = data.get("abstention_reason") or "The filings don't support an answer."
        return f"No answer: {reason}" + ("\n\nPassages checked:\n" + _format_sources(sources, False) if sources else "")
    # The API labels citations [S1]; number them [1] to match the source list.
    answer = re.sub(r"\bS(\d+)\b", r"\1", data.get("answer", ""))
    return f"{answer}\n\nSources:\n{_format_sources(sources, with_text=False)}"


if __name__ == "__main__":
    mcp.run()
