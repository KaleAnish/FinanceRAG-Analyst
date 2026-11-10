# FinanceRAG MCP server

Lets Claude Desktop (or any MCP client) use FinanceRAG as a tool. It talks to the live API at `api.financerag.anishkale.dev`, so you don't need the index or a Groq key locally.

Tools:

- `list_companies`: what's covered (companies, forms, date range)
- `search_filings`: ranked passages with filing dates and SEC links; no LLM call, so it's fast
- `ask_filings`: a dated, cited answer written only from filing passages, or a refusal when the filings don't support one

## Setup

```bash
pip install -r mcp_server/requirements.txt
```

Add this to Claude Desktop's config (Settings → Developer → Edit Config), using absolute paths:

```json
{
  "mcpServers": {
    "financerag": {
      "command": "/absolute/path/to/FinanceRAG-Analyst/.venv/bin/python",
      "args": ["/absolute/path/to/FinanceRAG-Analyst/mcp_server/financerag_mcp.py"]
    }
  }
}
```

Restart Claude Desktop, then ask something like "use financerag to compare how JPMorgan and Bank of America describe interest rate risk".

To point it at a local or different deployment, set `FINANCERAG_API_URL` in an `env` block in that config.

The public API allows about 3 questions a minute; `ask_filings` returns the wait time if you hit the limit.
