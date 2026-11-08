---
title: financerag API
emoji: 📄
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: Citation-grounded hybrid RAG API over 2,201 SEC filings
---

# financerag API

FastAPI backend for [FinanceRAG-Analyst](https://github.com/KaleAnish/FinanceRAG-Analyst):
hybrid retrieval (BGE dense + BM25, reciprocal rank fusion) over 161,176 passages from
2,201 SEC filings (10-K, 10-Q, 8-K) of 50 companies, with citation-validated answers.

Endpoints: `GET /ready`, `GET /v1/corpus`, `POST /v1/search`, `POST /v1/ask` — see `/docs`.

The runtime index is downloaded from Cloudflare R2 at start-up and verified against
its SHA-256 build manifest. Configuration is via Space secrets (see the main repo's
`deploy/DEPLOY.md`).
