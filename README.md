# FinanceRAG

Ask questions about SEC filings and get answers that point to the exact passage they came from.

**Live:** [financerag.anishkale.dev](https://financerag.anishkale.dev) · **API docs:** [api.financerag.anishkale.dev/docs](https://api.financerag.anishkale.dev/docs) · **Frontend repo:** [KaleAnish/financerag-web](https://github.com/KaleAnish/financerag-web)

![CI/CD](https://github.com/KaleAnish/FinanceRAG-Analyst/actions/workflows/ci.yml/badge.svg)

I built this because most "chat with your PDFs" demos will happily make up a number if the source doesn't have one. For financial filings that's the one thing you can't allow. So the rule here is simple: every factual sentence in an answer has to cite a passage that was actually retrieved, and if the filings don't support an answer, the system says so instead of guessing.

## What's in it

- 2,201 filings from 50 large US companies across 8 sectors
  - 242 10-Ks, 727 10-Qs, 1,232 8-Ks
  - 10-K and 10-Q filings from Nov 2021 to Nov 2026; 8-Ks from Nov 2024 onward
- 161,176 passages after chunking
- Raw filings come from SEC EDGAR and are stored in Cloudflare R2

You can search one company, compare up to five, or search all 50, and filter by form type and date range.

## How it works

```
question
   │
   ├─► question understanding: companies named, time phrases, question type
   │
   ├─► dense search   BGE-small-en-v1.5 embeddings, FAISS (exact inner product), top 30
   ├─► keyword search BM25 (sparse CSR, scores match rank_bm25 exactly), top 20
   │
   └─► Reciprocal Rank Fusion (k = 30) ─► top 5 passages
                                              │
                                              ▼
                         LLM (gpt-oss-120b on Groq) writes a structured answer
                                              │
                                              ▼
                  citation check: every cited label must be one of the 5 passages,
                  and an answer with no citations is rejected and regenerated
                  (up to 3 tries). The model is allowed to abstain.
```

Some details that took longer than they look:

- **Chunking.** Filings are parsed from the raw HTML with lxml. Hidden XBRL headers are dropped and table cells are kept on one line. The text is split into 480-token passages with about 80 tokens of overlap, counted with the same tokenizer the embedding model uses.
- **BM25 at this size.** The usual `rank_bm25` library keeps everything in Python lists, which got slow and memory-hungry at 161K passages. I rewrote the scoring on a sparse matrix and tested it against the original until the scores matched exactly.
- **Reproducible builds.** The index is built in stages (select → fetch → chunk → embed → finalize). Each stage can resume, and the final bundle has a manifest with SHA-256 hashes for every file. The server downloads the bundle from R2 at startup and refuses to start if a hash doesn't match.
- **Reading the question.** You don't have to set filters. "What has Boeing been up to this year?" is turned into a plan before searching:
  - company names, nicknames and tickers map to tickers ("JPMorgan", "BofA", "$NVDA"); words like "AI", "CEO" or "cost" don't;
  - time phrases ("this year", "last quarter", "since 2024", "Q3 2025", "fiscal 2026") become filing-date windows;
  - questions with no time phrase prefer recent filings (a 3-year half-life on top of the relevance ranking), and a second search over the last 12 months makes sure recent filings always compete;
  - comparisons are ranked per company and alternated, so neither side gets crowded out;
  - exhibit indexes, certifications and cover pages are filtered out before ranking;
  - companies that aren't covered get a plain "not covered yet" instead of a search over everything.

  Filters you set by hand always win. All of this is plain Python with no extra model calls (`src/finance_rag/query_understanding.py`).
- **Dates and scope in answers.** The prompt makes the model anchor each claim to its filing and period ("in its 10-K filed January 2024…"), include changes when the passage has them (up 117% year over year), and say when a figure is a segment number rather than a company total. The frontend also shows the date span of the evidence and flags answers whose newest source is more than a year old.

## Evaluation

I didn't want to report a number I couldn't defend, so I followed the setup used in TREC-style retrieval evaluations.

- 50 questions covering single-company facts, quarterly figures, 8-K events, two-company comparisons, and 7 questions the filings can't answer.
- The top results from all three retrievers were pooled and judged blind (no label showing which method retrieved what), using a 0–3 relevance scale. That came to 444 judgments.
- A subset was re-labeled later to check consistency: Cohen's κ = 0.84 with grades collapsed to relevant/not relevant (0.65 weighted on the full 0–3 scale). Disagreements were adjudicated against written guidelines (`eval/ANNOTATION_GUIDELINES.md`).
- 95% confidence intervals come from bootstrap resampling over questions.

Retrieval, top 5:

| Method | MRR@5 | nDCG@5 |
|---|---|---|
| Hybrid (BM25 + dense, RRF) | 0.850 | 0.777 |
| Dense only | 0.891 | 0.780 |
| BM25 only | 0.746 | 0.675 |

Hybrid also reached 97.7% Hit@5 and 74.9% precision@5.

How I read these results:

- Hybrid beats BM25 by 13.9% MRR, and the confidence interval doesn't cross zero.
- Hybrid vs dense is a statistical tie; dense is slightly ahead on MRR, but the interval includes zero.
- I kept hybrid because it covered both companies in more of the comparison questions (6/8, vs 5/8 for each of the others) and holds up better on exact terms like tickers and product names.

Generation, over the same 50 questions:
- 40 of 43 answerable questions were answered, and all 40 had valid citations.
- All 7 unanswerable questions were correctly declined.
- 3 answerable questions were wrongly declined.

Median latency on the server is about 6 seconds.

One thing I want to be clear about: citation validity means every cited passage exists and was retrieved. It does not prove that the passage supports the sentence. Checking that claim by claim is the next evaluation I'm adding.

## Running it locally

You'll need Python 3.12, a Groq API key, and either the R2 bucket credentials or your own built bundle.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add GROQ_API_KEY and the R2_* values
python -m pytest -q
```

To build the search index yourself (this takes a few hours on a laptop, mostly the embedding step):

```bash
python scripts/build_runtime.py --help   # flags, plus worked examples for a sample and the full build
```

On macOS, set `OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE` before starting the API. PyTorch and FAISS ship different OpenMP libraries, and without these the process can hang or crash.

The Dockerfile is the version that runs in production. It installs CPU-only PyTorch, bakes the embedding model into the image, and runs as a non-root user.

## Deployment

- **API:** FastAPI in Docker on an Oracle Cloud ARM instance, behind Caddy for HTTPS. Uptime is checked by UptimeRobot.
- **Rate limits:** questions are limited per IP to 3 a minute and 60 a day, and only one answer is generated at a time. That's because the LLM runs on Groq's free tier. When Groq is out of capacity, the API returns a 429 with `Retry-After`, and the frontend shows a countdown.
- **Frontend:** React and TypeScript on Vercel.
- **CI/CD:** GitHub Actions runs the test suite and a production Docker build on every push and pull request. Pushes to `main` then deploy to the Oracle server over SSH with a dedicated deploy key, and the run checks `/ready` before it counts as done.

## Using it from Claude (MCP)

`mcp_server/` is a small MCP server that exposes the live API as three tools: `list_companies`, `search_filings` and `ask_filings`. Add it to Claude Desktop (setup in [`mcp_server/README.md`](mcp_server/README.md)) and you can ask Claude to look things up in the filings. It gets back the same cited, dated answers as the website.

## Repo layout

```
src/finance_rag/
  api.py              FastAPI app, CORS, rate limiting
  app_service.py      request handling and filters
  retrieval.py        dense + BM25 + RRF, ticker/form/date filters
  generation.py       prompt, structured output, citation validation
  bm25_sparse.py      sparse BM25 implementation
  runtime_remote.py   publish/fetch the index bundle to and from R2
  query_understanding.py  companies, time windows, intent, recency, boilerplate filter
  processing/         HTML extraction, chunking, staged index build
mcp_server/           MCP tools over the live API
.github/workflows/    CI/CD
scripts/              build and publish CLIs
eval/                 questions, labeling app, scoring, agreement, results
config/universe/      the 50 tickers
tests/
```

## Known limitations

- 8-K coverage only starts in November 2024.
- ExxonMobil has only a few months of filings, and BlackRock about two years (most likely because it changed SEC company IDs in a 2024 reorganization). Both are on the fix list.
- Passages aren't tagged by filing section yet, so text from the auditor's report can occasionally be described as the company's own statement.
- Companies with non-calendar fiscal years (Costco, NVIDIA, Apple) can confuse quarter labels. The model sometimes assumes calendar quarters.
- Figures are read from passages, not from structured data. A segment number can occasionally be presented without saying it's a segment.
- I wrote the evaluation questions and did the labeling myself. The κ figure is a consistency check on my own labels, not agreement between two people.
- The free LLM tier allows about two questions a minute.

## What I'm working on next

- Tagging passages by filing section (risk factors, MD&A, auditor's report, exhibits) and routing question types to the right sections.
- Using the SEC's structured XBRL data to check numbers. If a passage says net interest income was $4.8B but the company total is $89B, the system should know it's looking at a segment.
- Aligning fiscal periods across companies, since NVIDIA's fiscal year doesn't line up with JPMorgan's.
- Claim-level faithfulness grading, a second annotator, and a held-out question set written by someone else.
- Indexing earnings press releases (8-K Exhibit 99.1), and extending the corpus toward the roughly 22,000 filings already in R2.

---

Built by [Anish Kale](https://anishkale.dev).
