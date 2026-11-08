# Deploying FinanceRAG

```
Vercel (Lovable React frontend) ──HTTPS──▶ Hugging Face Space (Docker, FastAPI)
                                              │  start-up: fetch runtime bundle from R2 (SHA-256 verified)
                                              └─ per request: hybrid retrieval in memory → Groq LLM
```

## 1. Publish the runtime bundle to R2 (once per build)

```bash
./.venv/bin/python scripts/publish_runtime.py --version runtime-v2-mvp50
```

## 2. Backend: Hugging Face Space

1. huggingface.co → New Space → SDK **Docker** → template **Blank** → hardware **CPU basic (free)** → name `financerag-api`.
2. Space → Settings → **Variables and secrets**. Add as *secrets*:

| Name | Value |
|---|---|
| `GROQ_API_KEY` | your Groq key |
| `STORAGE_BACKEND` | `r2` |
| `R2_BUCKET`, `R2_ENDPOINT_URL`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_REGION` | same as `.env` (ideally a read-only R2 token) |
| `FINANCE_RAG_RUNTIME_R2_PREFIX` | `runtime/runtime-v2-mvp50` |
| `ALLOWED_ORIGINS` | your frontend URLs, comma-separated, e.g. `https://financerag.vercel.app` |
| `ALLOWED_ORIGIN_REGEX` | optional, e.g. `^https://.*\.lovable\.app$` for Lovable previews |

Optional tuning: `ASK_LIMIT_PER_MINUTE` (default 3), `ASK_LIMIT_PER_DAY` (60), `SEARCH_LIMIT_PER_MINUTE` (30).

3. Push the code: `scripts/prepare_hf_space.sh <hf-username>/financerag-api`
   (HF asks for a username + access token with write scope as the password.)
4. Watch **Logs**: image build (~5–8 min first time), then runtime download, then `Application startup complete`.
5. Check: `https://<hf-username>-financerag-api.hf.space/ready` → `{"status":"ready"}`.

The free tier sleeps after inactivity; the first request after a nap takes ~1 min (the frontend shows a "waking up" state).

## 3. Frontend: Vercel

1. vercel.com → Add New Project → import the GitHub repo Lovable created.
2. Environment variables: `VITE_API_URL=https://<hf-username>-financerag-api.hf.space`, `VITE_USE_MOCK=false`.
3. Deploy, then add the Vercel URL to the Space's `ALLOWED_ORIGINS` and restart the Space.
