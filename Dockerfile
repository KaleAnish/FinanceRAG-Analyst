# FinanceRAG API — container for Hugging Face Spaces (Docker) or Google Cloud Run.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/app/.cache/huggingface \
    FINANCE_RAG_PROJECT_DIR=/app \
    FINANCE_RAG_RUNTIME_DIR=/app/runtime \
    PORT=7860

WORKDIR /app

# CPU-only PyTorch first (the default wheel bundles CUDA and is ~5x larger).
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# Bake the embedding model into the image so start-up never depends on the HF Hub.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY src/ src/
COPY config/ config/
ENV PYTHONPATH=/app/src

# Non-root user (required by HF Spaces, good practice everywhere).
RUN useradd -m -u 1000 app && mkdir -p /app/runtime && chown -R app:app /app
USER app

EXPOSE 7860
CMD ["sh", "-c", "uvicorn finance_rag.api:app --host 0.0.0.0 --port ${PORT} --workers 1"]
