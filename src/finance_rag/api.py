from contextlib import asynccontextmanager
import logging
import os
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(
    os.getenv(
        "FINANCE_RAG_PROJECT_DIR",
        str(Path(__file__).resolve().parents[2]),
    )
)


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=1200)
    ticker: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9.-]{0,9}$",
    )
    form_type: Literal["10-K"] = "10-K"
    filing_date: date | None = None
    method: Literal["hybrid_rrf", "dense", "bm25"] = "hybrid_rrf"
    top_k: int = Field(default=5, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Enter a research question.")
        return value


def _clean_value(value):
    if value is None:
        return None

    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass

    try:
        if pd.isna(value):
            return None
    except (ValueError, TypeError):
        pass

    if isinstance(value, date):
        return value.isoformat()

    return value


def _retrieval_records(evidence: pd.DataFrame) -> list[dict]:
    source_columns = [
        "chunk_id",
        "ticker",
        "company",
        "form_type",
        "filing_date",
        "document_id",
        "source_url",
        "rank",
        "retrieval_score",
        "dense_rank",
        "bm25_rank",
    ]

    records = []

    for _, row in evidence.iterrows():
        record = {}

        for column in source_columns:
            if column in evidence.columns:
                record[column] = _clean_value(row[column])

        records.append(record)

    return records


def create_app(service_factory=None):

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = None
        app.state.startup_error = None

        try:
            factory = service_factory

            if factory is None:
                from .app_service import FinanceRAGService
                factory = FinanceRAGService

            app.state.service = factory(project_dir=PROJECT_DIR)
            logger.info("FinanceRAG service initialized.")

        except Exception as exc:
            app.state.startup_error = exc
            logger.exception("FinanceRAG service initialization failed.")

        yield

    application = FastAPI(
        title="FinanceRAG Analyst API",
        version="0.1.0",
        description="Grounded financial research over SEC filings.",
        lifespan=lifespan,
    )

    def get_service():
        service = application.state.service

        if service is None:
            raise HTTPException(
                status_code=503,
                detail="FinanceRAG service is not ready.",
            )

        return service

    @application.get("/health")
    def health():
        return {"status": "alive"}

    @application.get("/ready")
    def ready():
        if application.state.service is None:
            raise HTTPException(
                status_code=503,
                detail="FinanceRAG service is not ready.",
            )

        return {"status": "ready"}

    @application.post("/v1/search")
    def search(request: ResearchRequest):
        service = get_service()

        try:
            evidence = service.retriever.search(
                request.query,
                method=request.method,
                ticker=request.ticker.upper() if request.ticker else None,
                form_type=request.form_type,
                filing_date=(
                    request.filing_date.isoformat()
                    if request.filing_date
                    else None
                ),
                top_k=request.top_k,
            )

            sources = _retrieval_records(evidence)

            return {
                "query": request.query,
                "method": request.method,
                "count": len(sources),
                "sources": sources,
            }

        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc

        except Exception as exc:
            logger.exception("Retrieval request failed.")

            raise HTTPException(
                status_code=500,
                detail="An internal retrieval error occurred.",
            ) from exc

    @application.post("/v1/ask")
    def ask(request: ResearchRequest):
        service = get_service()

        try:
            result = service.search(
                request.query,
                ticker=request.ticker,
                form_type=request.form_type,
                filing_date=(
                    request.filing_date.isoformat()
                    if request.filing_date
                    else None
                ),
                method=request.method,
                top_k=request.top_k,
            )

            return result.to_dict()

        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc

        except Exception as exc:
            logger.exception("Grounded answer request failed.")

            raise HTTPException(
                status_code=500,
                detail="An internal research error occurred.",
            ) from exc

    return application


app = create_app()
