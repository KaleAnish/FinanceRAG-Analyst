from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from finance_rag.api import create_app


class MockRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))

        return pd.DataFrame([
            {
                "chunk_id": "AAPL-10K-001",
                "ticker": "AAPL",
                "company": "Apple Inc.",
                "form_type": "10-K",
                "filing_date": "2025-10-31",
                "document_id": "AAPL-2025",
                "source_url": "https://example.com/filing",
                "rank": 1,
                "retrieval_score": 0.85,
                "text": "This full chunk must not be returned.",
            }
        ])


class MockService:
    def __init__(self, project_dir):
        self.retriever = MockRetriever()
        self.ask_calls = []

    def search(self, query, **kwargs):
        self.ask_calls.append((query, kwargs))

        return SimpleNamespace(
            to_dict=lambda: {
                "query": query,
                "answer": "Apple disclosed several business risks [S1].",
                "abstained": False,
                "abstention_reason": None,
                "sources": [{"chunk_id": "AAPL-10K-001"}],
                "citation_validation": {"valid": True},
                "provider": "mock",
                "model": "mock-model",
                "policy_version": "test",
                "error": None,
            }
        )


@pytest.fixture
def client():
    app = create_app(service_factory=MockService)

    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_ready(client):
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_search_returns_retrieval_results_without_chunk_text(client):
    response = client.post(
        "/v1/search",
        json={
            "query": "What risks did Apple disclose?",
            "ticker": "aapl",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["count"] == 1
    assert body["sources"][0]["ticker"] == "AAPL"
    assert "text" not in body["sources"][0]


def test_ask_returns_grounded_answer(client):
    response = client.post(
        "/v1/ask",
        json={"query": "What risks did Apple disclose?"},
    )

    assert response.status_code == 200
    assert response.json()["abstained"] is False
    assert response.json()["provider"] == "mock"


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "A" * 1201},
        {"query": "Valid query", "method": "unsupported"},
        {"query": "Valid query", "top_k": 0},
        {"query": "Valid query", "top_k": 11},
        {"query": "Valid query", "ticker": "INVALID TICKER"},
        {"query": "Valid query", "form_type": "S-1"},
        {"query": "Valid query", "tickers": []},
        {"query": "Valid query", "tickers": ["A", "B", "C", "D", "E", "F"]},
        {"query": "Valid query", "tickers": ["BAD TICKER"]},
        {"query": "Valid query", "ticker": "AAPL", "tickers": ["MSFT"]},
        {"query": "Valid query", "start_date": "2026-01-01", "end_date": "2025-01-01"},
        {"query": "Valid query", "filing_date": "not-a-date"},
        {"query": "Valid query", "unexpected_field": True},
        {"query": "   "},
    ],
)
def test_invalid_requests_return_422(client, payload):
    response = client.post("/v1/search", json=payload)

    assert response.status_code == 422


def test_search_forwards_multi_ticker_and_date_range(client):
    response = client.post(
        "/v1/search",
        json={
            "query": "Compare data center growth",
            "tickers": ["nvda", "amd", "NVDA"],
            "form_type": "10-Q",
            "start_date": "2025-01-01",
            "end_date": "2026-06-30",
        },
    )
    assert response.status_code == 200
    _, kwargs = client.app.state.service.retriever.calls[-1]
    assert kwargs["ticker"] == ["NVDA", "AMD"]
    assert kwargs["form_type"] == "10-Q"
    assert kwargs["start_date"] == "2025-01-01"
    assert kwargs["end_date"] == "2026-06-30"
    assert kwargs["filing_date"] is None


def test_all_form_types_accepted(client):
    for form in ("10-K", "10-Q", "8-K"):
        assert client.post("/v1/search", json={"query": "q", "form_type": form}).status_code == 200


def test_form_type_defaults_to_all_forms(client):
    client.post("/v1/ask", json={"query": "q"})
    _, kwargs = client.app.state.service.ask_calls[-1]
    assert kwargs["form_type"] is None


def test_ready_reports_startup_error_type():
    def broken_factory(project_dir):
        raise FileNotFoundError("/secret/path/chunks_df.parquet")

    with TestClient(create_app(service_factory=broken_factory)) as broken:
        response = broken.get("/ready")
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["startup_error"] == "FileNotFoundError"
    assert "secret" not in str(detail)


def test_corpus_endpoint(client):
    retriever = client.app.state.service.retriever
    retriever.available_tickers = ["AAPL", "NVDA"]
    retriever.available_form_types = ["10-K", "10-Q"]
    retriever.chunks_df = pd.DataFrame({
        "document_id": ["a", "a", "b"],
        "filing_date": ["2024-01-01", "2024-01-01", "2026-05-01"],
    })
    retriever.metadata = {"runtime_version": "runtime-v2-mvp50"}
    body = client.get("/v1/corpus").json()
    assert body["tickers"] == ["AAPL", "NVDA"]
    assert body["documents"] == 2 and body["chunks"] == 3
    assert body["filing_date_max"] == "2026-05-01"
    assert body["runtime_version"] == "runtime-v2-mvp50"
