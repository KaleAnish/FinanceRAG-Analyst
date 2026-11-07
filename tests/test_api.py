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
        {"query": "Valid query", "form_type": "8-K"},
        {"query": "Valid query", "filing_date": "not-a-date"},
        {"query": "Valid query", "unexpected_field": True},
        {"query": "   "},
    ],
)
def test_invalid_requests_return_422(client, payload):
    response = client.post("/v1/search", json=payload)

    assert response.status_code == 422
