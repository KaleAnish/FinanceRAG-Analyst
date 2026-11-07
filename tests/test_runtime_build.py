"""End-to-end: fake R2 -> fetch -> chunk -> embed -> finalize -> retriever/API."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from conftest import FakeFastTokenizer
from finance_rag.processing.chunking import DocumentChunker
from finance_rag.processing.runtime_build import (
    BuildConfig,
    chunk_all,
    embed_chunks,
    fetch_raw,
    finalize,
    load_chunks,
    read_ticker_file,
    select_filings,
)
from finance_rag.retrieval import FinanceRAGRetriever
from finance_rag.storage.local import LocalStorage


class FakeEmbedder:
    """Deterministic bag-of-words hashing into 384 dims, unit-normalized."""

    def encode(self, texts, **kwargs):
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                out[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % 384] += 1.0
        out[out.sum(axis=1) == 0, 0] = 1.0
        return out / np.linalg.norm(out, axis=1, keepdims=True)


COMPANIES = {
    "0000000001": ("AAA", "Alpha Corp", "semiconductor supply constraints and data center demand"),
    "0000000002": ("BBB", "Beta Bank", "credit losses, deposit outflows and net interest margin"),
    "0000000003": ("CCC", "Gamma Energy", "oil price volatility and refinery throughput"),
}


def _html(company, topic, form, n):
    paras = "".join(
        f"<p>{company} discusses {topic} in section {i}. "
        f"Management believes {topic} affected results for the period. "
        f"Additional commentary sentence {i} explains the {form} disclosure in detail.</p>"
        for i in range(n)
    )
    table = "<table><tr><td>Revenue</td><td>$ 1,000</td></tr><tr><td>Net income</td><td>$ 200</td></tr></table>"
    return (f"<html><head><title>x</title></head><body>"
            f"<div style='display:none'><ix:header>hidden</ix:header></div>"
            f"{paras}{table}</body></html>").encode()


@pytest.fixture
def corpus(tmp_path):
    storage = LocalStorage(tmp_path / "r2")
    manifest, inventory, securities = [], [], []
    seq = 0
    for cik, (ticker, name, topic) in COMPANIES.items():
        securities.append({"cik": cik, "ticker": ticker, "exchange": "NYSE", "share_class": ""})
        for form, n, date in [("10-K", 25, "2025-02-01"), ("10-Q", 12, "2025-05-01"),
                              ("10-Q", 12, "2025-08-01"), ("8-K", 2, "2025-09-01")]:
            seq += 1
            accession = f"{cik}-25-{seq:06d}"
            key = f"raw/sec/{int(cik)}/{accession.replace('-', '')}/doc{seq}.htm"
            payload = _html(name, topic, form, n)
            storage.put_bytes(key, payload)
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/doc{seq}.htm"
            manifest.append({
                "cik": cik, "ticker": ticker, "form_type": form, "accession_number": accession,
                "filing_date": date, "source_url": url, "object_key": key,
                "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload),
                "downloaded_at_utc": "", "status": "downloaded", "error_type": "", "error": "",
            })
            inventory.append({"cik": cik, "ticker": ticker, "company": name, "form_type": form,
                              "accession_number": accession, "filing_date": date,
                              "report_date": date, "primary_document": f"doc{seq}.htm",
                              "source_url": url})
    pd.DataFrame(manifest).to_csv(tmp_path / "manifest.csv.gz", index=False)
    pd.DataFrame(inventory).to_csv(tmp_path / "inventory.csv.gz", index=False)
    # BBB is also reachable through an alias, like GOOG/GOOGL.
    securities.append({"cik": "0000000002", "ticker": "BBB.A", "exchange": "NYSE", "share_class": "A"})
    pd.DataFrame(securities).to_csv(tmp_path / "securities.csv", index=False)
    return tmp_path, storage


def _config(tmp_path, tickers, **kw):
    return BuildConfig(
        version="runtime-test", tickers=tickers,
        manifest_path=tmp_path / "manifest.csv.gz", inventory_path=tmp_path / "inventory.csv.gz",
        securities_path=tmp_path / "securities.csv", out_dir=tmp_path / "builds",
        cache_dir=tmp_path / "cache", fetch_workers=2, chunk_workers=1,
        embed_checkpoint_rows=7, **kw,
    )


def _build(cfg, storage):
    chunker = DocumentChunker(FakeFastTokenizer(), max_chunk_tokens=120,
                              target_overlap_tokens=25, minimum_target_tokens=30)
    selected = select_filings(cfg)
    fetch_raw(cfg, selected, storage)
    chunk_all(cfg, selected, chunker=chunker)
    chunks = load_chunks(cfg, selected)
    embeddings = embed_chunks(cfg, chunks, model=FakeEmbedder())
    manifest = finalize(cfg, selected, chunks, embeddings)
    return selected, chunks, manifest


def test_end_to_end_build_and_retrieval(corpus, monkeypatch):
    tmp_path, storage = corpus
    cfg = _config(tmp_path, ["AAA", "BBB.A", "CCC"])
    selected, chunks, manifest = _build(cfg, storage)

    assert len(selected) == 12
    assert set(chunks["ticker"]) == {"AAA", "BBB.A", "CCC"}  # requested alias kept
    assert set(chunks["form_type"]) == {"10-K", "10-Q", "8-K"}
    assert "hidden" not in " ".join(chunks["text"])
    assert manifest["validation"]["self_retrieval"]
    assert manifest["inputs"]["documents_without_text"] == []

    runtime = cfg.runtime_dir
    meta = json.loads((runtime / "runtime_metadata.json").read_text())
    assert meta["corpus_rows"] == len(chunks) and meta["runtime_version"] == "runtime-test"
    for name, info in manifest["files"].items():
        assert hashlib.sha256((runtime / name).read_bytes()).hexdigest() == info["sha256"]

    # The production retriever loads the bundle unchanged, using the prebuilt BM25.
    project = tmp_path / "project"
    (project / "config").mkdir(parents=True)
    (project / "config" / "retrieval_config.json").write_text(json.dumps(
        {"dense_candidate_k": 30, "bm25_candidate_k": 20, "rrf_k": 30, "final_top_k": 10}))
    retriever = FinanceRAGRetriever(project, runtime_dir=runtime, embedding_model=FakeEmbedder())
    assert retriever.bm25_source == "prebuilt"
    assert retriever.available_tickers == ["AAA", "BBB.A", "CCC"]

    hits = retriever.search("credit losses deposit outflows", method="hybrid_rrf", top_k=3)
    assert hits["ticker"].iloc[0] == "BBB.A"

    compare = retriever.search("results for the period", ticker=["AAA", "CCC"], top_k=4)
    assert compare["ticker"].tolist() == ["AAA", "CCC", "AAA", "CCC"]

    dated = retriever.search("results", form_type="10-Q", start_date="2025-06-01", top_k=10)
    assert set(dated["filing_date"]) == {"2025-08-01"}

    with pytest.raises(ValueError):
        retriever.search("x", ticker="ZZZ")

    # And the API serves it.
    from finance_rag import api as api_module
    from finance_rag.app_service import FinanceRAGService

    def factory(project_dir):
        return FinanceRAGService(project_dir=project, retriever=retriever)

    with TestClient(api_module.create_app(service_factory=factory)) as client:
        assert client.get("/ready").status_code == 200
        corpus_info = client.get("/v1/corpus").json()
        assert corpus_info["runtime_version"] == "runtime-test"
        assert corpus_info["documents"] == 12
        body = client.post("/v1/search", json={
            "query": "oil price volatility", "tickers": ["AAA", "CCC"], "form_type": "10-K",
        }).json()
        assert {s["ticker"] for s in body["sources"]} == {"AAA", "CCC"}


def test_build_is_resumable_and_deterministic(corpus):
    tmp_path, storage = corpus
    cfg = _config(tmp_path, ["AAA", "CCC"])
    _, first_chunks, first = _build(cfg, storage)

    # Simulate a crash mid-embedding: roll progress back, rebuild, compare.
    progress = cfg.work_dir / "embeddings_progress.json"
    state = json.loads(progress.read_text())
    state["done"] = 7
    progress.write_text(json.dumps(state))
    _, second_chunks, second = _build(cfg, storage)

    assert first_chunks["chunk_id"].tolist() == second_chunks["chunk_id"].tolist()
    assert first["files"] == second["files"]


def test_fetch_rejects_corrupted_object(corpus):
    tmp_path, storage = corpus
    cfg = _config(tmp_path, ["AAA"])
    selected = select_filings(cfg)
    storage.put_bytes(selected.iloc[0]["object_key"], b"<html>tampered</html>")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        fetch_raw(cfg, selected, storage)


def test_selection_guards(corpus):
    tmp_path, _ = corpus
    with pytest.raises(ValueError, match="not found"):
        select_filings(_config(tmp_path, ["NOPE"]))
    with pytest.raises(ValueError, match="same issuer"):
        select_filings(_config(tmp_path, ["BBB", "BBB.A"]))
    sample = select_filings(_config(tmp_path, ["AAA"], limit_per_company=1))
    assert sorted(sample["form_type"]) == ["10-K", "10-Q", "8-K"]
    assert sample.loc[sample["form_type"] == "10-Q", "filing_date"].item() == "2025-08-01"
    only_k = select_filings(_config(tmp_path, ["AAA"], forms=("10-K",)))
    assert only_k["form_type"].unique().tolist() == ["10-K"]


def test_mvp50_list_resolves_against_frozen_corpus():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    tickers = read_ticker_file(root / "config" / "universe" / "mvp50.txt")
    assert len(tickers) == 50
    securities = root / "data" / "universe" / "securities.csv"
    if not securities.is_file():
        pytest.skip("data/universe/securities.csv not present")
    cfg = BuildConfig(
        version="probe", tickers=tickers,
        manifest_path=root / "artifacts/releases/raw-corpus-v1/download_manifest.csv.gz",
        inventory_path=root / "artifacts/releases/raw-corpus-v1/filing_inventory.csv.gz",
        securities_path=securities, out_dir=Path("/tmp/finance_rag_probe"),
    )
    selected = select_filings(cfg)
    assert selected["ticker"].nunique() == 50
    assert "GOOGL" in set(selected["ticker"])
