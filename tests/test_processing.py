import os
from pathlib import Path

import pytest

from conftest import FakeFastTokenizer
from finance_rag.processing.chunking import (
    DocumentChunker,
    make_document_id,
    normalize_filing_text,
    segment_sentence_like_spans,
)
from finance_rag.processing.html_text import extract_filing_text

ROOT = Path(__file__).resolve().parents[1]


def _long_text(paragraphs=12, sentences=9):
    out = []
    for p in range(paragraphs):
        sents = [
            f"Paragraph {p} sentence {s} discusses revenue, margins and risk factors "
            f"for the fiscal year in considerable and repetitive detail."
            for s in range(sentences)
        ]
        out.append(" ".join(sents))
    return "\n\n".join(out)


@pytest.fixture
def chunker():
    return DocumentChunker(FakeFastTokenizer(), max_chunk_tokens=120,
                           target_overlap_tokens=25, minimum_target_tokens=30)


def test_normalization_preserves_paragraphs():
    raw = "A  b\t\tc  \r\n\r\n\r\n\r\nNext   para\x00 "
    assert normalize_filing_text(raw) == "A b c\n\nNext para"


def test_spans_have_exact_offsets():
    text = normalize_filing_text(_long_text(2, 3))
    spans = segment_sentence_like_spans(text)
    assert len(spans) == 6
    for span in spans:
        assert text[span["start_char"]:span["end_char"]] == span["text"]


def test_chunks_respect_contract(chunker):
    text = normalize_filing_text(_long_text())
    chunks = chunker.chunk_document("acme_10k_000000000000000001", text)
    assert len(chunks) > 5
    assert all(c["token_count"] <= 120 for c in chunks)
    assert all(text[c["start_char"]:c["end_char"]] == c["text"] for c in chunks)
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert chunks[0]["chunk_id"] == "acme_10k_000000000000000001__chunk_0000"
    # consecutive chunks overlap (atomic-span overlap retained)
    assert any(b["start_char"] < a["end_char"] for a, b in zip(chunks, chunks[1:]))
    # whole document is covered
    assert chunks[0]["start_char"] == 0 and chunks[-1]["end_char"] == len(text)


def test_oversized_span_is_split(chunker):
    text = " ".join(f"word{i}" for i in range(500))  # one span, no punctuation
    chunks = chunker.chunk_document("acme_10q_000000000000000002", text)
    assert len(chunks) >= 4
    assert all(c["token_count"] <= 120 for c in chunks)


def test_empty_document_yields_no_chunks(chunker):
    assert chunker.chunk_document("x_8k_000000000000000003", "") == []


def test_document_id_convention():
    assert make_document_id("AAPL", "10-K", "0000320193-23-000106") == "aapl_10k_000032019323000106"
    assert make_document_id("BRK.B", "8-K", "0001067983-26-000001") == "brkb_8k_000106798326000001"
    with pytest.raises(ValueError):
        make_document_id("AAPL", "10-K", "bad")


SEC_HTML = b"""<html><head><title>abc-20250930</title><style>p{}</style></head><body>
<div style="display:none"><ix:header><ix:hidden>dei:Secret 123</ix:hidden></ix:header></div>
<p>Item 1A. Risk Factors</p><p>Our supply chain is concentrated. We depend on TSMC.</p>
<table><tr><td>Revenue</td><td></td><td>$ 1,234</td></tr><tr><td>Net income</td><td>567</td></tr></table>
<script>var x = 1;</script><div>Tail&nbsp;text<br/>next line</div></body></html>"""


def test_html_extraction_keeps_structure_and_drops_hidden():
    text = normalize_filing_text(extract_filing_text(SEC_HTML))
    assert "Secret" not in text and "abc-20250930" not in text and "var x" not in text
    assert "Item 1A. Risk Factors\n\nOur supply chain is concentrated." in text
    assert "Revenue | $ 1,234" in text
    assert "Net income | 567" in text
    assert "Tail text\nnext line" in text


def test_html_extraction_handles_empty():
    assert extract_filing_text(b"   ") == ""


# ---------------------------------------------------------------------------
# Regression against the frozen v1 bundle (runs on a machine with BGE access).
# Rebuilds each of the 30 baseline documents from its chunks and checks the
# ported chunker reproduces all 4,174 chunks exactly.
# ---------------------------------------------------------------------------
V1_CHUNKS = ROOT / "artifacts" / "runtime" / "chunks_df.parquet"


@pytest.mark.skipif(
    not V1_CHUNKS.is_file() or os.getenv("FINANCE_RAG_RUN_BGE_TESTS") != "1",
    reason="set FINANCE_RAG_RUN_BGE_TESTS=1 with the v1 runtime bundle present",
)
def test_port_reproduces_v1_chunks_exactly():
    import pandas as pd
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    chunker = DocumentChunker(tokenizer)
    v1 = pd.read_parquet(V1_CHUNKS)

    for document_id, group in v1.groupby("document_id", sort=False):
        group = group.sort_values("chunk_index")
        length = int(group["end_char"].max())
        buffer = [" "] * length  # v1 text had single-space whitespace runs
        for row in group.itertuples(index=False):
            buffer[row.start_char:row.end_char] = list(row.text)
        text = "".join(buffer)

        rebuilt = chunker.chunk_document(document_id, text)
        assert len(rebuilt) == len(group), document_id
        for new, old in zip(rebuilt, group.itertuples(index=False)):
            assert new["chunk_id"] == old.chunk_id
            assert (new["start_char"], new["end_char"]) == (old.start_char, old.end_char)
            assert new["token_count"] == old.token_count
