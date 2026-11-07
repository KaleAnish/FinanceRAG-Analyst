"""Canonical chunking — a faithful port of Notebook 02 (``bge_sentence_480_80_v1``).

The logic below mirrors the frozen research notebook cell by cell so that v2
runtime bundles use exactly the same chunking contract as the evaluated
baseline:

* conservative text normalization (Cell 13)
* sentence-like spans with exact character offsets (Cell 15)
* tokenizer-offset splitting of oversized spans (Cell 17)
* greedy packing to <= 480 BGE tokens with ~80-token atomic-span overlap,
  skipping overlap-only chunks and repairing an undersized final chunk (Cell 19)

The tokenizer is injected so tests can run without downloading BGE.
It must behave like a Hugging Face fast tokenizer: callable with
``add_special_tokens``, ``truncation`` and ``return_offsets_mapping`` and
exposing ``num_special_tokens_to_add(pair=False)``.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

PROCESSING_VERSION = "conservative_text_v1"
CHUNKING_VERSION = "bge_sentence_480_80_v1"
TOKENIZER_NAME = "BAAI/bge-small-en-v1.5"

MAX_CHUNK_TOKENS = 480
TARGET_OVERLAP_TOKENS = 80
MINIMUM_TARGET_TOKENS = 100

UNICODE_SPACE_PATTERN = re.compile(r"[   -   　]")
HORIZONTAL_WHITESPACE_PATTERN = re.compile(r"[ \t\f\v]+")
EXCESS_BLANK_LINES_PATTERN = re.compile(r"\n{3,}")

SENTENCE_BOUNDARY_PATTERN = re.compile(
    r"""
    (?:
        (?<=[.!?])          # sentence-ending punctuation
        (?=\s+[A-Z0-9(])    # likely beginning of the next sentence
    )
    |
    (?:
        \n{2,}              # paragraph boundary
    )
    """,
    flags=re.VERBOSE,
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_filing_text(raw_text: str) -> str:
    """Notebook 02 Cell 13: conservative, structure-preserving normalization."""
    text = unicodedata.normalize("NFC", raw_text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    text = UNICODE_SPACE_PATTERN.sub(" ", text)
    text = HORIZONTAL_WHITESPACE_PATTERN.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = EXCESS_BLANK_LINES_PATTERN.sub("\n\n", text)
    return text.strip()


def segment_sentence_like_spans(text: str) -> list[dict[str, Any]]:
    """Notebook 02 Cell 15: deterministic spans with exact character offsets."""
    if not text:
        return []

    boundary_positions = {m.start() for m in SENTENCE_BOUNDARY_PATTERN.finditer(text)}
    boundary_positions.add(len(text))

    spans = []
    start = 0
    for boundary in sorted(boundary_positions):
        if boundary <= start:
            continue
        raw_span = text[start:boundary]
        leading_whitespace = len(raw_span) - len(raw_span.lstrip())
        trailing_boundary = boundary - (len(raw_span) - len(raw_span.rstrip()))
        span_start = start + leading_whitespace
        span_end = trailing_boundary
        if span_start < span_end:
            spans.append({
                "span_index": len(spans),
                "start_char": span_start,
                "end_char": span_end,
                "text": text[span_start:span_end],
            })
        start = boundary
    return spans


class DocumentChunker:
    """Stateful wrapper so the tokenizer is bound once per process."""

    def __init__(
        self,
        tokenizer,
        *,
        max_chunk_tokens: int = MAX_CHUNK_TOKENS,
        target_overlap_tokens: int = TARGET_OVERLAP_TOKENS,
        minimum_target_tokens: int = MINIMUM_TARGET_TOKENS,
    ):
        self.tokenizer = tokenizer
        self.max_chunk_tokens = max_chunk_tokens
        self.target_overlap_tokens = target_overlap_tokens
        self.minimum_target_tokens = minimum_target_tokens
        self.special_token_count = tokenizer.num_special_tokens_to_add(pair=False)
        self.max_content_tokens = max_chunk_tokens - self.special_token_count
        if not minimum_target_tokens < max_chunk_tokens:
            raise ValueError("minimum_target_tokens must be below max_chunk_tokens")
        if not target_overlap_tokens < max_chunk_tokens:
            raise ValueError("target_overlap_tokens must be below max_chunk_tokens")

    # Cell 17 ---------------------------------------------------------------
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])

    def split_oversized_span(self, span: dict[str, Any]) -> list[dict[str, Any]]:
        span_text = span["text"]
        encoding = self.tokenizer(
            span_text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )
        token_offsets = encoding["offset_mapping"]

        if len(token_offsets) <= self.max_content_tokens:
            return [{
                **span,
                "source_span_index": span["span_index"],
                "split_index": 0,
                "was_split": False,
                "token_count": len(token_offsets) + self.special_token_count,
            }]

        pieces = []
        for split_index, token_start in enumerate(
            range(0, len(token_offsets), self.max_content_tokens)
        ):
            token_end = min(token_start + self.max_content_tokens, len(token_offsets))
            local_start = 0 if token_start == 0 else token_offsets[token_start][0]
            local_end = (
                len(span_text) if token_end == len(token_offsets)
                else token_offsets[token_end][0]
            )
            piece_text = span_text[local_start:local_end]
            piece = {
                "span_index": None,
                "source_span_index": span["span_index"],
                "split_index": split_index,
                "was_split": True,
                "start_char": span["start_char"] + local_start,
                "end_char": span["start_char"] + local_end,
                "text": piece_text,
                "token_count": self.count_tokens(piece_text),
            }
            if piece["token_count"] > self.max_chunk_tokens:
                raise AssertionError("Split piece exceeds the token ceiling.")
            pieces.append(piece)
        return pieces

    # Cell 18 ---------------------------------------------------------------
    def atomic_spans(self, processed_text: str) -> list[dict[str, Any]]:
        spans = []
        for sentence_span in segment_sentence_like_spans(processed_text):
            spans.extend(self.split_oversized_span(sentence_span))
        for atomic_index, span in enumerate(spans):
            span["span_index"] = atomic_index
            if processed_text[span["start_char"]:span["end_char"]] != span["text"]:
                raise AssertionError("Atomic span offsets do not match text.")
            if span["token_count"] > self.max_chunk_tokens:
                raise AssertionError("Atomic span exceeds the token ceiling.")
        return spans

    # Cell 19 ---------------------------------------------------------------
    def _make_chunk(self, document_id, chunk_index, processed_text, atomic_spans,
                    start_span_index, end_span_index):
        start_char = atomic_spans[start_span_index]["start_char"]
        end_char = atomic_spans[end_span_index]["end_char"]
        chunk_text = processed_text[start_char:end_char]
        token_count = self.count_tokens(chunk_text)
        if not chunk_text or token_count > self.max_chunk_tokens:
            raise AssertionError("Invalid chunk produced.")
        return {
            "document_id": document_id,
            "chunk_index": chunk_index,
            "start_span_index": start_span_index,
            "end_span_index": end_span_index,
            "start_char": start_char,
            "end_char": end_char,
            "text": chunk_text,
            "token_count": token_count,
        }

    def build_document_chunks(self, document_id: str, processed_text: str,
                              atomic_spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not atomic_spans:
            return []

        chunks: list[dict[str, Any]] = []
        start_span_index = 0

        while start_span_index < len(atomic_spans):
            accepted_end_span_index = None
            for candidate_end in range(start_span_index, len(atomic_spans)):
                candidate_text = processed_text[
                    atomic_spans[start_span_index]["start_char"]:
                    atomic_spans[candidate_end]["end_char"]
                ]
                if self.count_tokens(candidate_text) > self.max_chunk_tokens:
                    break
                accepted_end_span_index = candidate_end

            if accepted_end_span_index is None:
                raise AssertionError("An atomic span does not fit on its own.")

            # Overlap that adds no new span would be a redundant chunk.
            if chunks and accepted_end_span_index <= chunks[-1]["end_span_index"]:
                start_span_index = chunks[-1]["end_span_index"] + 1
                continue

            chunks.append(self._make_chunk(
                document_id, len(chunks), processed_text, atomic_spans,
                start_span_index, accepted_end_span_index,
            ))

            if accepted_end_span_index == len(atomic_spans) - 1:
                break

            if accepted_end_span_index == start_span_index:
                start_span_index += 1
                continue

            next_start = accepted_end_span_index
            while next_start > start_span_index + 1:
                overlap_text = processed_text[
                    atomic_spans[next_start]["start_char"]:
                    atomic_spans[accepted_end_span_index]["end_char"]
                ]
                if self.count_tokens(overlap_text) >= self.target_overlap_tokens:
                    break
                next_start -= 1

            if not next_start > start_span_index:
                raise AssertionError("Chunking failed to make forward progress.")
            start_span_index = next_start

        # Repair an undersized final residual when backward expansion is possible.
        if len(chunks) >= 2 and chunks[-1]["token_count"] < self.minimum_target_tokens:
            previous_chunk, final_chunk = chunks[-2], chunks[-1]
            repaired_start = final_chunk["start_span_index"]
            while repaired_start > previous_chunk["start_span_index"]:
                candidate_start = repaired_start - 1
                candidate_text = processed_text[
                    atomic_spans[candidate_start]["start_char"]:
                    atomic_spans[final_chunk["end_span_index"]]["end_char"]
                ]
                candidate_tokens = self.count_tokens(candidate_text)
                if candidate_tokens > self.max_chunk_tokens:
                    break
                repaired_start = candidate_start
                if candidate_tokens >= self.minimum_target_tokens:
                    break
            chunks[-1] = self._make_chunk(
                document_id, len(chunks) - 1, processed_text, atomic_spans,
                repaired_start, final_chunk["end_span_index"],
            )

        for chunk_index, chunk in enumerate(chunks):
            chunk["chunk_index"] = chunk_index
        return chunks

    def chunk_document(self, document_id: str, processed_text: str) -> list[dict[str, Any]]:
        """Spans -> chunks, with the Cell 20 coverage and offset invariants enforced."""
        spans = self.atomic_spans(processed_text)
        chunks = self.build_document_chunks(document_id, processed_text, spans)

        for previous, current in zip(chunks, chunks[1:]):
            if current["end_span_index"] <= previous["end_span_index"]:
                raise AssertionError("A chunk failed to extend coverage.")

        covered = {
            i for chunk in chunks
            for i in range(chunk["start_span_index"], chunk["end_span_index"] + 1)
        }
        if covered != set(range(len(spans))):
            raise AssertionError("Chunks do not cover every atomic span.")

        for chunk in chunks:
            if processed_text[chunk["start_char"]:chunk["end_char"]] != chunk["text"]:
                raise AssertionError("Chunk text is not an exact substring.")
            chunk["chunk_id"] = f"{document_id}__chunk_{chunk['chunk_index']:04d}"
            chunk["below_minimum_target"] = chunk["token_count"] < self.minimum_target_tokens
        return chunks


def make_document_id(ticker: str, form_type: str, accession_number: str) -> str:
    """Matches the baseline convention, e.g. ``aapl_10k_000032019323000106``."""
    form = re.sub(r"[^a-z0-9]", "", str(form_type).lower())
    accession = re.sub(r"\D", "", str(accession_number))
    tick = re.sub(r"[^a-z0-9]", "", str(ticker).lower())
    if not tick or not form or len(accession) != 18:
        raise ValueError(f"Cannot build document_id from {ticker!r}, {form_type!r}, {accession_number!r}")
    return f"{tick}_{form}_{accession}"
