from dataclasses import dataclass, asdict
from datetime import datetime
import re
from typing import Any

MAX_QUERY_LENGTH = 1200
TICKER_PATTERN = re.compile(r'^[A-Z0-9][A-Z0-9.-]{0,9}$')

from .retrieval import FinanceRAGRetriever
from .generation import GenerationConfig, generate_answer


def _plain_url(value):
    if not isinstance(value, str):
        return value
    value = value.strip()
    match = re.fullmatch(r"\[([^\]]+)\]\((https?://[^)]+)\)", value)
    return match.group(2) if match else value


@dataclass
class ResearchResponse:
    query: str
    answer: str
    abstained: bool
    abstention_reason: str | None
    sources: list[dict[str, Any]]
    citation_validation: dict[str, Any]
    provider: str
    model: str
    policy_version: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FinanceRAGService:
    """Application-facing service for SEC filing research."""

    def __init__(
        self,
        project_dir,
        generation_config: GenerationConfig | None = None,
        retriever=None,
    ):
        self.project_dir = project_dir
        self.config = generation_config or GenerationConfig.from_env()
        self.retriever = retriever or FinanceRAGRetriever(
            project_dir=project_dir
        )

    def search(
        self,
        query: str,
        *,
        ticker: str | None = None,
        form_type: str = "10-K",
        filing_date: str | None = None,
        method: str = "hybrid_rrf",
        top_k: int = 5,
    ) -> ResearchResponse:
        if not isinstance(query, str):
            raise ValueError("Enter a research question.")
        query = query.strip()
        if not query:
            raise ValueError("Enter a research question.")
        if len(query) > MAX_QUERY_LENGTH:
            raise ValueError(f"Keep the question under {MAX_QUERY_LENGTH + 1} characters.")
        if method not in {"hybrid_rrf", "dense", "bm25"}:
            raise ValueError("Unsupported retrieval method.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 10:
            raise ValueError("Evidence chunks must be an integer between 1 and 10.")
        if form_type not in {"10-K"}:
            raise ValueError("Unsupported filing type.")

        clean_ticker = None
        if ticker is not None and str(ticker).strip():
            clean_ticker = str(ticker).strip().upper()
            if not TICKER_PATTERN.fullmatch(clean_ticker):
                raise ValueError("Enter a valid ticker using letters, numbers, dots, or hyphens.")

        clean_filing_date = None
        if filing_date is not None and str(filing_date).strip():
            clean_filing_date = str(filing_date).strip()
            try:
                datetime.strptime(clean_filing_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError("Enter the filing date as YYYY-MM-DD.") from None

        evidence = self.retriever.search(
            query,
            method=method,
            ticker=clean_ticker,
            form_type=form_type,
            filing_date=clean_filing_date,
            top_k=top_k,
        )

        if evidence is None or len(evidence) == 0:
            return ResearchResponse(
                query=query,
                answer="",
                abstained=True,
                abstention_reason="No matching filing evidence was found.",
                sources=[],
                citation_validation={},
                provider=self.config.provider,
                model=self.config.model,
                policy_version=self.config.policy_version,
            )

        result = generate_answer(query, evidence, config=self.config)

        # Keep source metadata for the UI. Do not expose full chunk text by default.
        source_columns = [
            "chunk_id", "ticker", "company", "form_type",
            "filing_date", "document_id", "source_url", "rank",
        ]
        sources = []
        for _, row in evidence.iterrows():
            source = {
                key: (None if key not in evidence.columns else row[key])
                for key in source_columns
            }
            normalised = {}
            for key, value in source.items():
                if value is not None and hasattr(value, "item"):
                    try:
                        value = value.item()
                    except (ValueError, TypeError):
                        pass
                if key == "source_url":
                    value = _plain_url(value)
                normalised[key] = value
            sources.append(normalised)

        return ResearchResponse(
            query=query,
            answer=result.get("answer", ""),
            abstained=result.get("abstained", True),
            abstention_reason=result.get("abstention_reason"),
            sources=sources,
            citation_validation=result.get("citation_validation", {}),
            provider=result.get("provider", self.config.provider),
            model=result.get("model", self.config.model),
            policy_version=result.get(
                "policy_version", self.config.policy_version
            ),
        )
