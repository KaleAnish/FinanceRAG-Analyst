import logging
import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.finance_rag.app_service import FinanceRAGService, MAX_QUERY_LENGTH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('finance_rag.app')
MAX_SESSION_QUERIES = 10

if "research_queries_used" not in st.session_state:
    st.session_state.research_queries_used = 0
if "research_result" not in st.session_state:
    st.session_state.research_result = None

# Load the provider key from Streamlit secrets without displaying it.
try:
    key = st.secrets.get('GROQ_API_KEY')
    if key and not os.environ.get('GROQ_API_KEY'):
        os.environ['GROQ_API_KEY'] = str(key)
except Exception:
    pass


def ensure_runtime_available(project_dir):
    """Download and verify the pinned runtime bundle when runtime files are missing."""
    import hashlib
    import io
    import urllib.request
    import zipfile

    project_dir = Path(project_dir).resolve()
    expected_bundle_sha256 = (
        "cf14f78ed30cbb285e613803fe15a3c442934394386ed4757febcb22f081cedf"
    )
    bundle_url = (
        "https://github.com/KaleAnish/FinanceRAG-Analyst/"
        "releases/download/runtime-v1/finance_rag_runtime_v1.zip"
    )
    expected_files = {
        "artifacts/runtime/chunks_df.parquet":
            "c41480fd37f6714c27f1c1a4a69e90634ec841cde76523d20d53559266864dd5",
        "artifacts/runtime/chunk_embeddings.npy":
            "255b686b41dd35ae675a31ede84d0355c49c5f9800594ab6221828ef0fdea6c5",
        "artifacts/runtime/runtime_metadata.json":
            "47032782d9569334d454b61ef02c77acda540a9f7410ab4202fce49abe2209cd",
        "config/retrieval_config.json":
            "ce60e31de71cf27546c249a56fbbb074078ab5fe0b292c6bc720ebe6092e13c7",
    }

    def files_are_valid():
        for name, expected_hash in expected_files.items():
            path = project_dir / name
            if not path.is_file():
                return False
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected_hash:
                return False
        return True

    if files_are_valid():
        return

    request = urllib.request.Request(
        bundle_url, headers={"User-Agent": "FinRAG-Analyst/1.0"}
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        bundle = response.read(25 * 1024 * 1024 + 1)

    if len(bundle) > 25 * 1024 * 1024:
        raise RuntimeError("Runtime bundle exceeds the permitted size.")
    if hashlib.sha256(bundle).hexdigest() != expected_bundle_sha256:
        raise RuntimeError("Runtime bundle checksum mismatch; extraction aborted.")

    # Validate the complete archive before writing anything.
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        members = {
            item.filename: item
            for item in archive.infolist()
            if not item.is_dir()
        }
        if set(members) != set(expected_files):
            raise RuntimeError("Unexpected runtime archive contents.")

        payloads = {}
        for name, item in members.items():
            if item.flag_bits & 1 or item.file_size > 20 * 1024 * 1024:
                raise RuntimeError("Unsafe runtime archive member rejected.")
            payload = archive.read(item)
            if len(payload) != item.file_size:
                raise RuntimeError(f"Size check failed for {name}.")
            if hashlib.sha256(payload).hexdigest() != expected_files[name]:
                raise RuntimeError(f"Artifact checksum mismatch: {name}.")
            payloads[name] = payload

    # Write only validated, allowlisted files.
    for name, payload in payloads.items():
        target = (project_dir / name).resolve()
        if not target.is_relative_to(project_dir):
            raise RuntimeError("Unsafe runtime path rejected.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)

    if not files_are_valid():
        raise RuntimeError("Installed runtime verification failed.")


@st.cache_resource(show_spinner='Loading the research engine...')
@st.cache_resource(show_spinner='Loading the research engine...')
def get_service(project_dir):
    ensure_runtime_available(Path(project_dir))
    return FinanceRAGService(project_dir=Path(project_dir))


st.set_page_config(
    page_title="FinanceRAG Analyst",
    page_icon="📊",
    layout="wide",
)

st.title("FinanceRAG Analyst")
st.caption(
    "Research company disclosures using SEC filing evidence. "
    "Answers may be incomplete; verify important conclusions against the filings."
)

with st.sidebar:
    st.header("Research filters")
    ticker_input = st.text_input(
        "Ticker (optional)",
        placeholder="e.g. AAPL",
        help="Leave blank to search across available companies.",
    )
    filing_date = st.text_input(
        "Filing date (optional)",
        placeholder="YYYY-MM-DD",
    )
    form_type = st.selectbox("Filing type", ["10-K"])
    retrieval_method = st.selectbox(
        "Retrieval method",
        ["hybrid_rrf", "dense", "bm25"],
        format_func=lambda x: {
            "hybrid_rrf": "Hybrid (recommended)",
            "dense": "Dense semantic",
            "bm25": "BM25 keyword",
        }[x],
    )
    top_k = st.slider("Evidence chunks", min_value=3, max_value=10, value=5)
    budget_status = st.empty()


    st.divider()
    st.markdown("**Built by Anish Kale**")
    st.caption("Data Scientist")
    st.caption("Open to conversations, collaborations & opportunities in tech and fintech")
    st.caption("Built in SoCal")
    st.markdown(
        "[LinkedIn](https://www.linkedin.com/in/kaleanish) · "
        "[Portfolio](https://www.anishkale.dev)"
    )

query = st.text_area(
    "What would you like to research?",
    placeholder="e.g. What are Apple's principal business risks?",
    height=100,
    max_chars=MAX_QUERY_LENGTH,
    help=f"Maximum {MAX_QUERY_LENGTH} characters.",
)

if st.button("Research filings", type="primary", use_container_width=True, disabled=st.session_state.research_queries_used >= MAX_SESSION_QUERIES):
    if not query.strip():
        st.warning("Enter a research question first.")
    elif filing_date.strip() and len(filing_date.strip()) != 10:
        st.warning("Enter the filing date as YYYY-MM-DD, or leave it blank.")
    else:
        st.session_state.research_queries_used += 1
        st.session_state.research_result = None
        try:
            with st.spinner("Searching filings and generating an evidence-grounded answer..."):
                if "finance_rag_service" not in st.session_state:
                    ensure_runtime_available(PROJECT_DIR)
                    st.session_state.finance_rag_service = FinanceRAGService(
                        project_dir=PROJECT_DIR
                    )

                st.session_state.research_result = (
                    st.session_state.finance_rag_service.search(
                        query,
                        ticker=ticker_input.strip() or None,
                        form_type=form_type,
                        filing_date=filing_date.strip() or None,
                        method=retrieval_method,
                        top_k=top_k,
                    ).to_dict()
                )
        except Exception as exc:
            logger.exception("Research request failed.")
            category = getattr(exc, "category", None)

            if category == "daily":
                st.warning(
                    "The research service has reached its current usage quota. "
                    "Please try again later or check the provider's usage limits."
                )
            elif category == "temporary":
                st.warning(
                    "The research service is temporarily busy. "
                    "Please wait a minute and try again."
                )
            else:
                st.error(
                    "We couldn't complete this research request. "
                    "Please try again shortly."
                )

# Update the sidebar after counting the current attempt.
budget_status.caption(
    f"Session query budget: "
    f"{st.session_state.research_queries_used}/{MAX_SESSION_QUERIES} used."
)

result = st.session_state.research_result

if result:
    st.divider()
    st.subheader("Answer")

    if result.get("abstained"):
        st.warning(
            result.get("abstention_reason")
            or "The available evidence was insufficient to answer."
        )
    elif result.get("answer"):
        st.markdown(result["answer"])
    else:
        st.info("No answer was returned.")

    validation = result.get("citation_validation") or {}
    checks = validation.get("validation_checks", {})
    citations_valid = validation.get(
        "citations_valid",
        checks.get("all_citations_in_context", False),
    )

    if citations_valid:
        st.success("Citation labels passed context validation.")
    else:
        st.warning(
            "Citation validation did not confirm all labels. "
            "Check the source passages before relying on the answer."
        )

    st.subheader("Retrieved filing evidence")
    sources = result.get("sources", [])
    if not sources:
        st.info("No source metadata was returned.")
    else:
        for i, source in enumerate(sources, start=1):
            title = (
                f"[S{i}] {source.get('company') or source.get('ticker') or 'Company'}"
                f" — {source.get('form_type') or 'SEC filing'}"
                f" — {source.get('filing_date') or 'Date unavailable'}"
            )
            with st.expander(title):
                st.write("**Ticker:**", source.get("ticker") or "Not available")
                st.write("**Chunk ID:**", source.get("chunk_id") or "Not available")
                st.write("**Document ID:**", source.get("document_id") or "Not available")
                source_url = source.get("source_url")
                if source_url:
                    # Only render valid HTTP(S) URLs as clickable links.
                    parsed = urlparse(source_url)
                    if parsed.scheme in {"http", "https"} and parsed.netloc:
                        st.markdown(f"[Open SEC filing]({source_url})")
                    else:
                        st.text(source_url)

    st.caption(
        "Citation-label validation does not establish that every factual claim "
        "is fully supported by its cited passage. Verify consequential findings "
        "against the original SEC filing."
    )
