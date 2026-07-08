# FinanceRAG-Analyst

> Retrieval-Augmented Financial Intelligence over SEC EDGAR filings using hybrid retrieval (FAISS + BM25 + RRF), metadata-aware search, and citation-grounded LLM reasoning.

---

## Overview

FinanceRAG-Analyst is a Retrieval-Augmented Generation (RAG) system designed to answer financial research questions directly from SEC EDGAR filings.

Instead of relying solely on a large language model's internal knowledge, the system retrieves relevant sections from SEC filings using a hybrid dense+sparse retrieval pipeline before generating grounded, citation-backed responses.

The project focuses on reducing hallucinations while improving explainability through evidence-based retrieval and source citations.

---

## Motivation

Financial filings contain high-value information about:

- Business strategy
- Risk factors
- Capital expenditures
- AI initiatives
- Geographic exposure
- Supply-chain dependencies
- Revenue drivers
- Regulatory risks

However, these filings are often hundreds of pages long.

FinanceRAG-Analyst enables analysts, students, and researchers to query these filings using natural language while remaining grounded in the original SEC documents.

---

# System Architecture

```
SEC EDGAR Filings
        │
        ▼
Data Collection
        │
        ▼
Cleaning & Parsing
        │
        ▼
Semantic Chunking
        │
        ▼
Metadata Extraction
        │
        ▼
Embedding Generation
(BGE Small v1.5)
        │
        ├──────────────┐
        ▼              ▼
     FAISS          BM25
(Dense Search)  (Sparse Search)
        │              │
        └──────┬───────┘
               ▼
    Reciprocal Rank Fusion
               │
               ▼
Metadata Filtering
               │
               ▼
Evidence Guardrails
               │
               ▼
Groq / Gemini LLM
               │
               ▼
Citation-Grounded Answer
```

---

# Project Structure

```
FinanceRAG-Analyst/

├── notebooks/
│   ├── 01_sec_data_pipeline.ipynb
│   ├── 02_chunking_metadata.ipynb
│   ├── 03_hybrid_retrieval.ipynb
│   └── 04_rag_answering.ipynb
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── indexes/
│   ├── evaluation/
│   └── answers/
│
├── screenshots/
│
├── app/
│
├── requirements.txt
│
└── README.md
```

---

# Features

- SEC EDGAR filing ingestion
- Metadata-aware semantic chunking
- Dense retrieval using FAISS
- Sparse retrieval using BM25
- Hybrid retrieval using Reciprocal Rank Fusion (RRF)
- Automatic retrieval evaluation
- Company-aware filtering
- Citation-grounded answer generation
- Provider-flexible LLM support (Groq / Gemini)
- Basic hallucination and evidence guardrails

---

# Tech Stack

### Data Processing

- Python
- Pandas
- NumPy

### Retrieval

- FAISS
- BM25
- Sentence Transformers
- BAAI/bge-small-en-v1.5

### Large Language Models

- Groq API
- Google Gemini API

### Evaluation

- Reciprocal Rank Fusion (RRF)
- Retrieval benchmarking
- Citation validation

---

# Sample Questions

Example questions supported by the system:

- How does Nvidia discuss demand for AI infrastructure?
- What risks does Apple mention about China?
- Compare Microsoft's and Google's AI strategy.
- How does Amazon discuss AWS capital expenditures?
- What does Google say about advertising revenue?

---

# Repository Workflow

Notebook 1

- Download SEC EDGAR filings
- Parse documents
- Build unified dataset

Notebook 2

- Semantic chunking
- Metadata extraction
- Embedding generation

Notebook 3

- FAISS retrieval
- BM25 retrieval
- Hybrid RRF retrieval
- Retrieval evaluation
- Best configuration selection

Notebook 4

- Provider-flexible RAG pipeline
- Citation generation
- Company-aware retrieval
- Evidence guardrails
- Grounded answer generation

---

# Future Improvements

- Streamlit web application
- FastAPI backend
- Docker deployment
- Live SEC filing updates
- Multi-document reasoning
- Agentic financial workflows
- Advanced RAG evaluation
- Support for earnings call transcripts and financial reports beyond SEC filings

---

# Example Output

```text
Question:
What risks does Apple mention about China?

↓

Answer:
Apple notes several risks related to operations and supply-chain dependencies
associated with China, including manufacturing concentration, geopolitical
tensions, and regulatory uncertainty. [S1][S2]

↓

Sources

[S1] Apple 10-K (2024)

[S2] Apple 10-Q (2024)
```

---

# Why Hybrid Retrieval?

Dense retrieval captures semantic similarity.

Sparse retrieval captures exact keyword matches.

Reciprocal Rank Fusion combines both approaches to improve retrieval quality while maintaining robustness across diverse financial queries.

---

# Disclaimer

This project is intended for research and educational purposes.

Responses are generated only from retrieved SEC filing context and should not be interpreted as financial or investment advice.

---

# Author

**Anish A. Kale**

M.S. Computational Data Science  
UC Riverside
