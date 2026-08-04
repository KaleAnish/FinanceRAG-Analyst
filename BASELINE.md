# FinanceRAG-Analyst: Pre-Rebuild Baseline

## Purpose

This file records the state of the original notebook-based implementation before the retrieval and evaluation rebuild.

The original implementation is preserved by the Git tag:

`notebook-baseline-v1`

The rebuild is being developed on:

`evaluation-rebuild`

This baseline is descriptive, not a claim that the original stored outputs are independently reproducible or statistically validated.

## Original Pipeline

### Notebook 1 — SEC Filing Ingestion

Original file:

`notebooks/AAK_FinRAG_NtBk1_GIT.ipynb`

Primary responsibilities:

- load SEC request configuration;
- retrieve the SEC ticker-to-CIK mapping;
- select an MVP company set;
- retrieve recent filing metadata;
- construct SEC filing URLs;
- download filing HTML or text;
- save raw filings;
- flag suspicious filing records;
- save the final raw dataset;
- perform final validation checks.

Baseline inventory:

- 27 total cells;
- 14 code cells;
- 13 Markdown cells;
- 14 executed code cells;
- 17 stored outputs.

### Notebook 2 — Document Processing and Chunking

Original file:

`notebooks/02_document_processing_chunking_GIT.ipynb`

Primary responsibilities:

- load Notebook 1 output;
- validate raw filing input;
- clean filing text;
- chunk documents;
- assign lightweight topic labels;
- add chunk-level features;
- run chunk-quality checks;
- check document coverage;
- detect duplicate chunks;
- inspect sample chunks;
- save the processed chunk dataset;
- perform final validation checks.

Baseline inventory:

- 31 total cells;
- 18 code cells;
- 13 Markdown cells;
- 18 executed code cells;
- 22 stored outputs.

### Notebook 3 — Retrieval, Indexing, and Evaluation

Original file:

`notebooks/03_retrieval_indexing_evaluation_GIT.ipynb`

Primary responsibilities:

- load and validate chunks;
- save chunk metadata;
- load an embedding model;
- build or load document embeddings;
- build or load a FAISS index;
- build or load a BM25 index;
- perform query expansion;
- apply metadata and company filtering;
- run dense retrieval;
- run BM25 retrieval;
- run hybrid retrieval using Reciprocal Rank Fusion;
- define an evaluation query set;
- score retrieval relevance;
- evaluate a configuration grid;
- aggregate evaluation results;
- save results and a selected configuration;
- perform failure analysis and retrieval diagnostics;
- demonstrate metadata-filtered retrieval;
- test company-filter safeguards.

Baseline inventory:

- 59 total cells;
- 31 code cells;
- 28 Markdown cells;
- 31 executed code cells;
- 75 stored outputs.

### Notebook 4 — RAG Answering

Original file:

`notebooks/04_rag_answering_provider_flexible GIT.ipynb`

Primary responsibilities:

- load retrieval artifacts;
- load the embedding model;
- configure provider-flexible LLM access;
- reproduce query expansion and company detection;
- reproduce dense, BM25, and hybrid retrieval;
- load the selected retrieval configuration;
- evaluate evidence adequacy;
- build citation context;
- support multiple answer modes and prompts;
- call a configured LLM provider;
- execute the main RAG answering pipeline;
- validate generated answers;
- save sample answers;
- test unsupported and partially supported questions;
- perform final operational checks.

Baseline inventory:

- 52 total cells;
- 31 code cells;
- 21 Markdown cells;
- 31 executed code cells;
- 36 stored outputs.

## Baseline Strengths

The original implementation already contains several meaningful RAG and information-retrieval components:

- SEC EDGAR ingestion;
- structured filing metadata;
- document cleaning and chunking;
- dense retrieval with FAISS;
- sparse retrieval with BM25;
- hybrid retrieval using Reciprocal Rank Fusion;
- query expansion;
- metadata-aware retrieval;
- company-intent safeguards;
- a retrieval configuration grid;
- retrieval diagnostics and failure analysis;
- citation-oriented answer construction;
- evidence-adequacy guardrails;
- provider-flexible answer generation.

These components make the project more substantial than a basic document-chat demonstration.

## Baseline Limitations Requiring Validation

The existence of executed cells and stored outputs does not by itself establish reproducibility or evaluation credibility.

Before publishing final performance claims, the rebuild must verify:

- exact corpus composition and filing coverage;
- deterministic document and chunk identifiers;
- absence of accidental data leakage;
- evaluation-query provenance;
- relevance-label provenance and consistency;
- separation between tuning and final evaluation;
- explicit sparse, dense, and hybrid baselines;
- correct implementation of retrieval metrics;
- per-query results rather than aggregate results alone;
- sensitivity to retrieval depth and fusion parameters;
- statistical uncertainty where appropriate;
- failure-category definitions;
- citation correctness;
- answer faithfulness;
- unsupported-question abstention behavior;
- clean execution from declared dependencies and documented artifacts.

## Rebuild Objective

The rebuild will retain the useful pipeline design while replacing unsupported or ambiguous performance claims with a reproducible evaluation framework.

The target deliverables are:

1. a versioned evaluation query set;
2. explicit relevance judgments;
3. deterministic retrieval runs;
4. BM25, dense, and hybrid baseline comparisons;
5. standard retrieval metrics such as Recall@k, Precision@k, MRR, and nDCG@k where appropriate;
6. per-query and aggregate result files;
7. documented configuration selection;
8. failure analysis tied to actual query results;
9. citation and answer-grounding evaluation;
10. a concise, defensible project summary for GitHub and resume use.
