# UCR Crawler Query Interface

This project supports two retrieval backends for Part B:

- `lucene`: sparse retrieval through Elasticsearch
- `bert`: dense retrieval through DistilBERT embeddings stored in FAISS

`query_cli.py` can run either backend or both from the command prompt.

## Quick Start

Open PowerShell in:

```powershell
cd \ucr_crawler
```

## One-Time Setup

### 1. Install Python packages

```powershell
python -m pip install elasticsearch faiss-cpu torch transformers
```

### 2. Start Elasticsearch

This project expects Elasticsearch at `http://localhost:9200`.

Example:

```powershell
cd C:\ES\elasticsearch-9.3.0
.\bin\elasticsearch.bat
```

Keep that terminal open while using Lucene mode.

### 3. Make sure the Elasticsearch index exists

If needed, build the sparse index:

```powershell
cd C:\Users\tangd\Downloads\ucr_crawler
python .\lucene_es\index_docs.py --jsonl .\crawl_out\crawl_data_lucene.jsonl --index ucr_pages
```

### 4. Make sure the dense index exists

Dense files should already be present here:

- `Dense Indexing\dense_distilbert.index`
- `Dense Indexing\dense_distilbert_meta.jsonl`

If you need to rebuild them:

```powershell
python ".\Dense Indexing\dense_index_distilbert_faiss.py" ^
  --input ".\crawl_out\crawl_data.jsonl" ^
  --out_index ".\Dense Indexing\dense_distilbert.index" ^
  --out_meta ".\Dense Indexing\dense_distilbert_meta.jsonl" ^
  --batch_size 8 ^
  --log_every 100 ^
  --save_every_docs 250
```

## Using the Interface

### Lucene only

```powershell
python .\query_cli.py --engine lucene --query "nasa mission" --topk 5 --es_host http://localhost:9200 --es_index ucr_pages
```

### BERT only

```powershell
python .\query_cli.py --engine bert --query "nasa mission" --topk 5 --bert_index ".\Dense Indexing\dense_distilbert.index" --bert_meta ".\Dense Indexing\dense_distilbert_meta.jsonl"
```

### Compare both

```powershell
python .\query_cli.py --engine both --query "nasa mission" --topk 5 --es_host http://localhost:9200 --es_index ucr_pages --bert_index ".\Dense Indexing\dense_distilbert.index" --bert_meta ".\Dense Indexing\dense_distilbert_meta.jsonl"
```

## Interactive Mode

This is the easiest way to use the interface like a mini text-based search UI:

```powershell
python .\query_cli.py --engine both --topk 5 --es_host http://localhost:9200 --es_index ucr_pages --bert_index ".\Dense Indexing\dense_distilbert.index" --bert_meta ".\Dense Indexing\dense_distilbert_meta.jsonl"
```

You will then see prompts like:

```text
Interactive mode. Type 'exit' to quit.
Choose index [lucene/bert/both] (default: both):
Query>
```

Usage:

1. Type `lucene`, `bert`, or `both`
2. Enter a query
3. Review the ranked top-k results
4. Type `exit` to quit

Note: BERT mode may print a one-time DistilBERT model load report when the program starts. This is harmless and does not affect querying.

## Good Demo Queries

### Strong Lucene demos

```powershell
python .\query_cli.py --engine lucene --query "recommender systems collaborative filtering" --topk 5 --es_host http://localhost:9200 --es_index ucr_pages
python .\query_cli.py --engine lucene --query "capital punishment history" --topk 5 --es_host http://localhost:9200 --es_index ucr_pages
python .\query_cli.py --engine lucene --query "genocide convention definition" --topk 5 --es_host http://localhost:9200 --es_index ucr_pages
```

### Compare Lucene and BERT

```powershell
python .\query_cli.py --engine both --query "nasa mission" --topk 5 --es_host http://localhost:9200 --es_index ucr_pages --bert_index ".\Dense Indexing\dense_distilbert.index" --bert_meta ".\Dense Indexing\dense_distilbert_meta.jsonl"
python .\query_cli.py --engine both --query "artemis mission and moon program" --topk 5 --es_host http://localhost:9200 --es_index ucr_pages --bert_index ".\Dense Indexing\dense_distilbert.index" --bert_meta ".\Dense Indexing\dense_distilbert_meta.jsonl"
```

## Evaluation

Run evaluation on the sample query and qrels files:

```powershell
python .\evaluate_partb.py --engine both --queries ".\eval\queries_sample.jsonl" --qrels ".\eval\qrels_sample.jsonl" --k 5 --out_prefix ".\eval\partb_eval_sample" --es_host http://localhost:9200 --es_index ucr_pages --bert_index ".\Dense Indexing\dense_distilbert.index" --bert_meta ".\Dense Indexing\dense_distilbert_meta.jsonl"
```

Outputs are written under `eval\`.

## Notes

- Lucene retrieval is currently the stronger backend on the sample tests.
- BERT retrieval is fully integrated and selectable in the interface, but its current dense ranking quality is weaker with the present model configuration.
