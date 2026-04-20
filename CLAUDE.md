# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**LEC Trade Intelligence Retrieval Platform** — a production-grade RAG system built for a London Export Corporation engineering assignment (deadline: 23 April 2026 UK EOD).

LEC is a diversified holding company: beverages (Tsingtao UK importer), robotics, AI/healthcare, and fund management. The retrieval corpus is domain-specific: UK trade regulation (GOV.UK), WTO Trade Policy Reviews (PDFs), and FSA food/beverages guidance.

## Assignment requirements (idea.md)

- Ingest 1,000+ documents (PDFs, text, CSVs), incremental — skip unchanged content
- Hybrid search: BM25 + semantic + cross-encoder reranker, composable weights
- Metadata filtering (source, date, topic)
- `POST /search` → top-5 with score breakdown (BM25 / semantic / reranker contribution)
- Evaluation: ≥20 query/answer pairs, precision@5, recall@5, NDCG across 3 configs (semantic-only / hybrid / hybrid+rerank)
- Latency p95 < 500ms documented

## Stack

| Layer | Choice |
|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` (local, no API key) |
| Vector search | **MongoDB Atlas Vector Search** (free M0 tier) |
| BM25 / full-text | **MongoDB Atlas Search** (Lucene-based) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| PDF/HTML parsing | **MarkItDown** (PDF, HTML → Markdown) |
| API | FastAPI |
| MongoDB client | `motor` (async) |
| Infra | **Docker** |

MongoDB Atlas is the single store for documents, metadata, vectors, and full-text search.  
MarkItDown converts every source to clean Markdown before chunking, preserving headers/tables for structure-aware splitting.

## Project structure

The folders inside `src/` reflect the **pipeline process**, not the technology:

```
lec-retrieval/
├── src/
│   ├── ingesta/                # PASO 1 — descargar y preparar el corpus
│   │   └── ingest.py
│   ├── retrieval/              # PASO 2 — buscar chunks relevantes (BM25 + semántico + reranker)
│   │   └── retrieval.py
│   ├── generate/               # PASO 3 — generar respuesta con LLM sobre los chunks
│   │   └── generate.py
│   └── utils/                  # helpers compartidos (no son un paso del proceso)
│       ├── embeddings.py
│       ├── mongodb.py
│       └── llm.py
├── eval/
│   ├── qas.json                # 20+ query/answer pairs (LEC domain)
│   └── evaluate.py             # precision@5, recall@5, NDCG — 3 configs
├── prompts/
│   └── rag_prompt.txt          # prompt template for generation
├── tests/
│   └── test.py
├── corps/                      # raw downloaded documents (gitignored)
├── api/
│   └── main.py                 # entry point FastAPI (thin wrapper sobre retrieval/)
├── streamlit/
│   └── app.py                  # entry point Streamlit (thin wrapper sobre el pipeline)
├── create_vector_index.py      # one-time MongoDB Atlas index setup
├── Dockerfile
└── requirements.txt
```

## Running the project

```bash
# 1. Start services
docker compose up -d

# 2. Create MongoDB Atlas vector index (one-time)
python create_vector_index.py

# 3. Download corpus
python -m src.ingesta.ingest

# 4. Run Streamlit frontend
streamlit run streamlit/app.py

# 5. Run evaluation
python eval/evaluate.py

# Run tests
pytest tests/
```

## Collaboration style

- Go slow. Write code step by step, one file or function at a time.
- Before writing any code, explain what you're about to do and why.
- After each step, summarize what was written and what comes next.
- Never write multiple files at once without explicit instruction.
- Wait for confirmation before moving to the next step.

## Key design decisions to preserve

- **Chunking**: test 256/512/1024 token sizes, document the winner with numbers
- **Score breakdown**: every `/search` response must include per-result BM25, semantic, and reranker scores separately
- **Incremental ingest**: use SHA-256 checksum (in manifest) to skip re-embedding unchanged docs
- **Weights**: BM25/semantic/reranker blend must be configurable at query time (not hardcoded)
- **LangGraph graph**: each search step (BM25, semantic, rerank, fuse) is a named node — enables per-step latency tracing in LangSmith
