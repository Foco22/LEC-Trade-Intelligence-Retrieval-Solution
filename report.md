# LEC Trade Intelligence Retrieval Platform —  Personal Report

---

## Overview

Clara is a production-grade RAG system built for London Export Corporation that lets employees query UK trade regulations, WTO policy documents, and FSA food guidance in natural language. It combines BM25 full-text search, semantic vector search, and a cross-encoder reranker over 1,100+ indexed documents, exposed through a Streamlit chat interface and a FastAPI endpoint — deployed live on Google Cloud Run.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  INGESTION  (one-time + incremental)                            │
│                                                                 │
│  Corpus (GOV.UK · WTO PDFs · FSA)                              │
│       │  MarkItDown → Markdown → header-based chunks            │
│       │  SHA-256 manifest (skips unchanged docs)                │
│       ▼                                                         │
│  MongoDB Atlas ── Vector Index (cosine, 384 dims)               │
│                └─ Search Index (BM25 / Lucene)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │  1,100+ documents indexed
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  BACKEND  ·  Google Cloud Run                                   │
│                                                                 │
│  User question                                                  │
│       │                                                         │
│       ▼                                                         │
│  FastAPI  POST /search                                          │
│       │                                                         │
│       ├── BM25 fetch    ──▶  MongoDB Atlas Search               │
│       ├── Semantic fetch ──▶ MongoDB Atlas Vector Search        │
│       ├── RRF fusion    (Reciprocal Rank Fusion)                │
│       └── Reranker      (cross-encoder/ms-marco-MiniLM-L-6-v2) │
│       │                                                         │
│       ▼                                                         │
│  Generate  (GPT-4o-mini · context-only · cost tracked)         │
│       │                                                         │
│       └──▶  LangSmith  (traces every query + LLM call)         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │  answer + sources + scores
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  FRONTEND  ·  Google Cloud Run (same container)                 │
│                                                                 │
│  Streamlit                                                      │
│  ├── Chat tab       Clara — conversational Q&A                  │
│  ├── Evaluation tab precision@5 · recall@5 · NDCG@5            │
│  └── Ingesta Status docs · chunks · topics in MongoDB           │
└─────────────────────────────────────────────────────────────────┘
```

**Stack**

| Layer | Choice |
|---|---|
| Vector store | MongoDB Atlas (Vector Search + Atlas Search) |
| Embeddings | `BAAI/bge-small-en-v1.5` — local, no API key |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` — local |
| LLM | GPT-4o-mini via OpenAI API |
| API | FastAPI |
| Frontend | Streamlit |
| Monitoring | LangSmith (query traces + LLM call tracking) |
| Infra | Docker · GitHub Actions CI/CD · Google Cloud Run |

---

## What the project has

- **Hybrid search** — BM25 + semantic + cross-encoder reranker, composable weights at query time
- **Score breakdown** — every result returns BM25, semantic, and reranker scores separately
- **Metadata filtering** — filter by source (govuk / wto / fsa) and topic at query time
- **Incremental ingestion** — SHA-256 manifest skips re-embedding unchanged documents
- **1,100+ documents indexed** — GOV.UK trade guidance, WTO Trade Policy Reviews (PDFs), FSA food regulations
- **Evaluation suite** — 20 query/answer pairs, precision@5 / recall@5 / NDCG@5 across 3 configurations
- **FastAPI** — `POST /search` with mode selection, metadata filter, and full score breakdown
- **Streamlit frontend** — Chat (Clara), Evaluation tab, Ingesta Status tab
- **CI/CD** — GitHub Actions: pytest on every push, Docker build + Cloud Run deploy on merge to master
- **Live on Cloud Run** — https://lec-retrieval-380463151129.northamerica-northeast1.run.app/

---

## Hard Mode

### Which chunk size wins and why

The honest answer is: the exact token count matters less than the chunking strategy.

Best practices (and benchmarks on retrieval-augmented tasks) converge around 2,000–2,200 tokens with ~200 token overlap. Spending engineering time sweeping 256 / 512 / 1024 produces marginal NDCG gains that disappear when the embedding model changes.

What actually moves the needle is matching the chunking unit to the document's semantic structure. GOV.UK guidance and WTO policy documents are written with `##` headers that mark genuine topic boundaries — a section on "import duties" is a coherent retrieval unit. Chunking by header rather than by fixed token count means each chunk corresponds to something a human would call an "answer", not an arbitrary window. That is the architectural decision, and it is where the value is.

### When does BM25 beat semantic? Failure modes

**BM25 wins** when the query contains exact terminology that the embedding model generalises away:

- `"HS code 2203 beer import"` — BM25 matches the exact code; semantic retrieves documents about beer generally and may rank the wrong product category higher.
- `"Article 17.1 WTO Agreement"` — BM25 finds the exact article reference; semantic retrieves thematically related passages that never contain that string.
- Rare proper nouns, regulation IDs, and product codes are BM25's domain.

**Semantic wins** when the user paraphrases or uses different vocabulary than the document:

- `"rules about bringing alcohol into the UK"` matches documents that say "import duty on beverages" — BM25 scores zero, semantic retrieves correctly.
- Cross-lingual or domain-shifted queries (e.g. a non-expert asking about something an expert document calls by a technical name).

**Hybrid + reranker** is the robust default precisely because neither alone covers both failure modes.

### Cold-cache vs warm-cache latency profile

`[ PENDING — to be measured with time curl against live endpoint ]`

### Cost per 1,000 queries projected

Every query tracks `input_tokens`, `output_tokens`, and `cost_usd` via `LLMResponse`. At GPT-4o-mini pricing ($0.150 / 1M input · $0.600 / 1M output):

- Average observed: ~1,200 input tokens + ~300 output tokens per query
- Cost per query: ~$0.00018 + ~$0.00018 = **~$0.00036**
- **Projected cost per 1,000 queries: ~$0.36**

Retrieval is free (local models, MongoDB Atlas free tier). The bottleneck cost is purely LLM generation.

---

## What I'd ship next with one more week

**Analytics and session store.** You cannot improve what you do not measure. The first thing I would ship is a database of sessions and conversations. Every query, every result, every response — logged. Without this, iteration is guesswork.

**Conversation triage agent.** Once sessions are stored, I would run an agent over all conversations to automatically label and characterise ones with low engagement or failed answers. The agent flags what went wrong, clusters failure types, and feeds that signal back into evaluation. You cannot learn if you do not know where you are failing.

**Move RAG to a tool inside an agent.** The current architecture returns the 20 nearest chunks and summarises them. That breaks on complex queries: *"Give me the difference between FSA and GOV.UK guidance on the same topic"* requires reading across sources, planning sub-queries, and synthesising. RAG as a tool inside a reasoning agent — one that can decide to query multiple times, compare results, and structure its own answer — handles this. The retrieval pipeline stays; what changes is the orchestration layer above it.

**Live ingestion pipeline.** Trade regulations change daily. I would build a scheduled process that monitors source URLs for changes, re-indexes updated documents, and removes stale ones from MongoDB — keeping the corpus current without manual intervention.

**People and process.** The backend is the easy part. Before shipping to real users I would meet with key LEC stakeholders to define what success looks like — which questions Clara should answer well, which it should decline, and how employees will actually use it in their workflow. Technology is a solved problem here; adoption is not.

---

## AI Usage

`[ PENDING ]`
