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

To be honest, i think that chunk strategy is not matter regarding the exactly number of token and token overlap. 

There is a good practice brenchmarks that converge around 2.000-2.200 tokens with -200 token overlap. Spending time if the best number is 1900 or 1500 does not counts. I would spend time more thinking chunks strategy based on type of files more numbers.

My point: ¿What strategy should i use if i have a workflow document?. I think i need to capture the whole diagram and their explanation, and then transform to chunks more than divide the file into a chunks.

I focus in the answer, not in the parameters, so dependen of the files, it can be different type of parameters. 


### When does BM25 beat semantic?

**BM25 wins** when the query contains exact terminology that the embedding model generalises away:

- `"HS code 2203 beer import"` — BM25 matches the exact code; semantic retrieves documents about beer generally and may rank the wrong product category higher.
- `"Article 17.1 WTO Agreement"` — BM25 finds the exact article reference; semantic retrieves thematically related passages that never contain that string.

**Semantic wins** when the user paraphrases or uses different vocabulary than the document:

- `"rules about bringing alcohol into the UK"` matches documents that say "import duty on beverages" — BM25 scores zero, semantic retrieves correctly.
- Cross-lingual or domain-shifted queries (e.g. a non-expert asking about something an expert document calls by a technical name).

**Hybrid + reranker** is the robust precisely because neither alone covers both failure modes.


### Cost per 1,000 queries projected

Every query tracks `input_tokens`, `output_tokens`, and `cost_usd` via `LLMResponse`. Token counts measured across the 20 QA pairs using `tiktoken`. GPT-4o-mini pricing: $0.150 / 1M input · $0.600 / 1M output (source: https://developers.openai.com/api/docs/pricing).

| Component | Avg per query | Cost per query | Cost per 1,000 queries |
|---|---|---|---|
| Input tokens | 1,566 | $0.000235 | $0.235 |
| Output tokens (~300 est.) | ~300 | $0.000180 | $0.180 |
| Embeddings (local) | — | $0.000000 | $0.00 |
| Reranker (local) | — | $0.000000 | $0.00 |
| Cloud Run (2 vCPU / 2Gi, ~2s/query) | — | ~$0.000020 | ~$0.02 |
| **Total** | | **~$0.000435** | **~$0.44** |

_* Input token average measured with `tiktoken` across the 20 evaluation queries (`eval/count_tokens.py`). Output tokens estimated; to be updated with LangSmith trace data._

Retrieval and embeddings are free — local models, MongoDB Atlas free tier. The dominant cost is LLM generation.

### What problems?

I think RAG has a big two problems, and i would choose here:

    - **Business Oportunity**: No one want to system only give you answer, they want to work with you. RAG are hard to measure because there is not clear bussines KPI to measure the scuessuc, so it must to move this logic to agent that can work into a bussines process. 

    -**People ask how they want**: RAG is getting short in the scope, because people are tirdy to ask, so you need to have a layer of thinking before to answwer a question.  Besides, you need to have a layer of validation as well.

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
