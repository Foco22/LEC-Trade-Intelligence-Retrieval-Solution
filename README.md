# LEC Trade Intelligence Retrieval Platform

Production-grade RAG system for London Export Corporation. Clara is the internal knowledge assistant that answers questions about UK trade regulations, WTO policy, and FSA food guidance — grounded strictly in indexed documents.

**Live demo:**
- Chat (Clara): https://lec-retrieval-380463151129.northamerica-northeast1.run.app/
- API docs: https://lec-retrieval-380463151129.northamerica-northeast1.run.app/api/docs

> **Note — cold start:** The service runs on Cloud Run with no minimum instances. The first request after a period of inactivity may take ~60 seconds to respond while the container loads the embedding and reranker models. Subsequent requests are fast.

---

## Folder structure

```
lec-retrieval/
├── src/
│   ├── ingesta/        # Step 1 — download and index corpus
│   ├── retrieval/      # Step 2 — BM25 + semantic + reranker search
│   ├── generate/       # Step 3 — answer generation with LLM
│   └── utils/          # shared helpers (embeddings, mongodb, llm)
├── eval/
│   ├── qas.json        # 20+ query/answer pairs
│   └── evaluate.py     # precision@5, recall@5, NDCG@5
├── prompts/
│   └── rag_prompt.txt
├── api/
│   └── main.py         # FastAPI — POST /search
├── streamlit/
│   └── app.py          # Clara chat frontend
├── create_vector_index.py
├── Dockerfile
└── requirements.txt
```

---

## Run

### With Docker

```bash
# 1. Copy and fill in credentials
cp .env.example .env

# 2. Build and run the API
docker build -t lec-retrieval .
docker run --env-file .env -p 8000:8000 lec-retrieval
```

### Without Docker

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill in credentials
cp .env.example .env

# 3. Create MongoDB Atlas indexes (one-time)
python create_vector_index.py

# 4. Download and index corpus
python -m src.ingesta.ingest

# 5. Start Streamlit frontend
streamlit run streamlit/app.py

# 6. Or start the API
uvicorn api.main:app --reload
```

### API usage

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "import duties on beverages", "mode": "hybrid_rerank", "top_k": 5}'
```

### Evaluation

```bash
python eval/evaluate.py
```

Outputs precision@5, recall@5, and NDCG@5 across three search configurations: semantic, hybrid, and hybrid+rerank.