from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from src.retrieval.retrieval import retrieval
from src.utils.mongodb import mongo
from api.models import SearchRequest, SearchResponse, SearchResultResponse, ScoresResponse

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    mongo.connect()
    yield
    mongo.close()


app = FastAPI(title="LEC Retrieval API", lifespan=lifespan, root_path="/api")


# ── Endpoints ────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    try:
        if request.mode == "semantic":
            results = await retrieval.semantic_search(
                query=request.query,
                top_k=request.top_k,
                metadata_filter=request.metadata_filter,
            )
        elif request.mode == "hybrid":
            results = await retrieval.hybrid_search(
                query=request.query,
                top_k=request.top_k,
                bm25_weight=request.bm25_weight,
                semantic_weight=request.semantic_weight,
                metadata_filter=request.metadata_filter,
            )
        else:
            results = await retrieval.hybrid_rerank_search(
                query=request.query,
                top_k=request.top_k,
                bm25_weight=request.bm25_weight,
                semantic_weight=request.semantic_weight,
                metadata_filter=request.metadata_filter,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return SearchResponse(
        query=request.query,
        mode=request.mode,
        results=[
            SearchResultResponse(
                doc_id=r.doc_id,
                text=r.text,
                metadata=r.metadata,
                scores=ScoresResponse(
                    bm25=r.scores.bm25,
                    semantic=r.scores.semantic,
                    reranker=r.scores.reranker,
                ),
            )
            for r in results
        ],
    )