from typing import Literal
from pydantic import BaseModel


class SearchRequest(BaseModel):
    query:           str
    top_k:           int                                            = 5
    mode:            Literal["semantic", "hybrid", "hybrid_rerank"] = "hybrid_rerank"
    bm25_weight:     float                                          = 0.5
    semantic_weight: float                                          = 0.5
    metadata_filter: dict | None                                   = None


class ScoresResponse(BaseModel):
    bm25:     float
    semantic: float
    reranker: float


class SearchResultResponse(BaseModel):
    doc_id:   str
    text:     str
    metadata: dict
    scores:   ScoresResponse


class SearchResponse(BaseModel):
    query:   str
    mode:    str
    results: list[SearchResultResponse]