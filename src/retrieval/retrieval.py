import os
from dataclasses import dataclass, field
from sentence_transformers import CrossEncoder
from src.utils.mongodb import mongo
from src.utils.embeddings import embeddings

@dataclass
class Scores:
    bm25:     float = 0.0
    semantic: float = 0.0
    reranker: float = 0.0


@dataclass
class SearchResult:
    doc_id:   str
    text:     str
    metadata: dict
    scores:   Scores = field(default_factory=Scores)


class Retrieval:
    def __init__(self):
        self._reranker_model = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self._candidates     = int(os.getenv("RETRIEVAL_CANDIDATES", "20"))
        self._reranker: CrossEncoder | None = None

    def _get_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(self._reranker_model)
        return self._reranker

    # ── 1. Semántico ────────────────────────────────

    async def semantic_search(
        self,
        query:           str,
        top_k:           int = 5,
        metadata_filter: dict | None = None,
    ) -> list[SearchResult]:
        """Búsqueda por similitud de vectores (coseno)."""
        docs = await self._fetch_semantic(query, metadata_filter, top_k)
        return [
            SearchResult(
                doc_id=d["doc_id"],
                text=d["text"],
                metadata=d.get("metadata", {}),
                scores=Scores(semantic=d["semantic_score"]),
            )
            for d in docs
        ]

    # ── 2. Híbrido (BM25 + semántico) ───────────────

    async def hybrid_search(
        self,
        query:           str,
        top_k:           int   = 5,
        bm25_weight:     float = 0.5,
        semantic_weight: float = 0.5,
        metadata_filter: dict | None = None,
    ) -> list[SearchResult]:
        """BM25 + semántico fusionados con pesos configurables."""
        bm25_docs, semantic_docs = await self._fetch_both(query, metadata_filter)
        return self._fuse(bm25_docs, semantic_docs, bm25_weight, semantic_weight)[:top_k]

    # ── 3. Híbrido + Reranker ────────────────────────

    async def hybrid_rerank_search(
        self,
        query:           str,
        top_k:           int   = 5,
        bm25_weight:     float = 0.5,
        semantic_weight: float = 0.5,
        metadata_filter: dict | None = None,
    ) -> list[SearchResult]:
        """BM25 + semántico fusionados, reordenados por cross-encoder."""
        bm25_docs, semantic_docs = await self._fetch_both(query, metadata_filter)
        fused   = self._fuse(bm25_docs, semantic_docs, bm25_weight, semantic_weight)
        return self._rerank(query, fused[:top_k * 2])[:top_k]

    # ── Fetch helpers ────────────────────────────────

    async def _fetch_semantic(self, query: str, metadata_filter: dict | None, n: int) -> list[dict]:
        query_vector = embeddings.encode(query)
        vector_stage = {
            "$vectorSearch": {
                "index":         "vector_index",
                "path":          "embedding",
                "queryVector":   query_vector,
                "numCandidates": n * 10,
                "limit":         n,
            }
        }
        if metadata_filter:
            vector_stage["$vectorSearch"]["filter"] = {
                f"metadata.{k}": {"$eq": v} for k, v in metadata_filter.items()
            }
        pipeline = [
            vector_stage,
            {"$addFields": {"semantic_score": {"$meta": "vectorSearchScore"}}},
            {"$project": {"embedding": 0}},
        ]
        collection = mongo.get_collection()
        return await collection.aggregate(pipeline).to_list(n)

    async def _fetch_bm25(self, query: str, metadata_filter: dict | None, n: int) -> list[dict]:
        must    = [{"text": {"query": query, "path": "text"}}]
        filter_ = [
            {"equals": {"path": f"metadata.{k}", "value": v}}
            for k, v in (metadata_filter or {}).items()
        ]
        search_stage = {
            "$search": {
                "index": "search_index",
                "compound": {"must": must, "filter": filter_} if filter_ else {"must": must},
            }
        }
        pipeline = [
            search_stage,
            {"$limit": n},
            {"$addFields": {"bm25_score": {"$meta": "searchScore"}}},
            {"$project": {"embedding": 0}},
        ]
        collection = mongo.get_collection()
        return await collection.aggregate(pipeline).to_list(n)

    async def _fetch_both(self, query: str, metadata_filter: dict | None) -> tuple[list[dict], list[dict]]:
        bm25_docs     = await self._fetch_bm25(query, metadata_filter, self._candidates)
        semantic_docs = await self._fetch_semantic(query, metadata_filter, self._candidates)
        return bm25_docs, semantic_docs

    # ── Fusión ───────────────────────────────────────

    def _fuse(
        self,
        bm25_docs:       list[dict],
        semantic_docs:   list[dict],
        bm25_weight:     float,
        semantic_weight: float,
    ) -> list[SearchResult]:
        merged: dict[str, SearchResult] = {}
        bm25_max = max((d["bm25_score"] for d in bm25_docs), default=1.0) or 1.0

        for doc in bm25_docs:
            key = f"{doc['doc_id']}_{doc['chunk_index']}"
            merged[key] = SearchResult(
                doc_id=doc["doc_id"],
                text=doc["text"],
                metadata=doc.get("metadata", {}),
                scores=Scores(bm25=doc["bm25_score"] / bm25_max),
            )

        for doc in semantic_docs:
            key = f"{doc['doc_id']}_{doc['chunk_index']}"
            if key in merged:
                merged[key].scores.semantic = doc["semantic_score"]
            else:
                merged[key] = SearchResult(
                    doc_id=doc["doc_id"],
                    text=doc["text"],
                    metadata=doc.get("metadata", {}),
                    scores=Scores(semantic=doc["semantic_score"]),
                )

        for result in merged.values():
            result.scores.reranker = (
                result.scores.bm25 * bm25_weight +
                result.scores.semantic * semantic_weight
            )

        return sorted(merged.values(), key=lambda r: r.scores.reranker, reverse=True)

    # ── Reranker ─────────────────────────────────────

    def _rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        reranker = self._get_reranker()
        pairs    = [(query, r.text) for r in results]
        scores   = reranker.predict(pairs).tolist()

        for result, score in zip(results, scores):
            result.scores.reranker = round(float(score), 6)

        return sorted(results, key=lambda r: r.scores.reranker, reverse=True)


retrieval = Retrieval()