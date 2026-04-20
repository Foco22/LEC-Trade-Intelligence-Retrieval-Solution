import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import asyncio
import json
import math
from dotenv import load_dotenv
from src.retrieval.retrieval import retrieval, SearchResult
from src.utils.mongodb import mongo

load_dotenv()

QAS_PATH    = Path("eval/qas.json")
METRICS_PATH = Path("eval/metrics.json")
TOP_K       = 5
MODES       = ["semantic", "hybrid", "hybrid_rerank"]


class Evaluator:
    def __init__(self, top_k: int = TOP_K):
        self._top_k = top_k
        self._qas: list[dict] = []

    def load_qas(self) -> None:
        self._qas = json.loads(QAS_PATH.read_text(encoding="utf-8"))


    def _precision(self, doc_id: str, results: list[SearchResult]) -> float:
        hits = sum(1 for r in results[:self._top_k] if r.doc_id == doc_id)
        return hits / self._top_k

    def _recall(self, doc_id: str, results: list[SearchResult]) -> float:
        return 1.0 if any(r.doc_id == doc_id for r in results[:self._top_k]) else 0.0

    def _ndcg(self, doc_id: str, results: list[SearchResult]) -> float:
        for i, r in enumerate(results[:self._top_k]):
            if r.doc_id == doc_id:
                return 1.0 / math.log2(i + 2)
        return 0.0


    async def _search(self, query: str, source: str, mode: str) -> list[SearchResult]:
        filter_ = {"source": source}
        if mode == "semantic":
            return await retrieval.semantic_search(query=query, top_k=self._top_k, metadata_filter=filter_)
        elif mode == "hybrid":
            return await retrieval.hybrid_search(query=query, top_k=self._top_k, metadata_filter=filter_)
        else:
            return await retrieval.hybrid_rerank_search(query=query, top_k=self._top_k, metadata_filter=filter_)


    async def evaluate_mode(self, mode: str) -> dict:
        precisions, recalls, ndcgs = [], [], []

        for qa in self._qas:
            results = await self._search(qa["query"], qa["source"], mode)
            doc_id  = qa["doc_id"]
            precisions.append(self._precision(doc_id, results))
            recalls.append(self._recall(doc_id, results))
            ndcgs.append(self._ndcg(doc_id, results))

        return {
            "precision@5": round(sum(precisions) / len(precisions), 4),
            "recall@5":    round(sum(recalls)    / len(recalls),    4),
            "ndcg@5":      round(sum(ndcgs)      / len(ndcgs),      4),
        }


    async def run(self) -> dict:
        self.load_qas()
        results = {}
        for mode in MODES:
            results[mode] = await self.evaluate_mode(mode)

        output = {"top_k": self._top_k, "n_queries": len(self._qas), "results": results}
        METRICS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
        return output


async def main():
    mongo.connect()
    evaluator = Evaluator()
    results = await evaluator.run()
    mongo.close()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())