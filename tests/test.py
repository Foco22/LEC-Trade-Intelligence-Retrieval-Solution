import pytest
from dotenv import load_dotenv
from src.ingesta.ingest import chunk_document
from src.retrieval.retrieval import retrieval
from src.generate.generate import generate
from src.utils.mongodb import mongo

load_dotenv()


@pytest.fixture(autouse=True, scope="session")
def connect_mongo():
    mongo.connect()
    yield
    mongo.close()


# ── 1. Ingesta ───────────────────────────────────────

def test_ingest_chunks():
    text = """TITLE: Test Document
SOURCE: govuk
---

Introduction paragraph with some content about importing goods.

## Section One

This section covers customs declarations and import duties for goods
arriving from China into the United Kingdom.

## Section Two

This section covers alcohol licensing requirements for importers
of beverages including beer and wine products.
"""
    chunks = chunk_document(text)
    assert len(chunks) >= 2
    assert all(len(c) > 0 for c in chunks)
    assert any("customs" in c.lower() for c in chunks)


# ── 2. Semantic search ───────────────────────────────

@pytest.mark.asyncio
async def test_semantic_search():
    results = await retrieval.semantic_search(
        query="import duty beer UK",
        top_k=3,
    )
    assert len(results) > 0
    assert all(r.scores.semantic > 0 for r in results)
    assert all(r.text for r in results)


# ── 3. Hybrid search ─────────────────────────────────

@pytest.mark.asyncio
async def test_hybrid_search():
    results = await retrieval.hybrid_search(
        query="customs declaration import China",
        top_k=3,
        bm25_weight=0.5,
        semantic_weight=0.5,
    )
    assert len(results) > 0
    assert all(r.scores.bm25 > 0 or r.scores.semantic > 0 for r in results)


# ── 4. Hybrid + rerank ───────────────────────────────

@pytest.mark.asyncio
async def test_hybrid_rerank_search():
    results = await retrieval.hybrid_rerank_search(
        query="alcohol licence import regulations",
        top_k=3,
    )
    assert len(results) > 0
    assert all(r.scores.reranker != 0 for r in results)


# ── 5. Generate ──────────────────────────────────────

@pytest.mark.asyncio
async def test_generate():
    results = await retrieval.hybrid_rerank_search(
        query="what are the requirements to import beer into the UK",
        top_k=3,
    )
    response = generate.answer(
        query="what are the requirements to import beer into the UK",
        results=results,
    )
    assert response.content
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert response.cost_usd > 0