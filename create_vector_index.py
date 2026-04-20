"""
Script one-time para crear los índices en MongoDB Atlas:
  - vector_index  → búsqueda semántica por similitud coseno (384 dims)
  - search_index  → búsqueda full-text BM25 (Atlas Search / Lucene)

Uso:
    python create_vector_index.py
"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

load_dotenv()

MONGODB_URI   = os.environ["MONGODB_CONNECTION_STRING"]
DB_NAME       = os.getenv("MONGODB_DB", "lec-project")
COLLECTION    = os.getenv("MONGODB_COLLECTION", "embeddings")
EMBEDDING_DIM = 384  # bge-small-en-v1.5


def create_indexes() -> None:
    client = MongoClient(MONGODB_URI)
    collection = client[DB_NAME][COLLECTION]

    # ── 1. Vector Search index (búsqueda semántica)
    vector_index = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": EMBEDDING_DIM,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "metadata.source"},
                {"type": "filter", "path": "metadata.topic"},
                {"type": "filter", "path": "metadata.date"},
            ]
        },
        name="vector_index",
        type="vectorSearch",
    )

    # ── 2. Atlas Search index (BM25 / full-text)
    search_index = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": False,
                "fields": {
                    "text": {"type": "string"},
                    "metadata": {
                        "type": "document",
                        "fields": {
                            "title":  {"type": "string"},
                            "source": {"type": "string"},
                            "topic":  {"type": "string"},
                            "date":   {"type": "string"},
                        },
                    },
                },
            }
        },
        name="search_index",
        type="search",
    )

    existing = {idx["name"] for idx in collection.list_search_indexes()}

    for model, label in [(vector_index, "vector_index"), (search_index, "search_index")]:
        if label in existing:
            print(f"  ✓ '{label}' ya existe — se omite.")
        else:
            collection.create_search_index(model)
            print(f"  ✓ '{label}' creado.")

    print("\nListo. Los índices pueden tardar 1-2 min en activarse en Atlas.")
    client.close()


if __name__ == "__main__":
    create_indexes()