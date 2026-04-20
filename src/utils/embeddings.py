import os
from sentence_transformers import SentenceTransformer


class EmbeddingHuggingFace:
    def __init__(self):
        self._model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        self._model = SentenceTransformer(self._model_name)

    def encode(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()


embeddings = EmbeddingHuggingFace()