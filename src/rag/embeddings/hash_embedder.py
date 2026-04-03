import hashlib
import math

from rag.embeddings.base import Embedder


class HashingEmbedder(Embedder):
    """Minimal offline embedder for the MVP.

    This is intentionally simple so the project runs locally without external model
    dependencies. Replace this with a real embedding provider later.
    """

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._embed(query)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = text.lower().split()
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]
