from rag.embeddings.base import Embedder
from rag.schemas.retrieval import RetrievalResult
from rag.vectorstores.base import VectorStore


class Retriever:
    _OVERFETCH_FACTOR = 3
    _MIN_OVERFETCH = 5
    _DEDUPLICATION_THRESHOLD = 0.9

    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int,
        path_filter: str | None = None,
    ) -> list[RetrievalResult]:
        query_embedding = self.embedder.embed_query(query)
        raw_results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=max(top_k * self._OVERFETCH_FACTOR, top_k + self._MIN_OVERFETCH),
            path_filter=path_filter,
        )
        return self._deduplicate(raw_results, top_k=top_k)

    def _deduplicate(
        self,
        retrieval_results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        selected: list[RetrievalResult] = []
        for candidate in retrieval_results:
            if any(self._is_near_duplicate(candidate.text, existing.text) for existing in selected):
                continue
            selected.append(candidate)
            if len(selected) >= top_k:
                break
        return selected

    def _is_near_duplicate(self, left: str, right: str) -> bool:
        left_tokens = set(left.lower().split())
        right_tokens = set(right.lower().split())
        if not left_tokens or not right_tokens:
            return left.strip() == right.strip()

        overlap = len(left_tokens & right_tokens)
        total = len(left_tokens | right_tokens)
        if total == 0:
            return False
        return (overlap / total) >= self._DEDUPLICATION_THRESHOLD
