from rag.schemas.document import IndexedRecord
from rag.schemas.retrieval import RetrievalResult
from rag.vectorstores.base import VectorStore


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._records: list[IndexedRecord] = []

    def upsert(self, records: list[IndexedRecord]) -> None:
        if not records:
            return

        by_id = {record.chunk_id: record for record in self._records}
        for record in records:
            by_id[record.chunk_id] = record
        self._records = list(by_id.values())

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        path_filter: str | None = None,
    ) -> list[RetrievalResult]:
        candidates = self._records
        if path_filter:
            candidates = [
                record
                for record in self._records
                if record.source_path == path_filter
                or str(record.metadata.get("source_path", "")) == path_filter
                or str(record.metadata.get("document_path", "")) == path_filter
            ]

        ranked = sorted(
            candidates,
            key=lambda record: self._cosine_similarity(query_embedding, record.embedding),
            reverse=True,
        )
        results: list[RetrievalResult] = []
        for record in ranked[:top_k]:
            score = self._cosine_similarity(query_embedding, record.embedding)
            results.append(
                RetrievalResult(
                    chunk_id=record.chunk_id,
                    document_id=record.document_id,
                    source_path=record.source_path,
                    text=record.text,
                    score=score,
                    distance=1.0 - score,
                    relevance=self._score_to_relevance(score),
                    metadata=dict(record.metadata),
                )
            )
        return results

    def check_health(self) -> None:
        return None

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        return sum(left_value * right_value for left_value, right_value in zip(left, right))

    @staticmethod
    def _score_to_relevance(score: float) -> str:
        if score >= 0.35:
            return "high"
        if score >= 0.15:
            return "medium"
        return "low"
