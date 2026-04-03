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

    def query(self, query_embedding: list[float], top_k: int) -> list[RetrievalResult]:
        ranked = sorted(
            self._records,
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
                    metadata=dict(record.metadata),
                )
            )
        return results

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        return sum(left_value * right_value for left_value, right_value in zip(left, right))
