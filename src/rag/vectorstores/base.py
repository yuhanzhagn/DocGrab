from abc import ABC, abstractmethod

from rag.schemas.document import IndexedRecord
from rag.schemas.retrieval import RetrievalResult


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, records: list[IndexedRecord]) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        path_filter: str | None = None,
    ) -> list[RetrievalResult]:
        raise NotImplementedError
