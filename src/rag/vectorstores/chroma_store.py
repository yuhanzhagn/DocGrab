from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection

from rag.schemas.document import IndexedRecord
from rag.schemas.retrieval import RetrievalResult
from rag.vectorstores.base import VectorStore


class ChromaVectorStore(VectorStore):
    def __init__(self, persist_directory: Path, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(path=str(persist_directory))
        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, records: list[IndexedRecord]) -> None:
        if not records:
            return

        ids = [record.chunk_id for record in records]
        documents = [record.text for record in records]
        embeddings = [record.embedding for record in records]
        metadatas = [
            {
                "document_id": record.document_id,
                "source_path": record.source_path,
                **self._normalize_metadata(record.metadata),
            }
            for record in records
        ]

        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, query_embedding: list[float], top_k: int) -> list[RetrievalResult]:
        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        results: list[RetrievalResult] = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            metadata = metadata or {}
            score = 1.0 - float(distance)
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    document_id=str(metadata.get("document_id", "")),
                    source_path=str(metadata.get("source_path", "")),
                    text=text,
                    score=score,
                    metadata=dict(metadata),
                )
            )
        return results

    @staticmethod
    def _normalize_metadata(metadata: dict) -> dict:
        normalized: dict[str, str | int | float | bool] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                normalized[key] = value
            else:
                normalized[key] = str(value)
        return normalized
