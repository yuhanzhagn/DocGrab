from rag.embeddings.hash_embedder import HashingEmbedder
from rag.schemas.document import IndexedRecord
from rag.vectorstores.chroma_store import ChromaVectorStore


def test_chroma_store_upsert_and_query(temp_vector_store: ChromaVectorStore) -> None:
    embedder = HashingEmbedder(dimension=64)
    texts = [
        "Chroma stores embeddings for retrieval.",
        "Citations point users to source documents.",
    ]
    embeddings = embedder.embed_texts(texts)
    records = [
        IndexedRecord(
            chunk_id=f"chunk-{index}",
            document_id=f"doc-{index}",
            source_path=f"/tmp/doc-{index}.txt",
            text=text,
            embedding=embedding,
            metadata={
                "chunk_index": index,
                "start_char": 0,
                "end_char": len(text),
            },
        )
        for index, (text, embedding) in enumerate(zip(texts, embeddings))
    ]

    temp_vector_store.upsert(records)
    results = temp_vector_store.query(
        query_embedding=embedder.embed_query("Where are embeddings stored?"),
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-0"
    assert "Chroma" in results[0].text
    assert results[0].source_path == "/tmp/doc-0.txt"
    assert results[0].distance is not None
    assert results[0].relevance in {"high", "medium", "low"}


def test_chroma_store_query_supports_path_filter(temp_vector_store: ChromaVectorStore) -> None:
    embedder = HashingEmbedder(dimension=64)
    texts = [
        "Chroma stores embeddings for retrieval.",
        "Citations point users to source documents.",
    ]
    embeddings = embedder.embed_texts(texts)
    records = [
        IndexedRecord(
            chunk_id=f"chunk-{index}",
            document_id=f"doc-{index}",
            source_path=f"/tmp/doc-{index}.txt",
            text=text,
            embedding=embedding,
            metadata={
                "chunk_index": index,
                "start_char": 0,
                "end_char": len(text),
            },
        )
        for index, (text, embedding) in enumerate(zip(texts, embeddings))
    ]

    temp_vector_store.upsert(records)
    results = temp_vector_store.query(
        query_embedding=embedder.embed_query("source documents"),
        top_k=2,
        path_filter="/tmp/doc-1.txt",
    )

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"


def test_chroma_store_uses_http_client_when_host_is_configured(tmp_path) -> None:
    captured: dict[str, object] = {}

    class _FakeClient:
        def get_or_create_collection(self, name: str, metadata: dict) -> object:
            captured["collection_name"] = name
            captured["metadata"] = metadata

            class _FakeCollection:
                def upsert(self, **_: object) -> None:
                    return None

                def query(self, **_: object) -> dict[str, list[list[object]]]:
                    return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

            return _FakeCollection()

    def _fake_http_client_factory(host: str, port: int, ssl: bool) -> _FakeClient:
        captured["host"] = host
        captured["port"] = port
        captured["ssl"] = ssl
        return _FakeClient()

    ChromaVectorStore(
        persist_directory=tmp_path / "unused",
        collection_name="remote-docs",
        host="chroma",
        port=8000,
        ssl=False,
        http_client_factory=_fake_http_client_factory,
    )

    assert captured["host"] == "chroma"
    assert captured["port"] == 8000
    assert captured["ssl"] is False
    assert captured["collection_name"] == "remote-docs"
