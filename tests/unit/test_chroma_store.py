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
