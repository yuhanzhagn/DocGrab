from rag.embeddings.hash_embedder import HashingEmbedder
from rag.retrieval.retriever import Retriever
from rag.schemas.document import IndexedRecord
from rag.vectorstores.in_memory_store import InMemoryVectorStore


def test_retriever_deduplicates_highly_similar_chunks() -> None:
    embedder = HashingEmbedder(dimension=64)
    store = InMemoryVectorStore()
    retriever = Retriever(embedder=embedder, vector_store=store)

    texts = [
        "Chroma stores embeddings for retrieval and supports metadata filtering.",
        "Chroma stores embeddings for retrieval and supports metadata filtering.",
        "Citations point users back to source documents and chunk spans.",
    ]
    embeddings = embedder.embed_texts(texts)
    store.upsert(
        [
            IndexedRecord(
                chunk_id=f"chunk-{index}",
                document_id=f"doc-{index}",
                source_path=f"/tmp/doc-{index}.txt",
                text=text,
                embedding=embedding,
                metadata={"chunk_index": index},
            )
            for index, (text, embedding) in enumerate(zip(texts, embeddings))
        ]
    )

    results = retriever.retrieve(query="How does Chroma support retrieval?", top_k=3)

    assert len(results) == 2
    assert [result.chunk_id for result in results] == ["chunk-0", "chunk-2"]


def test_retriever_passes_path_filter() -> None:
    embedder = HashingEmbedder(dimension=64)
    store = InMemoryVectorStore()
    retriever = Retriever(embedder=embedder, vector_store=store)

    texts = [
        "Chroma stores embeddings for retrieval.",
        "Answers should include citations that point back to the source file and chunk.",
    ]
    embeddings = embedder.embed_texts(texts)
    store.upsert(
        [
            IndexedRecord(
                chunk_id=f"chunk-{index}",
                document_id=f"doc-{index}",
                source_path=f"/tmp/doc-{index}.txt",
                text=text,
                embedding=embedding,
                metadata={"chunk_index": index},
            )
            for index, (text, embedding) in enumerate(zip(texts, embeddings))
        ]
    )

    results = retriever.retrieve(
        query="Where do citations point?",
        top_k=2,
        path_filter="/tmp/doc-1.txt",
    )

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"
