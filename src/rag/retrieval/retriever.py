from rag.embeddings.base import Embedder
from rag.schemas.retrieval import RetrievalResult
from rag.vectorstores.base import VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        query_embedding = self.embedder.embed_query(query)
        return self.vector_store.query(query_embedding=query_embedding, top_k=top_k)
