from functools import lru_cache

from rag.chunkers.simple import SimpleTextChunker
from rag.config.settings import get_settings
from rag.embeddings.hash_embedder import HashingEmbedder
from rag.generation.simple import SimpleGroundedAnswerGenerator
from rag.loaders.text_loader import TextDocumentLoader
from rag.retrieval.retriever import Retriever
from rag.services.indexing_service import IndexingService
from rag.services.query_service import QueryService
from rag.vectorstores.chroma_store import ChromaVectorStore


@lru_cache
def get_vector_store() -> ChromaVectorStore:
    settings = get_settings()
    return ChromaVectorStore(
        persist_directory=settings.chroma_dir,
        collection_name=settings.chroma_collection_name,
    )


@lru_cache
def get_embedder() -> HashingEmbedder:
    settings = get_settings()
    return HashingEmbedder(dimension=settings.embedding_dimension)


@lru_cache
def get_indexing_service() -> IndexingService:
    settings = get_settings()
    return IndexingService(
        loader=TextDocumentLoader(),
        chunker=SimpleTextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        allowed_extensions=settings.allowed_extensions,
    )


@lru_cache
def get_query_service() -> QueryService:
    retriever = Retriever(embedder=get_embedder(), vector_store=get_vector_store())
    generator = SimpleGroundedAnswerGenerator()
    return QueryService(retriever=retriever, generator=generator)
