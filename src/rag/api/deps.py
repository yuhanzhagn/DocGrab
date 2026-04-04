from functools import lru_cache
from typing import TYPE_CHECKING

from rag.chunkers.simple import SimpleTextChunker
from rag.config.settings import get_settings
from rag.embeddings.base import Embedder
from rag.embeddings.factory import create_embedder
from rag.generation.base import AnswerGenerator
from rag.generation.factory import create_generator
from rag.loaders.multi_loader import MultiDocumentLoader
from rag.loaders.pdf_loader import PDFDocumentLoader
from rag.loaders.text_loader import TextDocumentLoader
from rag.retrieval.retriever import Retriever
from rag.services.indexing_service import IndexingService
from rag.services.query_service import QueryService

if TYPE_CHECKING:
    from rag.vectorstores.chroma_store import ChromaVectorStore


@lru_cache
def get_vector_store() -> "ChromaVectorStore":
    settings = get_settings()
    from rag.vectorstores.chroma_store import ChromaVectorStore

    return ChromaVectorStore(
        persist_directory=settings.chroma_dir,
        collection_name=settings.chroma_collection_name,
    )


@lru_cache
def get_embedder() -> Embedder:
    settings = get_settings()
    return create_embedder(settings)


@lru_cache
def get_indexing_service() -> IndexingService:
    settings = get_settings()
    return IndexingService(
        loader=MultiDocumentLoader(
            loaders=[
                TextDocumentLoader(),
                PDFDocumentLoader(),
            ]
        ),
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
    settings = get_settings()
    generator = create_generator(settings)
    return QueryService(retriever=retriever, generator=generator)
