from pathlib import Path

import httpx
import pytest

from rag.api.deps import get_indexing_service, get_query_service
from rag.chunkers.simple import SimpleTextChunker
from rag.embeddings.hash_embedder import HashingEmbedder
from rag.generation.simple import SimpleGroundedAnswerGenerator
from rag.loaders.text_loader import TextDocumentLoader
from rag.main import create_app
from rag.retrieval.retriever import Retriever
from rag.services.indexing_service import IndexingService
from rag.services.query_service import QueryService
from rag.vectorstores.in_memory_store import InMemoryVectorStore


@pytest.fixture
def sample_data_dir() -> Path:
    return Path("data/sample_docs").resolve()


@pytest.fixture
def temp_vector_store(tmp_path: Path):
    from uuid import uuid4

    from rag.vectorstores.chroma_store import ChromaVectorStore

    return ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name=f"test-{uuid4().hex}",
    )


@pytest.fixture
def test_vector_store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dimension=64)


@pytest.fixture
def indexing_service(
    embedder: HashingEmbedder,
    test_vector_store: InMemoryVectorStore,
) -> IndexingService:
    return IndexingService(
        loader=TextDocumentLoader(),
        chunker=SimpleTextChunker(chunk_size=120, chunk_overlap=20),
        embedder=embedder,
        vector_store=test_vector_store,
        allowed_extensions=(".md", ".txt"),
    )


@pytest.fixture
def query_service(
    embedder: HashingEmbedder,
    test_vector_store: InMemoryVectorStore,
) -> QueryService:
    retriever = Retriever(embedder=embedder, vector_store=test_vector_store)
    generator = SimpleGroundedAnswerGenerator()
    return QueryService(retriever=retriever, generator=generator)


@pytest.fixture
async def api_client(
    indexing_service: IndexingService,
    query_service: QueryService,
) -> httpx.AsyncClient:
    app = create_app()
    app.dependency_overrides[get_indexing_service] = lambda: indexing_service
    app.dependency_overrides[get_query_service] = lambda: query_service

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client

    app.dependency_overrides.clear()
