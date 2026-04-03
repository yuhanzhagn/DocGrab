from pathlib import Path

from rag.api.routes.ingest import ingest_documents
from rag.api.routes.query import query_documents
from rag.schemas.api import IngestRequest
from rag.schemas.retrieval import QueryRequest
from rag.services.indexing_service import IndexingService
from rag.services.query_service import QueryService


def test_ingest_and_query_api_returns_citations(
    indexing_service: IndexingService,
    query_service: QueryService,
    sample_data_dir: Path,
) -> None:
    ingest_payload = ingest_documents(
        IngestRequest(directory=str(sample_data_dir)),
        indexing_service=indexing_service,
    )

    assert ingest_payload.indexed_documents >= 2
    assert ingest_payload.indexed_chunks >= 2

    payload = query_documents(
        QueryRequest(query="Which database stores document embeddings?", top_k=3),
        query_service=query_service,
    )

    result = payload.result

    assert result.citations
    assert any("architecture.md" in citation.source_path for citation in result.citations)
    assert "Chroma" in result.answer_text
