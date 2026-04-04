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
    assert result.citations[0].document_title
    assert result.citations[0].file_name == "architecture.md"
    assert result.citations[0].file_type == "md"
    assert hasattr(result.citations[0], "section_header")
    assert hasattr(result.citations[0], "page_number")
    assert result.retrieved_chunks
    assert "score" in result.retrieved_chunks[0]
    assert "relevance" in result.retrieved_chunks[0]
    assert "document_title" in result.retrieved_chunks[0]
    assert "file_name" in result.retrieved_chunks[0]
    assert "file_type" in result.retrieved_chunks[0]


def test_query_api_supports_source_path_filter(
    indexing_service: IndexingService,
    query_service: QueryService,
    sample_data_dir: Path,
) -> None:
    ingest_documents(
        IngestRequest(directory=str(sample_data_dir)),
        indexing_service=indexing_service,
    )

    payload = query_documents(
        QueryRequest(
            query="Which database stores document embeddings?",
            top_k=3,
            source_path=str(sample_data_dir / "architecture.md"),
        ),
        query_service=query_service,
    )

    assert payload.result.citations
    assert all("architecture.md" in citation.source_path for citation in payload.result.citations)


def test_query_api_returns_grounded_fallback_for_weak_retrieval(
    indexing_service: IndexingService,
    query_service: QueryService,
    sample_data_dir: Path,
) -> None:
    ingest_documents(
        IngestRequest(directory=str(sample_data_dir)),
        indexing_service=indexing_service,
    )

    payload = query_documents(
        QueryRequest(query="What is the weather forecast tomorrow?", top_k=3),
        query_service=query_service,
    )

    assert payload.result.citations == []
    assert "could not find enough grounded support" in payload.result.answer_text.lower()
