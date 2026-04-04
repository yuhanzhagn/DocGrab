from pathlib import Path

from rag.schemas.api import IngestResponse
from rag.use_cases import answer_query, ingest_directory


def test_ingest_directory_use_case_returns_ingest_response(
    indexing_service,
    sample_data_dir: Path,
) -> None:
    result = ingest_directory(indexing_service=indexing_service, directory=str(sample_data_dir))

    assert isinstance(result, IngestResponse)
    assert result.indexed_documents >= 2
    assert result.indexed_chunks >= 2


def test_answer_query_use_case_supports_document_path_alias(
    indexing_service,
    query_service,
    sample_data_dir: Path,
) -> None:
    ingest_directory(indexing_service=indexing_service, directory=str(sample_data_dir))

    result = answer_query(
        query_service=query_service,
        query="Which database stores document embeddings?",
        top_k=3,
        document_path=str(sample_data_dir / "architecture.md"),
    )

    assert result.citations
    assert all("architecture.md" in citation.source_path for citation in result.citations)
