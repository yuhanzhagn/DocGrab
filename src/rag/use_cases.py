from rag.schemas.answer import FinalAnswer
from rag.schemas.api import IngestResponse
from rag.services.indexing_service import IndexingService
from rag.services.query_service import QueryService


def ingest_directory(indexing_service: IndexingService, directory: str) -> IngestResponse:
    return indexing_service.ingest_directory(directory)


def answer_query(
    query_service: QueryService,
    query: str,
    *,
    top_k: int,
    source_path: str | None = None,
    document_path: str | None = None,
) -> FinalAnswer:
    path_filter = source_path or document_path
    return query_service.answer(
        query=query,
        top_k=top_k,
        path_filter=path_filter,
    )
