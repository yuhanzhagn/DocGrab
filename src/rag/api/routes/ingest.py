from fastapi import APIRouter, Depends, HTTPException

from rag.api.deps import get_indexing_service
from rag.schemas.api import IngestRequest, IngestResponse
from rag.services.indexing_service import IndexingService


router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
def ingest_documents(
    request: IngestRequest,
    indexing_service: IndexingService = Depends(get_indexing_service),
) -> IngestResponse:
    try:
        return indexing_service.ingest_directory(request.directory)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
