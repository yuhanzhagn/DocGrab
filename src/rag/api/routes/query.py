from fastapi import APIRouter, Depends

from rag.api.deps import get_query_service
from rag.schemas.api import QueryResponse
from rag.schemas.retrieval import QueryRequest
from rag.services.query_service import QueryService
from rag.use_cases import answer_query


router = APIRouter()


@router.post("/", response_model=QueryResponse)
def query_documents(
    request: QueryRequest,
    query_service: QueryService = Depends(get_query_service),
) -> QueryResponse:
    result = answer_query(
        query_service=query_service,
        query=request.query,
        top_k=request.top_k,
        source_path=request.source_path,
        document_path=request.document_path,
    )
    return QueryResponse(result=result)
