from fastapi import APIRouter, Depends

from rag.api.deps import get_query_service
from rag.schemas.api import QueryResponse
from rag.schemas.retrieval import QueryRequest
from rag.services.query_service import QueryService


router = APIRouter()


@router.post("/", response_model=QueryResponse)
def query_documents(
    request: QueryRequest,
    query_service: QueryService = Depends(get_query_service),
) -> QueryResponse:
    result = query_service.answer(query=request.query, top_k=request.top_k)
    return QueryResponse(result=result)
