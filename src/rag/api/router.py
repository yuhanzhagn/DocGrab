from fastapi import APIRouter

from rag.api.routes.health import router as health_router
from rag.api.routes.ingest import router as ingest_router
from rag.api.routes.query import router as query_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
api_router.include_router(ingest_router, prefix="/documents", tags=["documents"])
api_router.include_router(query_router, prefix="/query", tags=["query"])
