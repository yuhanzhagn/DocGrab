from fastapi import APIRouter, Depends, HTTPException, status

from rag.api.deps import get_vector_store
from rag.vectorstores.base import VectorStore


router = APIRouter()


@router.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def readiness_check(
    vector_store: VectorStore = Depends(get_vector_store),
) -> dict[str, str]:
    try:
        vector_store.check_health()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector store is not ready: {exc}",
        ) from exc
    return {"status": "ready"}
