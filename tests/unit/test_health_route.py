from fastapi import HTTPException

from rag.api.routes.health import healthcheck, readiness_check
from rag.vectorstores.in_memory_store import InMemoryVectorStore


class _FailingVectorStore(InMemoryVectorStore):
    def check_health(self) -> None:
        raise RuntimeError("backend unavailable")


def test_healthcheck_returns_ok() -> None:
    assert healthcheck() == {"status": "ok"}


def test_readiness_check_returns_ready_for_healthy_store() -> None:
    result = readiness_check(vector_store=InMemoryVectorStore())

    assert result == {"status": "ready"}


def test_readiness_check_returns_503_for_unhealthy_store() -> None:
    try:
        readiness_check(vector_store=_FailingVectorStore())
    except HTTPException as exc:
        assert exc.status_code == 503
        assert "backend unavailable" in str(exc.detail)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected readiness_check to raise HTTPException")
