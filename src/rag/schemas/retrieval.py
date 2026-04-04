from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    source_path: str | None = None
    document_path: str | None = None

    @property
    def path_filter(self) -> str | None:
        return self.source_path or self.document_path


class RetrievalResult(BaseModel):
    chunk_id: str
    document_id: str
    source_path: str
    text: str
    score: float
    distance: float | None = None
    relevance: str = "low"
    metadata: dict[str, Any] = Field(default_factory=dict)
