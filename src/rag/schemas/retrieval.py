from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievalResult(BaseModel):
    chunk_id: str
    document_id: str
    source_path: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
