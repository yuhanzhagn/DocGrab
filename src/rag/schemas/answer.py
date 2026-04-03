from typing import Any

from pydantic import BaseModel, Field


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    source_path: str
    chunk_index: int
    start_char: int
    end_char: int
    snippet: str


class FinalAnswer(BaseModel):
    answer_text: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
