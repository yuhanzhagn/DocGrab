from pydantic import BaseModel, Field

from rag.schemas.answer import FinalAnswer


class IngestRequest(BaseModel):
    directory: str


class IngestResponse(BaseModel):
    indexed_documents: int
    indexed_chunks: int
    skipped_files: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    result: FinalAnswer
