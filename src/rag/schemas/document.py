from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class RawDocument(BaseModel):
    document_id: str
    source_path: str
    file_name: str
    file_extension: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_path(
        cls,
        document_id: str,
        path: Path,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> "RawDocument":
        return cls(
            document_id=document_id,
            source_path=str(path),
            file_name=path.name,
            file_extension=path.suffix.lower(),
            content=content,
            metadata=metadata or {},
        )


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexedRecord(BaseModel):
    chunk_id: str
    document_id: str
    source_path: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)
