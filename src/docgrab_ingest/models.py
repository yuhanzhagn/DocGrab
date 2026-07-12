from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DOCUMENT_SCHEMA_VERSION = "docgrab.ingest.document.v1"
CHUNK_SCHEMA_VERSION = "docgrab.ingest.chunk.v1"
HASH_PATTERN = r"^sha256:[0-9a-f]{64}$"


class ChunkMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_type: str = Field(min_length=1)
    repo: str | None = None
    file_path: str = Field(min_length=1)
    heading_path: tuple[str, ...] = Field(default_factory=tuple)
    symbol_name: str | None = None
    chunk_index: int = Field(ge=0)
    content_hash: str = Field(pattern=HASH_PATTERN)
    document_hash: str = Field(pattern=HASH_PATTERN)
    embedding_model: str | None = None
    parser_version: str = Field(min_length=1)
    chunker_version: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)

    @field_validator(
        "source_type",
        "repo",
        "file_path",
        "symbol_name",
        "embedding_model",
        "parser_version",
        "chunker_version",
    )
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped

    @field_validator("heading_path")
    @classmethod
    def validate_heading_path(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(part.strip() for part in value)
        if any(not part for part in normalized):
            raise ValueError("heading_path entries must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_offsets(self) -> "ChunkMetadata":
        if self.end_char < self.start_char:
            raise ValueError("end_char must be greater than or equal to start_char")
        return self


class IngestDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Literal["docgrab.ingest.document.v1"] = DOCUMENT_SCHEMA_VERSION
    source_type: str = Field(min_length=1)
    repo: str | None = None
    file_path: str = Field(min_length=1)
    content: str
    document_hash: str = Field(pattern=HASH_PATTERN)
    parser_version: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_type", "repo", "file_path", "parser_version")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped


class IngestChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Literal["docgrab.ingest.chunk.v1"] = CHUNK_SCHEMA_VERSION
    chunk_id: str = Field(min_length=1)
    document_hash: str = Field(pattern=HASH_PATTERN)
    text: str = Field(min_length=1)
    metadata: ChunkMetadata

    @field_validator("chunk_id")
    @classmethod
    def strip_chunk_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("chunk_id must not be blank")
        return stripped

    @model_validator(mode="after")
    def validate_document_hash(self) -> "IngestChunk":
        if self.document_hash != self.metadata.document_hash:
            raise ValueError("document_hash must match metadata.document_hash")
        return self

