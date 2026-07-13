from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from docgrab_ingest.hash_primitives import HASH_VERSION
from docgrab_ingest.hashing import (
    compute_chunk_hash_from_metadata,
    compute_chunk_id,
    compute_document_hash,
)
from docgrab_ingest.paths import normalize_relative_file_path

DOCUMENT_SCHEMA_VERSION = "docgrab.ingest.document.v2"
CHUNK_SCHEMA_VERSION = "docgrab.ingest.chunk.v2"
HASH_PATTERN = r"^sha256:[0-9a-f]{64}$"


class ChunkMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    hash_version: Literal["v2"] = HASH_VERSION
    source_type: str = Field(min_length=1)
    repo: str | None = None
    file_path: str = Field(min_length=1)
    heading_path: tuple[str, ...] = Field(default_factory=tuple)
    symbol_name: str | None = None
    chunk_index: int = Field(ge=0)
    occurrence_ordinal: int = Field(ge=0)
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

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, value: str) -> str:
        return normalize_relative_file_path(value)

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
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    schema_version: Literal["docgrab.ingest.document.v2"] = DOCUMENT_SCHEMA_VERSION
    hash_version: Literal["v2"] = HASH_VERSION
    source_type: str = Field(min_length=1)
    repo: str | None = None
    file_path: str = Field(min_length=1)
    content: str
    document_hash: str = Field(pattern=HASH_PATTERN)
    parser_version: str = Field(min_length=1)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("source_type", "repo", "parser_version")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, value: str) -> str:
        return normalize_relative_file_path(value)

    @model_validator(mode="after")
    def validate_document_hash(self) -> "IngestDocument":
        if self.document_hash != compute_document_hash(self.content):
            raise ValueError("document_hash must match content")
        return self


class IngestChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Literal["docgrab.ingest.chunk.v2"] = CHUNK_SCHEMA_VERSION
    hash_version: Literal["v2"] = HASH_VERSION
    chunk_id: str = Field(min_length=1, pattern=HASH_PATTERN)
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
    def validate_hashes_and_chunk_id(self) -> "IngestChunk":
        if self.document_hash != self.metadata.document_hash:
            raise ValueError("document_hash must match metadata.document_hash")
        if self.hash_version != self.metadata.hash_version:
            raise ValueError("hash_version must match metadata.hash_version")
        if self.metadata.content_hash != compute_chunk_hash_from_metadata(self.text, self.metadata):
            raise ValueError("metadata.content_hash must match text and metadata")
        if self.chunk_id != compute_chunk_id(self.metadata):
            raise ValueError("chunk_id must match metadata occurrence identity")
        return self
