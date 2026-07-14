from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from docgrab_ingest.hash_primitives import (
    HASH_VERSION,
    hash_payload,
    normalize_content as normalize_hash_content,
)
from docgrab_ingest.paths import normalize_relative_file_path


class ChunkHashMetadata(Protocol):
    source_type: str
    repo: str | None
    file_path: str
    heading_path: Sequence[str]
    symbol_name: str | None
    content_hash: str
    parser_version: str
    chunker_version: str
    occurrence_ordinal: int


def normalize_content(content: str) -> str:
    return normalize_hash_content(content)


def compute_document_hash(content: str) -> str:
    return hash_payload(
        {
            "hash_version": HASH_VERSION,
            "kind": "document",
            "content": normalize_content(content),
        }
    )


def compute_chunk_hash(
    content: str,
    *,
    source_type: str,
    repo: str | None = None,
    file_path: str,
    heading_path: Sequence[str] = (),
    symbol_name: str | None = None,
    parser_version: str,
    chunker_version: str,
) -> str:
    return hash_payload(
        {
            "hash_version": HASH_VERSION,
            "kind": "chunk",
            "content": normalize_content(content),
            "metadata": {
                "source_type": source_type,
                "repo": repo,
                "file_path": normalize_relative_file_path(file_path),
                "heading_path": list(heading_path),
                "symbol_name": symbol_name,
                "parser_version": parser_version,
                "chunker_version": chunker_version,
            },
        }
    )


def compute_chunk_hash_from_metadata(content: str, metadata: ChunkHashMetadata) -> str:
    return compute_chunk_hash(
        content,
        source_type=metadata.source_type,
        repo=metadata.repo,
        file_path=metadata.file_path,
        heading_path=metadata.heading_path,
        symbol_name=metadata.symbol_name,
        parser_version=metadata.parser_version,
        chunker_version=metadata.chunker_version,
    )


def compute_chunk_id(metadata: ChunkHashMetadata) -> str:
    """Build a deterministic occurrence ID distinct from a chunk content hash."""
    return hash_payload(
        {
            "hash_version": HASH_VERSION,
            "kind": "chunk-id",
            "source": {
                "source_type": metadata.source_type,
                "repo": metadata.repo,
                "file_path": normalize_relative_file_path(metadata.file_path),
            },
            "content_hash": metadata.content_hash,
            "occurrence_ordinal": metadata.occurrence_ordinal,
        }
    )
