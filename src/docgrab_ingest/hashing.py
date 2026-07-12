from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from docgrab_ingest.models import ChunkMetadata

HASH_ALGORITHM = "sha256"
HASH_VERSION = "v1"


def normalize_content(content: str) -> str:
    """Normalize text before hashing without changing retrieval meaning."""
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip(" \t") for line in normalized.split("\n")]
    return "\n".join(lines).strip()


def compute_document_hash(content: str) -> str:
    return _hash_payload(
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
    return _hash_payload(
        {
            "hash_version": HASH_VERSION,
            "kind": "chunk",
            "content": normalize_content(content),
            "metadata": {
                "source_type": source_type,
                "repo": repo,
                "file_path": file_path,
                "heading_path": list(heading_path),
                "symbol_name": symbol_name,
                "parser_version": parser_version,
                "chunker_version": chunker_version,
            },
        }
    )


def compute_chunk_hash_from_metadata(content: str, metadata: ChunkMetadata) -> str:
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


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{HASH_ALGORITHM}:{hashlib.sha256(encoded).hexdigest()}"

