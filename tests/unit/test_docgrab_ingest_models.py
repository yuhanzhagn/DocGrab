import pytest
from pydantic import ValidationError

from docgrab_ingest.hashing import compute_chunk_hash, compute_document_hash
from docgrab_ingest.models import ChunkMetadata, IngestChunk, IngestDocument


def test_ingest_document_serializes_to_json_safe_contract() -> None:
    document_hash = compute_document_hash("# Architecture\n\nChroma stores vectors.")
    document = IngestDocument(
        source_type="markdown",
        repo="docgrab",
        file_path="docs/architecture.md",
        content="# Architecture\n\nChroma stores vectors.",
        document_hash=document_hash,
        parser_version="markdown-parser-v1",
    )

    assert document.model_dump(mode="json") == {
        "schema_version": "docgrab.ingest.document.v1",
        "source_type": "markdown",
        "repo": "docgrab",
        "file_path": "docs/architecture.md",
        "content": "# Architecture\n\nChroma stores vectors.",
        "document_hash": document_hash,
        "parser_version": "markdown-parser-v1",
        "metadata": {},
    }


def test_chunk_metadata_validates_offsets_and_serializes_heading_path() -> None:
    document_hash = compute_document_hash("Section text")
    content_hash = compute_chunk_hash(
        "Section text",
        source_type="markdown",
        repo=None,
        file_path="docs/guide.md",
        heading_path=("Guide", "Install"),
        symbol_name=None,
        parser_version="markdown-parser-v1",
        chunker_version="markdown-chunker-v1",
    )

    metadata = ChunkMetadata(
        source_type=" markdown ",
        repo=None,
        file_path=" docs/guide.md ",
        heading_path=(" Guide ", " Install "),
        symbol_name=None,
        chunk_index=0,
        content_hash=content_hash,
        document_hash=document_hash,
        embedding_model=" text-embedding-3-small ",
        parser_version="markdown-parser-v1",
        chunker_version="markdown-chunker-v1",
        start_char=0,
        end_char=12,
    )

    serialized = metadata.model_dump(mode="json")
    assert serialized["source_type"] == "markdown"
    assert serialized["file_path"] == "docs/guide.md"
    assert serialized["heading_path"] == ["Guide", "Install"]
    assert serialized["embedding_model"] == "text-embedding-3-small"


def test_chunk_metadata_rejects_invalid_hash_and_reversed_offsets() -> None:
    with pytest.raises(ValidationError):
        ChunkMetadata(
            source_type="text",
            repo=None,
            file_path="notes.txt",
            heading_path=(),
            symbol_name=None,
            chunk_index=0,
            content_hash="not-a-hash",
            document_hash=compute_document_hash("content"),
            embedding_model=None,
            parser_version="plain-text-parser-v1",
            chunker_version="text-chunker-v1",
            start_char=10,
            end_char=2,
        )


def test_ingest_chunk_requires_matching_document_hash() -> None:
    document_hash = compute_document_hash("chunk text")
    content_hash = compute_chunk_hash(
        "chunk text",
        source_type="text",
        repo=None,
        file_path="notes.txt",
        heading_path=(),
        symbol_name=None,
        parser_version="plain-text-parser-v1",
        chunker_version="text-chunker-v1",
    )
    metadata = ChunkMetadata(
        source_type="text",
        repo=None,
        file_path="notes.txt",
        heading_path=(),
        symbol_name=None,
        chunk_index=0,
        content_hash=content_hash,
        document_hash=document_hash,
        embedding_model=None,
        parser_version="plain-text-parser-v1",
        chunker_version="text-chunker-v1",
        start_char=0,
        end_char=10,
    )

    chunk = IngestChunk(
        chunk_id=content_hash,
        document_hash=document_hash,
        text="chunk text",
        metadata=metadata,
    )
    assert chunk.model_dump(mode="json")["metadata"]["content_hash"] == content_hash

    with pytest.raises(ValidationError):
        IngestChunk(
            chunk_id=content_hash,
            document_hash=compute_document_hash("other document"),
            text="chunk text",
            metadata=metadata,
        )

