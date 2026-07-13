import pytest
from pydantic import ValidationError

from docgrab_ingest.hashing import compute_chunk_hash, compute_chunk_id, compute_document_hash
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
        metadata={
            "language": "en",
            "labels": ["architecture", "storage"],
            "published": True,
            "priority": 2,
            "score": 0.75,
            "extra": {"owner": None},
        },
    )

    assert document.model_dump(mode="json") == {
        "schema_version": "docgrab.ingest.document.v2",
        "hash_version": "v2",
        "source_type": "markdown",
        "repo": "docgrab",
        "file_path": "docs/architecture.md",
        "content": "# Architecture\n\nChroma stores vectors.",
        "document_hash": document_hash,
        "parser_version": "markdown-parser-v1",
        "metadata": {
            "language": "en",
            "labels": ["architecture", "storage"],
            "published": True,
            "priority": 2,
            "score": 0.75,
            "extra": {"owner": None},
        },
    }


def test_ingest_document_rejects_a_hash_for_different_content() -> None:
    with pytest.raises(ValidationError, match="document_hash must match content"):
        IngestDocument(
            source_type="markdown",
            repo=None,
            file_path="docs/architecture.md",
            content="# Updated architecture",
            document_hash=compute_document_hash("# Previous architecture"),
            parser_version="markdown-parser-v1",
        )


@pytest.mark.parametrize(
    "metadata",
    (
        {"labels": {"rag", "python"}},
        {"payload": b"bytes are not JSON"},
        {"opaque": object()},
        {1: "keys must be strings"},
    ),
)
def test_ingest_document_rejects_non_json_metadata(metadata: object) -> None:
    with pytest.raises(ValidationError):
        IngestDocument(
            source_type="markdown",
            repo=None,
            file_path="docs/architecture.md",
            content="# Architecture",
            document_hash=compute_document_hash("# Architecture"),
            parser_version="markdown-parser-v1",
            metadata=metadata,
        )


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_ingest_document_rejects_non_finite_json_numbers(value: float) -> None:
    with pytest.raises(ValidationError):
        IngestDocument(
            source_type="markdown",
            repo=None,
            file_path="docs/architecture.md",
            content="# Architecture",
            document_hash=compute_document_hash("# Architecture"),
            parser_version="markdown-parser-v1",
            metadata={"value": value},
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("schema_version", "docgrab.ingest.document.v1"),
        ("hash_version", "v1"),
    ),
)
def test_ingest_document_rejects_the_v1_persisted_contract(
    field_name: str, value: str
) -> None:
    document_data = {
        "source_type": "markdown",
        "repo": None,
        "file_path": "docs/architecture.md",
        "content": "# Architecture",
        "document_hash": compute_document_hash("# Architecture"),
        "parser_version": "markdown-parser-v1",
        field_name: value,
    }

    with pytest.raises(ValidationError):
        IngestDocument(**document_data)


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
        occurrence_ordinal=0,
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
    assert serialized["hash_version"] == "v2"
    assert serialized["occurrence_ordinal"] == 0


def test_chunk_metadata_rejects_invalid_hash_and_reversed_offsets() -> None:
    with pytest.raises(ValidationError):
        ChunkMetadata(
            source_type="text",
            repo=None,
            file_path="notes.txt",
            heading_path=(),
            symbol_name=None,
            chunk_index=0,
            occurrence_ordinal=0,
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
        occurrence_ordinal=0,
        content_hash=content_hash,
        document_hash=document_hash,
        embedding_model=None,
        parser_version="plain-text-parser-v1",
        chunker_version="text-chunker-v1",
        start_char=0,
        end_char=10,
    )

    chunk = IngestChunk(
        chunk_id=compute_chunk_id(metadata),
        document_hash=document_hash,
        text="chunk text",
        metadata=metadata,
    )
    assert chunk.model_dump(mode="json")["metadata"]["content_hash"] == content_hash

    with pytest.raises(ValidationError):
        IngestChunk(
            chunk_id=compute_chunk_id(metadata),
            document_hash=compute_document_hash("other document"),
            text="chunk text",
            metadata=metadata,
        )


def test_ingest_chunk_rejects_mismatched_content_hash_and_chunk_id() -> None:
    document_hash = compute_document_hash("chunk text")
    metadata = ChunkMetadata(
        source_type="text",
        repo=None,
        file_path="notes.txt",
        heading_path=(),
        symbol_name=None,
        chunk_index=0,
        occurrence_ordinal=0,
        content_hash=compute_chunk_hash(
            "chunk text",
            source_type="text",
            file_path="notes.txt",
            parser_version="plain-text-parser-v1",
            chunker_version="text-chunker-v1",
        ),
        document_hash=document_hash,
        embedding_model=None,
        parser_version="plain-text-parser-v1",
        chunker_version="text-chunker-v1",
        start_char=0,
        end_char=10,
    )

    with pytest.raises(ValidationError, match="content_hash must match"):
        IngestChunk(
            chunk_id=compute_chunk_id(metadata),
            document_hash=document_hash,
            text="changed text",
            metadata=metadata,
        )
    with pytest.raises(ValidationError, match="chunk_id must match"):
        IngestChunk(
            chunk_id=f"sha256:{'0' * 64}",
            document_hash=document_hash,
            text="chunk text",
            metadata=metadata,
        )


def test_duplicate_chunk_content_has_distinct_occurrence_ids() -> None:
    document_hash = compute_document_hash("chunk text\nchunk text")
    content_hash = compute_chunk_hash(
        "chunk text",
        source_type="text",
        file_path="notes.txt",
        parser_version="plain-text-parser-v1",
        chunker_version="text-chunker-v1",
    )
    first_metadata = ChunkMetadata(
        source_type="text",
        repo=None,
        file_path="notes.txt",
        heading_path=(),
        symbol_name=None,
        chunk_index=0,
        occurrence_ordinal=0,
        content_hash=content_hash,
        document_hash=document_hash,
        embedding_model=None,
        parser_version="plain-text-parser-v1",
        chunker_version="text-chunker-v1",
        start_char=0,
        end_char=10,
    )
    second_metadata = first_metadata.model_copy(
        update={"chunk_index": 1, "occurrence_ordinal": 1, "start_char": 11, "end_char": 21}
    )

    first = IngestChunk(
        chunk_id=compute_chunk_id(first_metadata),
        document_hash=document_hash,
        text="chunk text",
        metadata=first_metadata,
    )
    second = IngestChunk(
        chunk_id=compute_chunk_id(second_metadata),
        document_hash=document_hash,
        text="chunk text",
        metadata=second_metadata,
    )

    assert first.metadata.content_hash == second.metadata.content_hash
    assert first.chunk_id != second.chunk_id


def test_chunk_id_is_stable_when_unrelated_content_precedes_a_chunk() -> None:
    content_hash = compute_chunk_hash(
        "unchanged chunk",
        source_type="text",
        file_path="notes.txt",
        parser_version="plain-text-parser-v1",
        chunker_version="text-chunker-v1",
    )
    before_edit = ChunkMetadata(
        source_type="text",
        repo=None,
        file_path="notes.txt",
        heading_path=(),
        symbol_name=None,
        chunk_index=0,
        occurrence_ordinal=0,
        content_hash=content_hash,
        document_hash=compute_document_hash("unchanged chunk"),
        embedding_model=None,
        parser_version="plain-text-parser-v1",
        chunker_version="text-chunker-v1",
        start_char=0,
        end_char=15,
    )
    after_earlier_insert = before_edit.model_copy(
        update={
            "document_hash": compute_document_hash("new earlier content\nunchanged chunk"),
            "chunk_index": 1,
            "start_char": 20,
            "end_char": 35,
        }
    )

    assert before_edit.content_hash == after_earlier_insert.content_hash
    assert compute_chunk_id(before_edit) == compute_chunk_id(after_earlier_insert)


@pytest.mark.parametrize(
    "file_path",
    ("/absolute.md", "../outside.md", r"C:\\secret.md", "C:/secret.md"),
)
def test_persisted_models_reject_unstable_file_paths(file_path: str) -> None:
    document_hash = compute_document_hash("content")

    with pytest.raises(ValidationError, match="relative path"):
        IngestDocument(
            source_type="text",
            repo=None,
            file_path=file_path,
            content="content",
            document_hash=document_hash,
            parser_version="plain-text-parser-v1",
        )
    with pytest.raises(ValidationError, match="relative path"):
        ChunkMetadata(
            source_type="text",
            repo=None,
            file_path=file_path,
            heading_path=(),
            symbol_name=None,
            chunk_index=0,
            occurrence_ordinal=0,
            content_hash=compute_chunk_hash(
                "content",
                source_type="text",
                file_path="notes.txt",
                parser_version="plain-text-parser-v1",
                chunker_version="text-chunker-v1",
            ),
            document_hash=document_hash,
            embedding_model=None,
            parser_version="plain-text-parser-v1",
            chunker_version="text-chunker-v1",
            start_char=0,
            end_char=7,
        )


@pytest.mark.parametrize(
    "file_path",
    ("docs/file.md", "docs/./file.md", "docs//file.md", "docs/file.md/"),
)
def test_persisted_models_canonicalize_equivalent_file_paths(file_path: str) -> None:
    document = IngestDocument(
        source_type="markdown",
        repo=None,
        file_path=file_path,
        content="# Document",
        document_hash=compute_document_hash("# Document"),
        parser_version="markdown-parser-v1",
    )

    assert document.file_path == "docs/file.md"
