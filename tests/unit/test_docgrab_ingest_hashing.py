from dataclasses import dataclass

from docgrab_ingest.hashing import (
    compute_chunk_hash_from_metadata,
    compute_chunk_id,
    compute_chunk_hash,
    compute_document_hash,
    normalize_content,
)


@dataclass(frozen=True)
class _AlternateChunkMetadata:
    source_type: str = "markdown"
    repo: str | None = None
    file_path: str = "docs/guide.md"
    heading_path: tuple[str, ...] = ()
    symbol_name: str | None = None
    content_hash: str = "sha256:" + "a" * 64
    parser_version: str = "markdown-parser-v1"
    chunker_version: str = "markdown-chunker-v1"
    occurrence_ordinal: int = 0


def test_normalize_content_canonicalizes_line_endings_only() -> None:
    assert normalize_content("  Alpha  \r\n\tBeta\t\r\n\n") == "  Alpha  \n\tBeta\t\n\n"


def test_document_hash_is_stable_for_equivalent_line_endings() -> None:
    assert compute_document_hash("Alpha\r\nBeta  \r") == compute_document_hash("Alpha\nBeta  \n")


def test_document_hash_preserves_markdown_significant_whitespace() -> None:
    assert compute_document_hash("Alpha\nBeta") != compute_document_hash("Alpha  \nBeta")
    assert compute_document_hash("code") != compute_document_hash("    code")


def test_chunk_hash_includes_retrieval_structural_metadata() -> None:
    base_hash = compute_chunk_hash(
        "Retry policies live here.",
        source_type="markdown",
        repo="docgrab",
        file_path="docs/ops.md",
        heading_path=("Operations", "Retries"),
        symbol_name=None,
        parser_version="markdown-parser-v1",
        chunker_version="markdown-chunker-v1",
    )

    moved_hash = compute_chunk_hash(
        "Retry policies live here.",
        source_type="markdown",
        repo="docgrab",
        file_path="docs/ops.md",
        heading_path=("Operations", "Timeouts"),
        symbol_name=None,
        parser_version="markdown-parser-v1",
        chunker_version="markdown-chunker-v1",
    )

    assert base_hash.startswith("sha256:")
    assert len(base_hash) == 71
    assert base_hash != moved_hash


def test_hashing_accepts_metadata_without_a_pydantic_dependency() -> None:
    metadata = _AlternateChunkMetadata()

    assert compute_chunk_hash_from_metadata("Content", metadata).startswith("sha256:")
    assert compute_chunk_id(metadata).startswith("sha256:")


def test_chunk_id_canonicalizes_protocol_metadata_file_paths() -> None:
    canonical = _AlternateChunkMetadata(file_path="docs/guide.md")
    equivalent = _AlternateChunkMetadata(file_path="docs/./guide.md")

    assert compute_chunk_id(canonical) == compute_chunk_id(equivalent)


def test_chunk_hash_canonicalizes_equivalent_file_paths() -> None:
    hashes = {
        compute_chunk_hash(
            "Stable content",
            source_type="markdown",
            file_path=file_path,
            parser_version="markdown-parser-v1",
            chunker_version="markdown-chunker-v1",
        )
        for file_path in (
            "docs/guide.md",
            "docs/./guide.md",
            "docs//guide.md",
            "docs/guide.md/",
        )
    }

    assert len(hashes) == 1


def test_chunk_hash_preserves_significant_trailing_whitespace() -> None:
    first_hash = compute_chunk_hash(
        "Same chunk text.",
        source_type="text",
        repo=None,
        file_path="notes.txt",
        heading_path=(),
        symbol_name=None,
        parser_version="plain-text-parser-v1",
        chunker_version="text-chunker-v1",
    )
    second_hash = compute_chunk_hash(
        "Same chunk text.  \n",
        source_type="text",
        repo=None,
        file_path="notes.txt",
        heading_path=(),
        symbol_name=None,
        parser_version="plain-text-parser-v1",
        chunker_version="text-chunker-v1",
    )

    assert first_hash != second_hash
