from docgrab_ingest.hashing import (
    compute_chunk_hash,
    compute_document_hash,
    normalize_content,
)


def test_normalize_content_stabilizes_line_endings_and_trailing_space() -> None:
    assert normalize_content("  Alpha \r\nBeta\t\n\n") == "Alpha\nBeta"


def test_document_hash_is_stable_for_normalized_content() -> None:
    assert compute_document_hash("Alpha\r\nBeta  \n") == compute_document_hash("Alpha\nBeta")


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


def test_chunk_hash_ignores_offsets_and_chunk_index_by_api_shape() -> None:
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
        "Same chunk text.\n",
        source_type="text",
        repo=None,
        file_path="notes.txt",
        heading_path=(),
        symbol_name=None,
        parser_version="plain-text-parser-v1",
        chunker_version="text-chunker-v1",
    )

    assert first_hash == second_hash

