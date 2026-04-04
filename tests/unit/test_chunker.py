from rag.chunkers.simple import SimpleTextChunker
from rag.schemas.document import RawDocument


def test_simple_text_chunker_produces_multiple_chunks() -> None:
    chunker = SimpleTextChunker(chunk_size=80, chunk_overlap=10)
    document = RawDocument(
        document_id="doc-1",
        source_path="sample.txt",
        file_name="sample.txt",
        file_extension=".txt",
        content=(
            "Chunking should preserve source metadata.\n\n"
            "This text is long enough to require multiple chunks when the chunk size "
            "is intentionally small for testing.\n\n"
            "The final chunk should still have a stable chunk id."
        ),
        metadata={},
    )

    chunks = chunker.chunk(document)

    assert len(chunks) >= 2
    assert chunks[0].document_id == "doc-1"
    assert chunks[0].metadata["source_path"] == "sample.txt"
    assert chunks[0].metadata["document_title"] == "sample.txt"
    assert chunks[0].metadata["file_name"] == "sample.txt"
    assert chunks[0].metadata["file_type"] == "txt"
    assert "section_header" in chunks[0].metadata
    assert "page_number" in chunks[0].metadata
    assert chunks[0].start_char == 0
    assert chunks[0].end_char > chunks[0].start_char
    assert chunks[1].start_char < chunks[1].end_char


def test_simple_text_chunker_captures_markdown_section_header() -> None:
    chunker = SimpleTextChunker(chunk_size=60, chunk_overlap=10)
    document = RawDocument(
        document_id="doc-md",
        source_path="guide.md",
        file_name="guide.md",
        file_extension=".md",
        content=(
            "# Overview\n\n"
            "This introduction explains the architecture and enough text to form a chunk.\n\n"
            "## Retrieval\n\n"
            "This section describes chunk retrieval and citations in more detail."
        ),
        metadata={
            "document_title": "Overview",
            "file_type": "md",
            "page_number": None,
        },
    )

    chunks = chunker.chunk(document)

    assert chunks[0].metadata["section_header"] == "Overview"
    assert any(chunk.metadata["section_header"] == "Retrieval" for chunk in chunks[1:])


def test_simple_text_chunker_assigns_page_number_from_pdf_spans() -> None:
    chunker = SimpleTextChunker(chunk_size=30, chunk_overlap=0)
    document = RawDocument(
        document_id="doc-pdf",
        source_path="manual.pdf",
        file_name="manual.pdf",
        file_extension=".pdf",
        content="First page text here.\n\nSecond page text here.",
        metadata={
            "document_title": "manual",
            "file_type": "pdf",
            "page_spans": [
                {"page_number": 1, "start_char": 0, "end_char": 20},
                {"page_number": 2, "start_char": 22, "end_char": 44},
            ],
        },
    )

    chunks = chunker.chunk(document)

    assert chunks[0].metadata["page_number"] == 1
    assert any(chunk.metadata["page_number"] == 2 for chunk in chunks[1:])


def test_simple_text_chunker_returns_empty_for_blank_content() -> None:
    chunker = SimpleTextChunker(chunk_size=50, chunk_overlap=10)
    document = RawDocument(
        document_id="doc-blank",
        source_path="blank.txt",
        file_name="blank.txt",
        file_extension=".txt",
        content="   \n\n   ",
        metadata={},
    )

    assert chunker.chunk(document) == []


def test_simple_text_chunker_makes_progress_when_newline_falls_inside_overlap() -> None:
    chunker = SimpleTextChunker(chunk_size=10, chunk_overlap=4)
    document = RawDocument(
        document_id="doc-overlap",
        source_path="overlap.txt",
        file_name="overlap.txt",
        file_extension=".txt",
        content="abcd\nefghijklmnop",
        metadata={},
    )

    chunks = chunker.chunk(document)

    assert len(chunks) >= 2
    assert all(
        chunks[index].start_char < chunks[index + 1].start_char
        for index in range(len(chunks) - 1)
    )
