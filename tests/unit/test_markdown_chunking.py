import pytest

from docgrab_ingest.chunking.markdown import MARKDOWN_CHUNKER_VERSION, MarkdownChunker
from docgrab_ingest.hashing import compute_document_hash
from docgrab_ingest.models import IngestDocument
from docgrab_ingest.parsers.markdown import (
    MARKDOWN_PARSER_VERSION,
    MarkdownParser,
    MarkdownSection,
    ParsedMarkdownDocument,
)
from docgrab_ingest.sources.base import SourceItem


def _document(content: str, file_path: str = "docs/guide.md") -> IngestDocument:
    return IngestDocument(
        source_type="markdown",
        repo="docgrab",
        file_path=file_path,
        content=content,
        document_hash=compute_document_hash(content),
        parser_version=MARKDOWN_PARSER_VERSION,
    )


def test_markdown_parser_preserves_heading_paths_and_ignores_fenced_headings() -> None:
    content = (
        "Preamble.\n"
        "# Guide #\n"
        "Overview.\n"
        "## Install\n"
        "```markdown\n"
        "# Not a heading\n"
        "```\n"
        "Steps.\n"
        "### Linux\n"
        "Run it.\n"
        "# API\n"
        "Reference.\n"
    )

    parsed_document = MarkdownParser().parse(_document(content))
    sections = parsed_document.sections

    assert [section.heading_path for section in sections] == [
        (),
        ("Guide",),
        ("Guide", "Install"),
        ("Guide", "Install", "Linux"),
        ("API",),
    ]
    assert [section.text for section in sections] == [
        "Preamble.\n",
        "Overview.\n",
        "```markdown\n# Not a heading\n```\nSteps.\n",
        "Run it.\n",
        "Reference.\n",
    ]
    assert all(content[section.start_char : section.end_char] == section.text for section in sections)


def test_markdown_chunker_preserves_natural_boundaries_and_source_offsets() -> None:
    content = (
        "# Guide\n"
        "First paragraph.\n\n"
        "Second paragraph.\n\n"
        "Third paragraph.\n"
    )
    document = _document(content)
    parsed_document = MarkdownParser().parse(document)

    chunks = MarkdownChunker(chunk_size=20).chunk(parsed_document)

    assert [chunk.text for chunk in chunks] == [
        "First paragraph.\n\n",
        "Second paragraph.\n\n",
        "Third paragraph.\n",
    ]
    assert [chunk.metadata.heading_path for chunk in chunks] == [("Guide",)] * 3
    assert [chunk.metadata.chunk_index for chunk in chunks] == [0, 1, 2]
    assert all(chunk.metadata.chunker_version == MARKDOWN_CHUNKER_VERSION for chunk in chunks)
    assert all(
        content[chunk.metadata.start_char : chunk.metadata.end_char] == chunk.text for chunk in chunks
    )
    assert all(len(chunk.text) <= 20 for chunk in chunks)


def test_unchanged_markdown_chunk_keeps_its_identity_after_earlier_content_changes() -> None:
    original_content = "# Introduction\nIntroductory text.\n# Stable\nPreserve this chunk.\n"
    updated_content = (
        "# Earlier\nUnrelated material.\n"
        "# Introduction\nIntroductory text.\n"
        "# Stable\nPreserve this chunk.\n"
    )

    parser = MarkdownParser()
    chunker = MarkdownChunker(chunk_size=200)
    original_document = _document(original_content)
    updated_document = _document(updated_content)
    original_chunks = chunker.chunk(parser.parse(original_document))
    updated_chunks = chunker.chunk(parser.parse(updated_document))
    original_stable = next(chunk for chunk in original_chunks if chunk.text == "Preserve this chunk.\n")
    updated_stable = next(chunk for chunk in updated_chunks if chunk.text == "Preserve this chunk.\n")

    assert original_stable.metadata.content_hash == updated_stable.metadata.content_hash
    assert original_stable.chunk_id == updated_stable.chunk_id
    assert original_stable.document_hash != updated_stable.document_hash
    assert original_stable.metadata.start_char != updated_stable.metadata.start_char


def test_markdown_chunker_rejects_a_nonpositive_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        MarkdownChunker(chunk_size=0)


def test_markdown_parser_decodes_source_bytes_into_a_bound_document() -> None:
    source = SourceItem(
        source_type="local_file",
        source_uri="file:///workspace/docs/guide.md",
        file_path="docs/guide.md",
        repo="docgrab",
    )

    parsed_document = MarkdownParser().parse_bytes(source, b"# Guide\nSource bytes.\n")

    assert parsed_document.document.source_type == "local_file"
    assert parsed_document.document.file_path == "docs/guide.md"
    assert parsed_document.sections[0].heading_path == ("Guide",)
    assert parsed_document.sections[0].text == "Source bytes.\n"


def test_markdown_parser_rejects_non_utf8_source_bytes() -> None:
    source = SourceItem(
        source_type="local_file",
        source_uri="file:///workspace/docs/guide.md",
        file_path="docs/guide.md",
    )

    with pytest.raises(ValueError, match="Markdown source must be valid UTF-8"):
        MarkdownParser().parse_bytes(source, b"\xff")


@pytest.mark.parametrize("empty_heading", ("#\n", "#    \n"))
def test_empty_atx_heading_updates_hierarchy_without_persisting_a_blank_path(
    empty_heading: str,
) -> None:
    content = "# Parent\nParent text.\n## Child\nChild text.\n" + empty_heading + "After reset.\n"
    parsed_document = MarkdownParser().parse(_document(content))

    assert [section.heading_path for section in parsed_document.sections] == [
        ("Parent",),
        ("Parent", "Child"),
        (),
    ]
    assert [section.text for section in parsed_document.sections] == [
        "Parent text.\n",
        "Child text.\n",
        "After reset.\n",
    ]
    assert [chunk.metadata.heading_path for chunk in MarkdownChunker().chunk(parsed_document)] == [
        ("Parent",),
        ("Parent", "Child"),
        (),
    ]


def test_indented_atx_headings_preserve_structure_but_four_space_content_remains_code() -> None:
    content = " # Guide\nOverview.\n   ## Details\nDetail.\n    # Not a heading\n"

    sections = MarkdownParser().parse(_document(content)).sections

    assert [section.heading_path for section in sections] == [("Guide",), ("Guide", "Details")]
    assert [section.text for section in sections] == [
        "Overview.\n",
        "Detail.\n    # Not a heading\n",
    ]


def test_bom_prefixed_source_bytes_match_plain_utf8_heading_offsets_and_hashes() -> None:
    source = SourceItem(
        source_type="local_file",
        source_uri="file:///workspace/docs/guide.md",
        file_path="docs/guide.md",
        repo="docgrab",
    )
    plain_content = b"# Guide\nSource bytes.\n"
    parser = MarkdownParser()
    chunker = MarkdownChunker()
    plain = parser.parse_bytes(source, plain_content)
    bom_prefixed = parser.parse_bytes(source, b"\xef\xbb\xbf" + plain_content)

    plain_chunks = chunker.chunk(plain)
    bom_chunks = chunker.chunk(bom_prefixed)

    assert bom_prefixed.document.content == plain.document.content
    assert bom_prefixed.document.document_hash == plain.document.document_hash
    assert [section.heading_path for section in bom_prefixed.sections] == [("Guide",)]
    assert [(chunk.metadata.start_char, chunk.metadata.end_char, chunk.chunk_id) for chunk in bom_chunks] == [
        (chunk.metadata.start_char, chunk.metadata.end_char, chunk.chunk_id) for chunk in plain_chunks
    ]


def test_parsed_markdown_document_rejects_sections_from_another_document() -> None:
    parser = MarkdownParser()
    alpha = parser.parse(_document("# Alpha\nAlpha text.\n"))
    same_content_elsewhere = _document("# Alpha\nAlpha text.\n", file_path="docs/other.md")

    with pytest.raises(ValueError, match="section identity must match the source document"):
        ParsedMarkdownDocument(document=same_content_elsewhere, sections=alpha.sections)


def test_parsed_markdown_document_rejects_a_section_with_an_invalid_source_slice() -> None:
    alpha = MarkdownParser().parse(_document("# Alpha\nAlpha text.\n"))
    beta = _document("# Delta\nOther text.\n")
    invalid_section = MarkdownSection(
        document_identity=(beta.source_type, beta.repo, beta.file_path, beta.document_hash),
        heading_path=("Alpha",),
        text=alpha.sections[0].text,
        start_char=alpha.sections[0].start_char,
        end_char=alpha.sections[0].end_char,
    )

    with pytest.raises(ValueError, match="section text must match the source document slice"):
        ParsedMarkdownDocument(document=beta, sections=(invalid_section,))


def test_invalid_fence_closer_does_not_expose_code_headings() -> None:
    content = "```python\n```not-a-close\n# Still code\n```\n# Guide\nVisible text.\n"

    sections = MarkdownParser().parse(_document(content)).sections

    assert [section.heading_path for section in sections] == [(), ("Guide",)]
    assert sections[0].text == "```python\n```not-a-close\n# Still code\n```\n"


def test_backtick_fence_info_cannot_contain_backticks_but_tilde_info_can() -> None:
    content = (
        "```language `invalid`\n"
        "# Visible\n"
        "Visible text.\n"
        "~~~language `allowed`\n"
        "# Hidden\n"
        "~~~\n"
        "# Final\n"
        "Final text.\n"
    )

    sections = MarkdownParser().parse(_document(content)).sections

    assert [section.heading_path for section in sections] == [(), ("Visible",), ("Final",)]
    assert sections[0].text == "```language `invalid`\n"
    assert sections[1].text == "Visible text.\n~~~language `allowed`\n# Hidden\n~~~\n"


def test_lf_and_crlf_equivalent_documents_keep_chunk_boundaries_and_ids() -> None:
    lf_content = "# Guide\nAlpha\nBeta\nGamma\n"
    crlf_content = lf_content.replace("\n", "\r\n")
    parser = MarkdownParser()
    chunker = MarkdownChunker(chunk_size=6)
    lf_parsed = parser.parse(_document(lf_content))
    crlf_parsed = parser.parse(_document(crlf_content))

    lf_chunks = chunker.chunk(lf_parsed)
    crlf_chunks = chunker.chunk(crlf_parsed)

    assert lf_parsed.document.document_hash == crlf_parsed.document.document_hash
    assert [(chunk.text, chunk.chunk_id) for chunk in lf_chunks] == [
        (chunk.text, chunk.chunk_id) for chunk in crlf_chunks
    ]
    assert all(
        crlf_content[chunk.metadata.start_char : chunk.metadata.end_char]
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        == chunk.text
        for chunk in crlf_chunks
    )
