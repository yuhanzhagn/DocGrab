from __future__ import annotations

from docgrab_ingest.hashing import compute_chunk_hash, compute_chunk_id
from docgrab_ingest.models import ChunkMetadata, IngestChunk
from docgrab_ingest.parsers.markdown import MarkdownParser, MarkdownSection, ParsedMarkdownDocument

MARKDOWN_CHUNKER_VERSION = "markdown-chunker-v1"


class MarkdownChunker:
    """Create bounded, heading-aware chunks from parsed Markdown sections."""

    def __init__(self, chunk_size: int = 800) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size

    def chunk(self, parsed_document: ParsedMarkdownDocument) -> tuple[IngestChunk, ...]:
        document = parsed_document.document
        chunks: list[IngestChunk] = []
        occurrence_counts: dict[str, int] = {}

        for section in parsed_document.sections:
            for text, start_char, end_char in self._split_section(section):
                content_hash = compute_chunk_hash(
                    text,
                    source_type=document.source_type,
                    repo=document.repo,
                    file_path=document.file_path,
                    heading_path=section.heading_path,
                    parser_version=document.parser_version,
                    chunker_version=MARKDOWN_CHUNKER_VERSION,
                )
                occurrence_ordinal = occurrence_counts.get(content_hash, 0)
                occurrence_counts[content_hash] = occurrence_ordinal + 1
                metadata = ChunkMetadata(
                    source_type=document.source_type,
                    repo=document.repo,
                    file_path=document.file_path,
                    heading_path=section.heading_path,
                    symbol_name=None,
                    chunk_index=len(chunks),
                    occurrence_ordinal=occurrence_ordinal,
                    content_hash=content_hash,
                    document_hash=document.document_hash,
                    embedding_model=None,
                    parser_version=document.parser_version,
                    chunker_version=MARKDOWN_CHUNKER_VERSION,
                    start_char=start_char,
                    end_char=end_char,
                )
                chunks.append(
                    IngestChunk(
                        chunk_id=compute_chunk_id(metadata),
                        document_hash=document.document_hash,
                        text=text,
                        metadata=metadata,
                    )
                )

        return tuple(chunks)

    def _split_section(
        self, section: MarkdownSection
    ) -> tuple[tuple[str, int, int], ...]:
        text, source_offsets = MarkdownParser.normalized_text_and_source_offsets(section)
        chunks: list[tuple[str, int, int]] = []
        position = 0
        while position < len(text):
            limit = min(position + self.chunk_size, len(text))
            end = limit
            if limit < len(text):
                end = self._natural_boundary(text, position, limit) or limit

            chunk_text = text[position:end]
            if chunk_text.strip():
                chunks.append(
                    (
                        chunk_text,
                        section.start_char + source_offsets[position],
                        section.start_char + source_offsets[end],
                    )
                )
            position = end

        return tuple(chunks)

    @staticmethod
    def _natural_boundary(text: str, start: int, limit: int) -> int | None:
        paragraph_break = text.rfind("\n\n", start, limit)
        if paragraph_break > start:
            return paragraph_break + 2

        line_break = text.rfind("\n", start, limit)
        if line_break > start:
            return line_break + 1

        return None
