from __future__ import annotations

from dataclasses import dataclass
import re

from docgrab_ingest.hashing import compute_document_hash
from docgrab_ingest.models import IngestDocument
from docgrab_ingest.sources.base import SourceItem

MARKDOWN_PARSER_VERSION = "markdown-parser-v1"

_ATX_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*$")
_CLOSING_HASHES = re.compile(r"[ \t]+#+[ \t]*$")
_FENCE_OPENING = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})(.*)$")
_FENCE_CLOSING = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})[ \t]*(?:\r\n|\r|\n)?$")


@dataclass(frozen=True, slots=True)
class MarkdownSection:
    """A Markdown body region associated with its enclosing heading path."""

    document_identity: tuple[str, str | None, str, str]
    heading_path: tuple[str, ...]
    text: str
    start_char: int
    end_char: int


@dataclass(frozen=True, slots=True)
class ParsedMarkdownDocument:
    """A parsed document whose sections are bound to the source text."""

    document: IngestDocument
    sections: tuple[MarkdownSection, ...]

    def __post_init__(self) -> None:
        for section in self.sections:
            if section.document_identity != _document_identity(self.document):
                raise ValueError("section identity must match the source document")
            if section.start_char < 0 or section.end_char < section.start_char:
                raise ValueError("section offsets must be ordered, non-negative values")
            if section.end_char > len(self.document.content):
                raise ValueError("section offsets must be within the source document")
            if self.document.content[section.start_char : section.end_char] != section.text:
                raise ValueError("section text must match the source document slice")


class MarkdownParser:
    """Split Markdown into heading-scoped body regions without normalizing text."""

    def parse_bytes(self, source: SourceItem, content: bytes) -> ParsedMarkdownDocument:
        """Decode source bytes and parse them into a document-bound Markdown result."""
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("Markdown source must be valid UTF-8") from exc

        document = IngestDocument(
            source_type=source.source_type,
            repo=source.repo,
            file_path=source.file_path,
            content=text,
            document_hash=compute_document_hash(text),
            parser_version=MARKDOWN_PARSER_VERSION,
        )
        return self.parse(document)

    def parse(self, document: IngestDocument) -> ParsedMarkdownDocument:
        sections: list[MarkdownSection] = []
        heading_stack: list[tuple[int, str]] = []
        section_start = 0
        offset = 0
        open_fence: tuple[str, int] | None = None

        for line in document.content.splitlines(keepends=True):
            line_start = offset
            offset += len(line)

            fence = _FENCE_OPENING.match(line)
            if open_fence is not None:
                if self._is_closing_fence(line, open_fence):
                    open_fence = None
                continue
            if fence and self._is_valid_opening_fence(fence):
                open_fence = (fence.group(1)[0], len(fence.group(1)))
                continue

            heading = _ATX_HEADING.match(line.rstrip("\r\n"))
            if heading is None:
                continue

            self._append_section(sections, document, section_start, line_start, heading_stack)
            title = _CLOSING_HASHES.sub("", heading.group(2) or "").strip()
            level = len(heading.group(1))
            heading_stack = [item for item in heading_stack if item[0] < level]
            if title:
                heading_stack.append((level, title))
            section_start = offset

        self._append_section(sections, document, section_start, len(document.content), heading_stack)
        return ParsedMarkdownDocument(document=document, sections=tuple(sections))

    @staticmethod
    def _is_closing_fence(line: str, opening_fence: tuple[str, int]) -> bool:
        fence = _FENCE_CLOSING.match(line)
        return bool(
            fence
            and fence.group(1)[0] == opening_fence[0]
            and len(fence.group(1)) >= opening_fence[1]
        )

    @staticmethod
    def _is_valid_opening_fence(fence: re.Match[str]) -> bool:
        return fence.group(1)[0] == "~" or "`" not in fence.group(2)

    @staticmethod
    def normalized_text_and_source_offsets(section: MarkdownSection) -> tuple[str, tuple[int, ...]]:
        """Return LF-normalized text and logical-boundary offsets into the raw section."""
        normalized: list[str] = []
        source_offsets = [0]
        source_position = 0

        while source_position < len(section.text):
            character = section.text[source_position]
            if character == "\r":
                if source_position + 1 < len(section.text) and section.text[source_position + 1] == "\n":
                    source_position += 2
                else:
                    source_position += 1
                normalized.append("\n")
            else:
                source_position += 1
                normalized.append(character)
            source_offsets.append(source_position)

        return "".join(normalized), tuple(source_offsets)

    @staticmethod
    def _append_section(
        sections: list[MarkdownSection],
        document: IngestDocument,
        start_char: int,
        end_char: int,
        heading_stack: list[tuple[int, str]],
    ) -> None:
        text = document.content[start_char:end_char]
        if text.strip():
            sections.append(
                MarkdownSection(
                    document_identity=_document_identity(document),
                    heading_path=tuple(title for _, title in heading_stack),
                    text=text,
                    start_char=start_char,
                    end_char=end_char,
                )
            )


def _document_identity(document: IngestDocument) -> tuple[str, str | None, str, str]:
    return document.source_type, document.repo, document.file_path, document.document_hash
