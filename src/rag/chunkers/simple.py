import hashlib
import re

from rag.schemas.document import Chunk, RawDocument
from rag.chunkers.base import Chunker


class SimpleTextChunker(Chunker):
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: RawDocument) -> list[Chunk]:
        text = document.content.strip()
        if not text:
            return []

        chunks: list[Chunk] = []
        start = 0
        chunk_index = 0

        while start < len(text):
            default_end = min(start + self.chunk_size, len(text))
            end = default_end
            if default_end < len(text):
                split_at = text.rfind("\n\n", start, default_end)
                if split_at == -1:
                    split_at = text.rfind("\n", start, default_end)
                # Favor natural boundaries, but never allow overlap handling to stall progress.
                if split_at != -1 and split_at > start + self.chunk_overlap:
                    end = split_at

            if end <= start:
                end = default_end

            chunk_text = text[start:end].strip()
            if chunk_text:
                section_header = self._find_section_header(text=text, start=start, end=end)
                if section_header is None and chunk_index == 0:
                    section_header = str(document.metadata.get("document_title") or "").strip() or None
                digest = hashlib.sha256(
                    f"{document.document_id}:{chunk_index}:{start}:{end}".encode("utf-8")
                ).hexdigest()
                chunks.append(
                    Chunk(
                        chunk_id=digest,
                        document_id=document.document_id,
                        text=chunk_text,
                        chunk_index=chunk_index,
                        start_char=start,
                        end_char=end,
                        metadata={
                            "source_path": document.source_path,
                            "document_title": document.metadata.get("document_title")
                            or document.file_name,
                            "file_name": document.file_name,
                            "file_type": document.metadata.get("file_type")
                            or document.file_extension.lstrip("."),
                            "section_header": section_header,
                            "page_number": document.metadata.get("page_number"),
                            "chunk_index": chunk_index,
                            "start_char": start,
                            "end_char": end,
                        },
                    )
                )
                chunk_index += 1

            if end >= len(text):
                break

            next_start = max(end - self.chunk_overlap, 0)
            start = end if next_start <= start else next_start

        return chunks

    @staticmethod
    def _find_section_header(text: str, start: int, end: int) -> str | None:
        latest_header: str | None = None
        latest_position = -1

        for match in re.finditer(r"(?m)^(#{1,6})\s+(.+?)\s*$", text):
            header_text = match.group(2).strip()
            if not header_text:
                continue
            position = match.start()
            if position <= end and position >= latest_position:
                latest_header = header_text
                latest_position = position

        if latest_header is not None:
            return latest_header
        return None
