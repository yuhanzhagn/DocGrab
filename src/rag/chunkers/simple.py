import hashlib

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
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                split_at = text.rfind("\n\n", start, end)
                if split_at == -1:
                    split_at = text.rfind("\n", start, end)
                if split_at != -1 and split_at > start:
                    end = split_at

            chunk_text = text[start:end].strip()
            if chunk_text:
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
                            "file_name": document.file_name,
                            "chunk_index": chunk_index,
                            "start_char": start,
                            "end_char": end,
                        },
                    )
                )
                chunk_index += 1

            if end >= len(text):
                break

            start = max(end - self.chunk_overlap, 0)

        return chunks
