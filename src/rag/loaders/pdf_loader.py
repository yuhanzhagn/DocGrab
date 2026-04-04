import hashlib
from pathlib import Path
from typing import Any

from rag.loaders.base import DocumentLoader
from rag.schemas.document import RawDocument


class PDFDocumentLoader(DocumentLoader):
    SUPPORTED_EXTENSIONS = {".pdf"}

    def __init__(self, reader_factory: type | None = None) -> None:
        self._reader_factory = reader_factory or self._default_reader_factory

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def load(self, path: Path) -> RawDocument:
        reader = self._reader_factory(path)
        page_texts: list[str] = []
        page_spans: list[dict[str, int]] = []
        cursor = 0

        for page_index, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                continue

            if page_texts:
                cursor += 2
            start_char = cursor
            page_texts.append(page_text)
            cursor += len(page_text)
            page_spans.append(
                {
                    "page_number": page_index,
                    "start_char": start_char,
                    "end_char": cursor,
                }
            )

        content = "\n\n".join(page_texts)
        document_id = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
        metadata = {
            "source_type": "pdf",
            "extension": path.suffix.lower(),
            "file_name": path.name,
            "file_type": "pdf",
            "document_title": path.stem,
            "section_header": None,
            "page_number": page_spans[0]["page_number"] if len(page_spans) == 1 else None,
            "page_spans": page_spans,
            "page_count": len(getattr(reader, "pages", [])),
        }
        return RawDocument.from_path(
            document_id=document_id,
            path=path,
            content=content,
            metadata=metadata,
        )

    @staticmethod
    def _default_reader_factory(path: Path) -> Any:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("PDF support requires the 'pypdf' package to be installed.") from exc
        return PdfReader(str(path))
