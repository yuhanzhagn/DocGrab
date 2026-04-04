import hashlib
from pathlib import Path

from rag.loaders.base import DocumentLoader
from rag.schemas.document import RawDocument


class TextDocumentLoader(DocumentLoader):
    SUPPORTED_EXTENSIONS = {".md", ".txt"}

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def load(self, path: Path) -> RawDocument:
        content = path.read_text(encoding="utf-8")
        document_id = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
        title = self._extract_title(path=path, content=content)
        metadata = {
            "source_type": "text",
            "extension": path.suffix.lower(),
            "file_name": path.name,
            "file_type": path.suffix.lower().lstrip("."),
            "document_title": title,
            "section_header": None,
            "page_number": None,
        }
        return RawDocument.from_path(
            document_id=document_id,
            path=path,
            content=content,
            metadata=metadata,
        )

    @staticmethod
    def _extract_title(path: Path, content: str) -> str:
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip() or path.stem
            return stripped
        return path.stem
