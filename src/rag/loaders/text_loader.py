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
        metadata = {
            "source_type": "text",
            "extension": path.suffix.lower(),
        }
        return RawDocument.from_path(
            document_id=document_id,
            path=path,
            content=content,
            metadata=metadata,
        )
