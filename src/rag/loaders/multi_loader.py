from pathlib import Path

from rag.loaders.base import DocumentLoader
from rag.schemas.document import RawDocument


class MultiDocumentLoader(DocumentLoader):
    def __init__(self, loaders: list[DocumentLoader]) -> None:
        self.loaders = loaders

    def supports(self, path: Path) -> bool:
        return any(loader.supports(path) for loader in self.loaders)

    def load(self, path: Path) -> RawDocument:
        for loader in self.loaders:
            if loader.supports(path):
                return loader.load(path)
        raise ValueError(f"No loader available for path: {path}")
