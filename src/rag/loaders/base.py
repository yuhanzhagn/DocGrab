from abc import ABC, abstractmethod
from pathlib import Path

from rag.schemas.document import RawDocument


class DocumentLoader(ABC):
    @abstractmethod
    def supports(self, path: Path) -> bool:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path) -> RawDocument:
        raise NotImplementedError
