from abc import ABC, abstractmethod

from rag.schemas.document import Chunk, RawDocument


class Chunker(ABC):
    @abstractmethod
    def chunk(self, document: RawDocument) -> list[Chunk]:
        raise NotImplementedError
