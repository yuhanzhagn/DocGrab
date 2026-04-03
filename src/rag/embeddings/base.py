from abc import ABC, abstractmethod


class Embedder(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        raise NotImplementedError
