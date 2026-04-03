from abc import ABC, abstractmethod

from rag.schemas.answer import FinalAnswer
from rag.schemas.retrieval import RetrievalResult


class AnswerGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, retrieval_results: list[RetrievalResult]) -> FinalAnswer:
        raise NotImplementedError
