from rag.generation.base import AnswerGenerator
from rag.retrieval.retriever import Retriever
from rag.schemas.answer import FinalAnswer


class QueryService:
    def __init__(self, retriever: Retriever, generator: AnswerGenerator) -> None:
        self.retriever = retriever
        self.generator = generator

    def answer(self, query: str, top_k: int) -> FinalAnswer:
        retrieval_results = self.retriever.retrieve(query=query, top_k=top_k)
        return self.generator.generate(query=query, retrieval_results=retrieval_results)
