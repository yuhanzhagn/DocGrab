from rag.generation.base import AnswerGenerator
from rag.generation.grounded import GroundedAnswerBuilder
from rag.schemas.answer import FinalAnswer
from rag.schemas.retrieval import RetrievalResult


class SimpleGroundedAnswerGenerator(AnswerGenerator):
    """Minimal grounded generator for the MVP.

    This does not call an LLM yet. It extracts the most relevant chunks and returns
    a concise answer with citations so the whole project is runnable locally.
    """

    def __init__(self) -> None:
        self._builder = GroundedAnswerBuilder()

    def generate(self, query: str, retrieval_results: list[RetrievalResult]) -> FinalAnswer:
        if self._builder.should_fallback(retrieval_results):
            return self._builder.build_fallback_answer(retrieval_results=retrieval_results)

        answer_text = self._builder.build_prompt(query=query, retrieval_results=retrieval_results)
        return self._builder.build_final_answer(
            answer_text=answer_text,
            retrieval_results=retrieval_results,
        )
