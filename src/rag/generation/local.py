from collections.abc import Callable
from typing import Any

from rag.generation.base import AnswerGenerator
from rag.generation.grounded import GroundedAnswerBuilder
from rag.schemas.answer import FinalAnswer
from rag.schemas.retrieval import RetrievalResult


class LocalModelAnswerGenerator(AnswerGenerator):
    """Local generator backed by a lightweight transformers pipeline."""

    def __init__(
        self,
        model_name: str,
        *,
        max_new_tokens: int = 160,
        pipeline_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not model_name:
            raise ValueError("A local generator model name is required.")

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._pipeline_factory = pipeline_factory or self._default_pipeline_factory
        self._pipeline: Any | None = None
        self._builder = GroundedAnswerBuilder()

    def generate(self, query: str, retrieval_results: list[RetrievalResult]) -> FinalAnswer:
        if self._builder.should_fallback(retrieval_results):
            return self._builder.build_fallback_answer(retrieval_results=retrieval_results)

        prompt = self._builder.build_prompt(query=query, retrieval_results=retrieval_results)
        answer_text = self._generate_answer_text(prompt).strip()
        if not answer_text:
            return self._builder.build_fallback_answer(retrieval_results=retrieval_results)

        return self._builder.build_final_answer(
            answer_text=answer_text,
            retrieval_results=retrieval_results,
        )

    def _generate_answer_text(self, prompt: str) -> str:
        pipeline = self._get_pipeline()
        result = pipeline(prompt, max_new_tokens=self.max_new_tokens, truncation=True)
        if not isinstance(result, list) or not result:
            return ""

        first = result[0]
        if "generated_text" in first:
            return str(first["generated_text"]).strip()
        if "summary_text" in first:
            return str(first["summary_text"]).strip()
        return ""

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            self._pipeline = self._pipeline_factory(
                "text2text-generation",
                model=self.model_name,
            )
        return self._pipeline

    @staticmethod
    def _default_pipeline_factory(*args: Any, **kwargs: Any) -> Any:
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Local generator provider requires the optional 'transformers' package "
                "to be installed."
            ) from exc
        return pipeline(*args, **kwargs)
