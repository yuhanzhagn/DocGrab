from collections.abc import Callable
from typing import Any

import httpx

from rag.generation.base import AnswerGenerator
from rag.generation.grounded import GroundedAnswerBuilder
from rag.schemas.answer import FinalAnswer
from rag.schemas.retrieval import RetrievalResult


class OllamaAnswerGenerator(AnswerGenerator):
    """Answer generator backed by an Ollama model server."""

    def __init__(
        self,
        model_name: str,
        endpoint: str | None,
        *,
        timeout_seconds: float = 120.0,
        temperature: float = 0.0,
        max_new_tokens: int = 160,
        client_factory: Callable[..., httpx.Client] | None = None,
    ) -> None:
        if not model_name:
            raise ValueError("An Ollama generator model name is required.")
        if not endpoint:
            raise ValueError("Ollama generator provider requires LOCAL_MODEL_ENDPOINT.")

        self.model_name = model_name
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self._client_factory = client_factory or httpx.Client
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
        payload = {
            "model": self.model_name,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer only from retrieved context. "
                        "If context is insufficient, say so clearly. "
                        "Do not invent citations."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "options": {
                "num_predict": self.max_new_tokens,
                "temperature": self.temperature,
            },
        }

        with self._client_factory(timeout=self.timeout_seconds) as client:
            try:
                response = client.post(f"{self.endpoint}/api/chat", json=payload)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = self._extract_error_detail(exc.response)
                raise RuntimeError(
                    "Ollama generator request failed "
                    f"with status {exc.response.status_code}: {detail}"
                ) from exc
            except httpx.HTTPError as exc:
                raise RuntimeError(f"Ollama generator request failed: {exc}") from exc

        body = response.json()
        message = body.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("Ollama generator response did not contain a valid message.")
        return str(message.get("content", ""))

    @staticmethod
    def _extract_error_detail(response: httpx.Response) -> str:
        try:
            payload: Any = response.json()
        except ValueError:
            return response.text[:200].strip() or "unknown error"

        if isinstance(payload, dict):
            if payload.get("error"):
                return str(payload["error"])
            if payload.get("message"):
                return str(payload["message"])
        return "unknown error"
