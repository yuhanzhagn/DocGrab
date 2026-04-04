from collections.abc import Callable
from typing import Any

import httpx

from rag.generation.base import AnswerGenerator
from rag.generation.grounded import GroundedAnswerBuilder
from rag.schemas.answer import FinalAnswer
from rag.schemas.retrieval import RetrievalResult


class ExternalAPIAnswerGenerator(AnswerGenerator):
    """External generator using an OpenAI-compatible chat completions API."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None,
        *,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 30.0,
        temperature: float = 0.0,
        client_factory: Callable[..., httpx.Client] | None = None,
    ) -> None:
        if not model_name:
            raise ValueError("An external generator model name is required.")
        if not api_key:
            raise ValueError(
                "External generator provider requires EXTERNAL_GENERATOR_API_KEY."
            )

        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
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
            "temperature": self.temperature,
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
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        with self._client_factory(timeout=self.timeout_seconds) as client:
            try:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = self._extract_error_detail(exc.response)
                raise RuntimeError(
                    "External generator request failed "
                    f"with status {exc.response.status_code}: {detail}"
                ) from exc
            except httpx.HTTPError as exc:
                raise RuntimeError(f"External generator request failed: {exc}") from exc

        body = response.json()
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("External generator response did not contain any choices.")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        return str(content)

    @staticmethod
    def _extract_error_detail(response: httpx.Response) -> str:
        try:
            payload: Any = response.json()
        except ValueError:
            return response.text[:200].strip() or "unknown error"

        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict) and error.get("message"):
                return str(error["message"])
            if payload.get("message"):
                return str(payload["message"])
        return "unknown error"
