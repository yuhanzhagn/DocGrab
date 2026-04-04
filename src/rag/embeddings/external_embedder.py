from collections.abc import Callable
from typing import Any

import httpx

from rag.embeddings.base import Embedder


class ExternalAPIEmbedder(Embedder):
    """External embedding provider using an OpenAI-compatible embeddings endpoint."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None,
        *,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 30.0,
        client_factory: Callable[..., httpx.Client] | None = None,
    ) -> None:
        if not model_name:
            raise ValueError("An external embedding model name is required.")
        if not api_key:
            raise ValueError(
                "External embedding provider requires EXTERNAL_EMBEDDING_API_KEY."
            )

        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._client_factory = client_factory or httpx.Client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._request_embeddings(texts)

    def embed_query(self, query: str) -> list[float]:
        vectors = self._request_embeddings([query])
        return vectors[0]

    def _request_embeddings(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self.model_name,
            "input": texts,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        with self._client_factory(timeout=self.timeout_seconds) as client:
            try:
                response = client.post(
                    f"{self.base_url}/embeddings",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = self._extract_error_detail(exc.response)
                raise RuntimeError(
                    "External embedding request failed "
                    f"with status {exc.response.status_code}: {detail}"
                ) from exc
            except httpx.HTTPError as exc:
                raise RuntimeError(
                    f"External embedding request failed: {exc}"
                ) from exc

        body = response.json()
        data = body.get("data")
        if not isinstance(data, list):
            raise RuntimeError("External embedding response did not contain a valid data list.")

        ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
        try:
            return [list(item["embedding"]) for item in ordered]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                "External embedding response did not contain valid embedding vectors."
            ) from exc

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
