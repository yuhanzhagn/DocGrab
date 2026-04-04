from collections.abc import Callable
from typing import Any

from rag.embeddings.base import Embedder


class LocalSentenceTransformerEmbedder(Embedder):
    """Local embedding provider backed by sentence-transformers."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        model_loader: Callable[[str], Any] | None = None,
    ) -> None:
        if not model_name:
            raise ValueError("A local embedding model name is required.")

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self._model_loader = model_loader or self._default_model_loader
        self._model: Any | None = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(texts)

    def embed_query(self, query: str) -> list[float]:
        vectors = self._encode([query])
        return vectors[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._get_model().encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = self._model_loader(self.model_name)
        return self._model

    @staticmethod
    def _default_model_loader(model_name: str) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError(
                "Local embedding provider requires the optional "
                "'sentence-transformers' package to be installed."
            ) from exc
        return SentenceTransformer(model_name)
