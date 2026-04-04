import httpx
import pytest

from rag.config.settings import Settings
from rag.embeddings.external_embedder import ExternalAPIEmbedder
from rag.embeddings.factory import create_embedder
from rag.embeddings.hash_embedder import HashingEmbedder
from rag.embeddings.local_embedder import LocalSentenceTransformerEmbedder


class _FakeEmbeddings:
    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors

    def tolist(self) -> list[list[float]]:
        return self._vectors


class _FakeSentenceTransformer:
    def encode(self, texts: list[str], **_: object) -> _FakeEmbeddings:
        return _FakeEmbeddings([[float(index + 1), 0.5] for index, _text in enumerate(texts)])


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://example.test/embeddings")
            response = httpx.Response(
                self.status_code,
                json=self._payload,
                request=request,
            )
            raise httpx.HTTPStatusError("request failed", request=request, response=response)

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    def __init__(self, response: _FakeResponse, **_: object) -> None:
        self._response = response

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def post(self, url: str, json: dict, headers: dict) -> _FakeResponse:
        assert url.endswith("/embeddings")
        assert json["model"]
        assert "Authorization" in headers
        return self._response


def test_embedder_factory_returns_hash_provider_by_default() -> None:
    settings = Settings()

    embedder = create_embedder(settings)

    assert isinstance(embedder, HashingEmbedder)


def test_embedder_factory_returns_local_provider() -> None:
    settings = Settings(
        embedder_provider="local",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    embedder = create_embedder(settings)

    assert isinstance(embedder, LocalSentenceTransformerEmbedder)
    assert embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"


def test_embedder_factory_requires_external_credentials() -> None:
    settings = Settings(
        embedder_provider="external",
        embedding_model_name="text-embedding-3-small",
        external_embedding_api_key=None,
    )

    with pytest.raises(ValueError, match="EXTERNAL_EMBEDDING_API_KEY"):
        create_embedder(settings)


def test_local_embedder_uses_model_loader() -> None:
    embedder = LocalSentenceTransformerEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_loader=lambda _name: _FakeSentenceTransformer(),
    )

    vectors = embedder.embed_texts(["alpha beta", "gamma"])

    assert vectors == [[1.0, 0.5], [2.0, 0.5]]
    assert embedder.embed_query("delta") == [1.0, 0.5]


def test_external_embedder_uses_api_response() -> None:
    embedder = ExternalAPIEmbedder(
        model_name="text-embedding-3-small",
        api_key="test-key",
        client_factory=lambda **kwargs: _FakeClient(
            _FakeResponse(
                {
                    "data": [
                        {"index": 1, "embedding": [0.3, 0.4]},
                        {"index": 0, "embedding": [0.1, 0.2]},
                    ]
                }
            ),
            **kwargs,
        ),
    )

    vectors = embedder.embed_texts(["first", "second"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_external_embedder_surfaces_http_errors() -> None:
    embedder = ExternalAPIEmbedder(
        model_name="text-embedding-3-small",
        api_key="test-key",
        client_factory=lambda **kwargs: _FakeClient(
            _FakeResponse(
                {"error": {"message": "bad credentials"}},
                status_code=401,
            ),
            **kwargs,
        ),
    )

    with pytest.raises(RuntimeError, match="bad credentials"):
        embedder.embed_query("hello")
