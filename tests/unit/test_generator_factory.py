import httpx
import pytest

from rag.config.settings import Settings
from rag.generation.external import ExternalAPIAnswerGenerator
from rag.generation.factory import create_generator
from rag.generation.local import LocalModelAnswerGenerator
from rag.generation.simple import SimpleGroundedAnswerGenerator
from rag.schemas.retrieval import RetrievalResult


class _FakePipeline:
    def __call__(self, prompt: str, **_: object) -> list[dict[str, str]]:
        assert "Answer only from the retrieved context." in prompt
        return [{"generated_text": "Chroma stores document embeddings."}]


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://example.test/chat/completions")
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
        assert url.endswith("/chat/completions")
        assert json["model"]
        assert json["messages"][0]["role"] == "system"
        assert "Authorization" in headers
        return self._response


def _sample_results() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            source_path="/tmp/architecture.md",
            text="Chroma stores document embeddings for retrieval.",
            score=0.45,
            distance=0.55,
            relevance="high",
            metadata={"chunk_index": 0, "start_char": 0, "end_char": 48},
        ),
        RetrievalResult(
            chunk_id="chunk-2",
            document_id="doc-1",
            source_path="/tmp/architecture.md",
            text="Answers should stay grounded in retrieved chunks.",
            score=0.31,
            distance=0.69,
            relevance="medium",
            metadata={"chunk_index": 1, "start_char": 49, "end_char": 98},
        ),
    ]


def test_generator_factory_returns_simple_provider_by_default() -> None:
    generator = create_generator(Settings())
    assert isinstance(generator, SimpleGroundedAnswerGenerator)


def test_generator_factory_returns_local_provider() -> None:
    generator = create_generator(
        Settings(
            generator_provider="local",
            generator_model_name="google/flan-t5-small",
        )
    )
    assert isinstance(generator, LocalModelAnswerGenerator)
    assert generator.model_name == "google/flan-t5-small"


def test_generator_factory_requires_external_credentials() -> None:
    with pytest.raises(ValueError, match="EXTERNAL_GENERATOR_API_KEY"):
        create_generator(
            Settings(
                generator_provider="external",
                generator_model_name="gpt-4o-mini",
                external_generator_api_key=None,
            )
        )


def test_local_generator_returns_grounded_answer_with_builder_citations() -> None:
    generator = LocalModelAnswerGenerator(
        model_name="google/flan-t5-small",
        pipeline_factory=lambda *args, **kwargs: _FakePipeline(),
    )

    answer = generator.generate(
        query="Which database stores document embeddings?",
        retrieval_results=_sample_results(),
    )

    assert answer.answer_text == "Chroma stores document embeddings."
    assert len(answer.citations) == 2
    assert answer.citations[0].source_path == "/tmp/architecture.md"
    assert answer.retrieved_chunks[0]["chunk_id"] == "chunk-1"


def test_external_generator_uses_api_response() -> None:
    generator = ExternalAPIAnswerGenerator(
        model_name="gpt-4o-mini",
        api_key="test-key",
        client_factory=lambda **kwargs: _FakeClient(
            _FakeResponse(
                {
                    "choices": [
                        {"message": {"content": "Chroma stores document embeddings."}}
                    ]
                }
            ),
            **kwargs,
        ),
    )

    answer = generator.generate(
        query="Which database stores document embeddings?",
        retrieval_results=_sample_results(),
    )

    assert answer.answer_text == "Chroma stores document embeddings."
    assert len(answer.citations) == 2


def test_external_generator_surfaces_http_errors() -> None:
    generator = ExternalAPIAnswerGenerator(
        model_name="gpt-4o-mini",
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
        generator.generate(
            query="Which database stores document embeddings?",
            retrieval_results=_sample_results(),
        )
