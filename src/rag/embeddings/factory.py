from rag.config.settings import Settings
from rag.embeddings.base import Embedder
from rag.embeddings.external_embedder import ExternalAPIEmbedder
from rag.embeddings.hash_embedder import HashingEmbedder
from rag.embeddings.local_embedder import LocalSentenceTransformerEmbedder


def create_embedder(settings: Settings) -> Embedder:
    provider = settings.embedder_provider.strip().lower()

    if provider == "hash":
        return HashingEmbedder(dimension=settings.embedding_dimension)
    if provider == "local":
        return LocalSentenceTransformerEmbedder(model_name=settings.embedding_model_name)
    if provider == "external":
        return ExternalAPIEmbedder(
            model_name=settings.embedding_model_name,
            api_key=settings.external_embedding_api_key,
            base_url=settings.external_embedding_base_url,
            timeout_seconds=settings.external_embedding_timeout_seconds,
        )

    raise ValueError(
        "Unsupported embedder provider. Expected one of: hash, local, external."
    )
