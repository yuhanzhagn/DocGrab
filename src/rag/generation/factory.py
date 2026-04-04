from rag.config.settings import Settings
from rag.generation.base import AnswerGenerator
from rag.generation.external import ExternalAPIAnswerGenerator
from rag.generation.local import LocalModelAnswerGenerator
from rag.generation.simple import SimpleGroundedAnswerGenerator


def create_generator(settings: Settings) -> AnswerGenerator:
    provider = settings.generator_provider.strip().lower()

    if provider == "simple":
        return SimpleGroundedAnswerGenerator()
    if provider == "local":
        return LocalModelAnswerGenerator(
            model_name=settings.generator_model_name,
            max_new_tokens=settings.local_generator_max_new_tokens,
        )
    if provider == "external":
        return ExternalAPIAnswerGenerator(
            model_name=settings.generator_model_name,
            api_key=settings.external_generator_api_key,
            base_url=settings.external_generator_base_url,
            timeout_seconds=settings.external_generator_timeout_seconds,
            temperature=settings.external_generator_temperature,
        )

    raise ValueError(
        "Unsupported generator provider. Expected one of: simple, local, external."
    )
