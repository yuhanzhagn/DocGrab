from fastapi import FastAPI

from rag.api.router import api_router
from rag.config.settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Local-first RAG MVP with document ingestion and grounded answers.",
    )
    app.include_router(api_router, prefix=settings.api_prefix)
    return app


app = create_app()
