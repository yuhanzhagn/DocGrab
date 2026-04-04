from typing import Any

import httpx
from django.conf import settings


class BackendClientError(Exception):
    pass


def call_ingest(directory: str) -> dict[str, Any]:
    return _post("/documents/ingest", {"directory": directory})


def call_query(question: str, top_k: int) -> dict[str, Any]:
    return _post(
        "/query/",
        {
            "query": question,
            "top_k": top_k,
        },
    )


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{settings.FASTAPI_BASE_URL.rstrip('/')}{path}"
    try:
        response = httpx.post(url, json=payload, timeout=30.0)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = _extract_detail(exc.response)
        raise BackendClientError(detail) from exc
    except httpx.HTTPError as exc:
        raise BackendClientError(f"Could not reach FastAPI backend at {url}.") from exc
    return response.json()


def _extract_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or f"Request failed with status {response.status_code}."

    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str):
            return detail
    return f"Request failed with status {response.status_code}."
