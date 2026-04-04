from typing import Any

from django.contrib import messages
from django.shortcuts import render

from demo_ui.forms import IngestForm, QueryForm
from demo_ui.models import QueryHistory
from demo_ui.services import BackendClientError, call_ingest, call_query


def home(request):
    return render(
        request,
        "demo_ui/home.html",
        {
            "recent_queries": QueryHistory.objects.all()[:5],
        },
    )


def ingest_view(request):
    result: dict[str, Any] | None = None
    form = IngestForm(request.POST or None)

    if request.method == "POST" and form.is_valid():
        try:
            result = call_ingest(directory=form.cleaned_data["directory"])
            messages.success(request, "Ingestion completed.")
        except BackendClientError as exc:
            messages.error(request, str(exc))

    return render(
        request,
        "demo_ui/ingest.html",
        {
            "form": form,
            "result": result,
        },
    )


def query_view(request):
    result: dict[str, Any] | None = None
    form = QueryForm(request.POST or None)

    if request.method == "POST" and form.is_valid():
        question = form.cleaned_data["question"]
        top_k = form.cleaned_data.get("top_k") or 5
        try:
            response_payload = call_query(question=question, top_k=top_k)
            result = response_payload.get("result", {})
            QueryHistory.objects.create(
                question=question,
                answer_text=str(result.get("answer_text", "")),
            )
            messages.success(request, "Query completed.")
        except BackendClientError as exc:
            messages.error(request, str(exc))

    return render(
        request,
        "demo_ui/query.html",
        {
            "form": form,
            "result": result,
        },
    )
