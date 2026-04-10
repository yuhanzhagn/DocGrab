from unittest.mock import patch

from django.contrib.admin.sites import site
from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from demo_ui.admin import QueryHistoryAdmin
from demo_ui.models import QueryHistory


class DemoUISmokeTests(TestCase):
    def test_home_page_renders(self) -> None:
        QueryHistory.objects.create(
            question="What is Chroma?",
            answer_text="Chroma stores embeddings.",
        )

        response = self.client.get(reverse("home"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "RAG Demo Shell")
        self.assertContains(response, "Recent Queries")
        self.assertContains(response, "What is Chroma?")

    def test_ingest_page_renders(self) -> None:
        response = self.client.get(reverse("ingest"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Ingest Documents")
        self.assertContains(response, "Directory path")

    def test_query_page_renders(self) -> None:
        response = self.client.get(reverse("query"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Query Documents")
        self.assertContains(response, "Question")

    @patch("demo_ui.views.call_ingest")
    def test_ingest_view_calls_backend_helper(self, mock_call_ingest) -> None:
        mock_call_ingest.return_value = {
            "indexed_documents": 2,
            "indexed_chunks": 5,
            "skipped_files": [],
        }

        response = self.client.post(
            reverse("ingest"),
            {"directory": "/tmp/docs"},
        )

        self.assertEqual(response.status_code, 200)
        mock_call_ingest.assert_called_once_with(directory="/tmp/docs")
        self.assertContains(response, "Ingestion completed.")
        self.assertContains(response, "Indexed documents:")
        self.assertContains(response, "2")

    @patch("demo_ui.views.call_query")
    def test_query_view_calls_backend_helper_and_saves_history(self, mock_call_query) -> None:
        mock_call_query.return_value = {
            "result": {
                "answer_text": "Chroma stores document embeddings.",
                "citations": [
                    {
                        "source_path": "/tmp/architecture.md",
                        "file_name": "architecture.md",
                        "section_header": "Overview",
                        "page_number": None,
                        "snippet": "Chroma stores document embeddings.",
                    }
                ],
                "retrieved_chunks": [
                    {
                        "file_name": "architecture.md",
                        "source_path": "/tmp/architecture.md",
                        "score": 0.45,
                        "relevance": "high",
                        "page_number": None,
                        "text": "Chroma stores document embeddings.",
                    }
                ],
            }
        }

        response = self.client.post(
            reverse("query"),
            {
                "question": "Which database stores document embeddings?",
                "top_k": 3,
            },
        )

        self.assertEqual(response.status_code, 200)
        mock_call_query.assert_called_once_with(
            question="Which database stores document embeddings?",
            top_k=3,
        )
        self.assertContains(response, "Query completed.")
        self.assertContains(response, "Chroma stores document embeddings.")
        self.assertEqual(QueryHistory.objects.count(), 1)
        history = QueryHistory.objects.get()
        self.assertEqual(history.question, "Which database stores document embeddings?")
        self.assertEqual(history.answer_text, "Chroma stores document embeddings.")

    def test_query_history_is_registered_in_admin(self) -> None:
        self.assertIn(QueryHistory, site._registry)
        self.assertIsInstance(site._registry[QueryHistory], QueryHistoryAdmin)

    def test_query_history_admin_page_renders(self) -> None:
        user_model = get_user_model()
        admin_user = user_model.objects.create_superuser(
            username="admin",
            email="admin@example.com",
            password="password123",
        )
        self.client.force_login(admin_user)
        QueryHistory.objects.create(
            question="What is retrieval?",
            answer_text="Retrieval ranks chunks by similarity.",
        )

        response = self.client.get(reverse("admin:demo_ui_queryhistory_changelist"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "What is retrieval?")
