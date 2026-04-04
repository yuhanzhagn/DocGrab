from django.urls import path

from demo_ui import views


urlpatterns = [
    path("", views.home, name="home"),
    path("ingest/", views.ingest_view, name="ingest"),
    path("query/", views.query_view, name="query"),
]
