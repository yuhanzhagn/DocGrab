from django.contrib import admin

from demo_ui.models import QueryHistory


@admin.register(QueryHistory)
class QueryHistoryAdmin(admin.ModelAdmin):
    list_display = ("short_question", "created_at")
    search_fields = ("question", "answer_text")
    readonly_fields = ("created_at",)

    @staticmethod
    def short_question(obj: QueryHistory) -> str:
        preview = obj.question.strip().replace("\n", " ")
        return preview[:80] + ("..." if len(preview) > 80 else "")

    short_question.short_description = "Question"
