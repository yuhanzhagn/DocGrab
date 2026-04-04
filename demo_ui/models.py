from django.db import models


class QueryHistory(models.Model):
    question = models.TextField()
    answer_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        preview = self.question.strip().replace("\n", " ")
        return preview[:60] + ("..." if len(preview) > 60 else "")
