from django import forms


class IngestForm(forms.Form):
    directory = forms.CharField(
        label="Directory path",
        max_length=1000,
        widget=forms.TextInput(
            attrs={
                "placeholder": "/absolute/path/to/documents",
            }
        ),
    )


class QueryForm(forms.Form):
    question = forms.CharField(
        label="Question",
        widget=forms.Textarea(
            attrs={
                "rows": 4,
                "placeholder": "Which database stores document embeddings?",
            }
        ),
    )
    top_k = forms.IntegerField(label="Top K", min_value=1, max_value=20, initial=5, required=False)
