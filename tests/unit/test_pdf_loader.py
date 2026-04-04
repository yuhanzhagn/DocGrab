from pathlib import Path

from rag.loaders.multi_loader import MultiDocumentLoader
from rag.loaders.pdf_loader import PDFDocumentLoader
from rag.loaders.text_loader import TextDocumentLoader


class _FakePDFPage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _FakePDFReader:
    def __init__(self, _path: Path) -> None:
        self.pages = [
            _FakePDFPage("First page about Chroma."),
            _FakePDFPage("Second page about citations."),
        ]


def test_pdf_loader_loads_pdf_with_page_spans(tmp_path: Path) -> None:
    path = tmp_path / "manual.pdf"
    path.write_bytes(b"%PDF-1.4 fake")
    loader = PDFDocumentLoader(reader_factory=_FakePDFReader)

    document = loader.load(path)

    assert document.file_name == "manual.pdf"
    assert document.file_extension == ".pdf"
    assert document.metadata["source_type"] == "pdf"
    assert document.metadata["file_type"] == "pdf"
    assert document.metadata["document_title"] == "manual"
    assert document.metadata["page_number"] is None
    assert document.metadata["page_count"] == 2
    assert document.metadata["page_spans"][0]["page_number"] == 1
    assert document.metadata["page_spans"][1]["page_number"] == 2
    assert "First page about Chroma." in document.content
    assert "Second page about citations." in document.content


def test_multi_loader_supports_pdf_and_text(tmp_path: Path) -> None:
    loader = MultiDocumentLoader(
        loaders=[
            TextDocumentLoader(),
            PDFDocumentLoader(reader_factory=_FakePDFReader),
        ]
    )

    text_path = tmp_path / "notes.txt"
    text_path.write_text("hello", encoding="utf-8")
    pdf_path = tmp_path / "manual.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    assert loader.supports(text_path) is True
    assert loader.supports(pdf_path) is True
    assert loader.load(pdf_path).metadata["source_type"] == "pdf"
