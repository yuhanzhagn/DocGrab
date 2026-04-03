from pathlib import Path

from rag.loaders.text_loader import TextDocumentLoader


def test_text_loader_loads_markdown_file(sample_data_dir: Path) -> None:
    loader = TextDocumentLoader()
    path = sample_data_dir / "architecture.md"

    document = loader.load(path)

    assert document.file_name == "architecture.md"
    assert document.file_extension == ".md"
    assert document.source_path == str(path)
    assert "Chroma" in document.content
    assert document.metadata["source_type"] == "text"


def test_text_loader_supports_txt_and_md() -> None:
    loader = TextDocumentLoader()

    assert loader.supports(Path("notes.md")) is True
    assert loader.supports(Path("notes.txt")) is True
    assert loader.supports(Path("notes.pdf")) is False
