from pathlib import Path

import pytest

from docgrab_ingest.sources import LocalFileSource, SourceItem


def test_local_file_source_filters_extensions_and_uses_stable_relative_paths(tmp_path: Path) -> None:
    root = tmp_path / "documents"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "README.MD").write_text("# Read me", encoding="utf-8")
    (nested / "notes.txt").write_text("notes", encoding="utf-8")
    (nested / "ignored.py").write_text("print('ignore')", encoding="utf-8")

    source = LocalFileSource(root, allowed_extensions=(".md", "txt"))

    assert source.discover() == (
        SourceItem(
            source_type="local_file",
            source_uri=(root / "README.MD").resolve().as_uri(),
            file_path="README.MD",
        ),
        SourceItem(
            source_type="local_file",
            source_uri=(nested / "notes.txt").resolve().as_uri(),
            file_path="nested/notes.txt",
        ),
    )


def test_local_file_source_requires_an_existing_directory(tmp_path: Path) -> None:
    missing_source = LocalFileSource(tmp_path / "missing", allowed_extensions=(".md",))
    file_root = tmp_path / "notes.md"
    file_root.write_text("notes", encoding="utf-8")
    file_source = LocalFileSource(file_root, allowed_extensions=(".md",))

    with pytest.raises(FileNotFoundError, match="Source root not found"):
        missing_source.discover()
    with pytest.raises(NotADirectoryError, match="Source root is not a directory"):
        file_source.discover()


def test_local_file_source_excludes_files_resolving_outside_its_root(tmp_path: Path) -> None:
    root = tmp_path / "documents"
    root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    (root / "inside.md").write_text("inside", encoding="utf-8")
    (root / "escaped.md").symlink_to(outside)

    source = LocalFileSource(root, allowed_extensions=(".md",))

    assert [item.file_path for item in source.discover()] == ["inside.md"]


def test_local_file_source_reads_discovered_text(tmp_path: Path) -> None:
    root = tmp_path / "documents"
    root.mkdir()
    path = root / "notes.txt"
    path.write_text("local source text", encoding="utf-8")
    source = LocalFileSource(root, allowed_extensions=(".txt",))

    assert source.read_text(source.discover()[0]) == "local source text"


def test_source_item_supports_remote_identity_without_a_local_path() -> None:
    item = SourceItem(
        source_type="github_issue",
        source_uri="https://api.github.com/repos/docgrab/docgrab/issues/42",
        repo="docgrab/docgrab",
        file_path="issues/42.md",
    )

    assert item.source_uri.endswith("/issues/42")


@pytest.mark.parametrize(
    "file_path",
    (".", "../outside.md", r"C:\\secret.md", "C:/secret.md", r"\\\\server\\share\\secret.md"),
)
def test_source_item_rejects_unstable_relative_paths(tmp_path: Path, file_path: str) -> None:
    with pytest.raises(ValueError, match="relative path"):
        SourceItem(
            source_type="local_file",
            source_uri="file:///tmp/outside.md",
            file_path=file_path,
        )
