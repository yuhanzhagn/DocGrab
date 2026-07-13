from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from docgrab_ingest.sources.base import SourceItem, SourceLoader


class LocalFileSource(SourceLoader):
    """Discover supported files below one local source root."""

    SOURCE_TYPE = "local_file"

    def __init__(self, root: str | Path, *, allowed_extensions: Iterable[str]) -> None:
        self.root = Path(root).expanduser()
        self.allowed_extensions = frozenset(
            self._normalize_extension(extension) for extension in allowed_extensions
        )

    def discover(self) -> tuple[SourceItem, ...]:
        root = self._resolve_root()

        items: list[SourceItem] = []
        for path in sorted(root.rglob("*"), key=lambda candidate: candidate.as_posix()):
            if not self._is_supported_file(path, root=root):
                continue

            items.append(
                SourceItem(
                    source_type=self.SOURCE_TYPE,
                    source_uri=path.resolve().as_uri(),
                    file_path=path.relative_to(root).as_posix(),
                )
            )
        return tuple(items)

    def read_text(self, item: SourceItem) -> str:
        root = self._resolve_root()
        if item.source_type != self.SOURCE_TYPE:
            raise ValueError("source item is not a local file")

        path = (root / item.file_path).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.as_uri() != item.source_uri:
            raise ValueError("source item is outside the configured local root")
        return path.read_text(encoding="utf-8")

    def _is_supported_file(self, path: Path, *, root: Path) -> bool:
        if not path.is_file() or path.suffix.lower() not in self.allowed_extensions:
            return False
        return path.resolve().is_relative_to(root)

    def _resolve_root(self) -> Path:
        root = self.root.resolve()
        if not root.exists():
            raise FileNotFoundError(f"Source root not found: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Source root is not a directory: {root}")
        return root

    @staticmethod
    def _normalize_extension(extension: str) -> str:
        normalized = extension.strip().lower()
        if not normalized:
            raise ValueError("allowed extensions must not be blank")
        return normalized if normalized.startswith(".") else f".{normalized}"
