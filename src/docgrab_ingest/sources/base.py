from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from urllib.parse import urlsplit

from docgrab_ingest.paths import normalize_relative_file_path


@dataclass(frozen=True, slots=True)
class SourceItem:
    """A discovered source with stable identity independent of materialization."""

    source_type: str
    source_uri: str
    file_path: str
    repo: str | None = None

    def __post_init__(self) -> None:
        source_type = self.source_type.strip()
        if not source_type:
            raise ValueError("source_type must not be blank")

        source_uri = self.source_uri.strip()
        if not source_uri or not urlsplit(source_uri).scheme:
            raise ValueError("source_uri must be a non-empty URI")

        file_path = normalize_relative_file_path(self.file_path)

        if self.repo is not None:
            repo = self.repo.strip()
            if not repo:
                raise ValueError("repo must not be blank")
            object.__setattr__(self, "repo", repo)

        object.__setattr__(self, "source_type", source_type)
        object.__setattr__(self, "source_uri", source_uri)
        object.__setattr__(self, "file_path", file_path)


class SourceLoader(ABC):
    """Discovers source items without reading or parsing their contents."""

    @abstractmethod
    def discover(self) -> tuple[SourceItem, ...]:
        raise NotImplementedError

    @abstractmethod
    def read_text(self, item: SourceItem) -> str:
        raise NotImplementedError
