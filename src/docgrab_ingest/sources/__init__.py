"""Source discovery primitives for offline ingestion."""

from docgrab_ingest.sources.base import SourceItem, SourceLoader
from docgrab_ingest.sources.local_files import LocalFileSource

__all__ = ["LocalFileSource", "SourceItem", "SourceLoader"]
