from pathlib import Path

from rag.chunkers.base import Chunker
from rag.embeddings.base import Embedder
from rag.loaders.base import DocumentLoader
from rag.schemas.api import IngestResponse
from rag.schemas.document import IndexedRecord
from rag.vectorstores.base import VectorStore


class IndexingService:
    def __init__(
        self,
        loader: DocumentLoader,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
        allowed_extensions: tuple[str, ...],
        allowed_root: Path | None = None,
    ) -> None:
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.allowed_extensions = {extension.lower() for extension in allowed_extensions}
        self.allowed_root = allowed_root.expanduser().resolve() if allowed_root else None

    def ingest_directory(self, directory: str) -> IngestResponse:
        root = Path(directory).expanduser().resolve()
        self._ensure_allowed_root(root)
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {root}")

        indexed_documents = 0
        indexed_chunks = 0
        skipped_files: list[str] = []

        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in self.allowed_extensions:
                skipped_files.append(str(path))
                continue
            if not self.loader.supports(path):
                skipped_files.append(str(path))
                continue

            document = self.loader.load(path)
            chunks = self.chunker.chunk(document)
            if not chunks:
                skipped_files.append(str(path))
                continue

            embeddings = self.embedder.embed_texts([chunk.text for chunk in chunks])
            records = [
                IndexedRecord(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    source_path=document.source_path,
                    text=chunk.text,
                    embedding=embedding,
                    metadata=chunk.metadata,
                )
                for chunk, embedding in zip(chunks, embeddings)
            ]
            self.vector_store.upsert(records)
            indexed_documents += 1
            indexed_chunks += len(records)

        return IngestResponse(
            indexed_documents=indexed_documents,
            indexed_chunks=indexed_chunks,
            skipped_files=skipped_files,
        )

    def _ensure_allowed_root(self, root: Path) -> None:
        if self.allowed_root is None:
            return
        if not root.is_relative_to(self.allowed_root):
            raise PermissionError(
                f"Directory is outside allowed data directory: {root}. "
                f"Allowed root: {self.allowed_root}"
            )
