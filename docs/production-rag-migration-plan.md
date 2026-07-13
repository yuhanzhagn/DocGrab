# Production RAG Migration Plan

This document captures the audit and staged migration plan for upgrading this repository from a local RAG MVP into a production-style backend/system engineering project suitable for resume and interview discussion.

The guiding constraint is to keep the working MVP intact while adding production-oriented capabilities in small, reviewable commits.

## Implemented Migration Progress

- Step 1 is complete: `src/docgrab_ingest/` now contains versioned ingestion document and chunk models plus deterministic content hashing helpers.
- Hash payloads use version `v2`: line endings are canonicalized, but all other whitespace is preserved because it can be meaningful Markdown content. The persisted document and chunk schemas are also `v2` and carry `hash_version="v2"`; existing `v1` hashes are incompatible and must be rebuilt before use.
- Documents and chunks verify their supplied hashes. Chunk occurrence IDs use source identity, content hash, and a duplicate occurrence ordinal, so duplicate text cannot overwrite another occurrence and earlier edits with distinct chunk content do not churn unchanged IDs.
- Persisted file paths are canonical portable relative paths, and document metadata accepts recursive JSON values only.
- Step 2 is complete: `src/docgrab_ingest/sources/` now defines source discovery contracts and a local-file adapter with deterministic root-relative paths, extension filtering, and source-root containment checks.
- The existing `src/rag/` FastAPI RAG path remains unchanged.
- The next incomplete step is structure-aware Markdown parsing and chunking.

## Current Repository Audit

### Current Purpose

The repository is a local-first Retrieval-Augmented Generation MVP built with FastAPI, Chroma, and a React TypeScript workbench.

It currently demonstrates:

- document ingestion for Markdown, text, and PDF files
- simple text chunking with citation metadata
- provider-based embeddings
- provider-based grounded answer generation
- Chroma-backed vector storage
- FastAPI query and ingestion endpoints
- framework-agnostic use-case functions
- a React document retrieval workbench
- Docker Compose startup for the backend, frontend, Chroma, and optional Ollama model service

### Current Folder Structure

Important areas:

- `src/rag/loaders/`: file loading for text, Markdown, and PDF
- `src/rag/chunkers/`: chunking logic
- `src/rag/embeddings/`: hash, local, and external embedding providers
- `src/rag/vectorstores/`: Chroma and in-memory vector store implementations
- `src/rag/retrieval/`: vector retrieval and deduplication
- `src/rag/generation/`: answer generation providers and grounded response building
- `src/rag/services/`: indexing and query orchestration
- `src/rag/api/`: FastAPI routes and dependency wiring
- `frontend/`: React TypeScript document workbench
- `tests/`: unit and integration tests
- `data/sample_docs/`: sample Markdown and text documents
- `docker-compose.yml`: app, frontend, Chroma, and optional model services

### Current RAG Flow

1. A client calls the ingestion API with a local directory.
2. `IndexingService` walks the directory.
3. `MultiDocumentLoader` dispatches to a text or PDF loader.
4. `SimpleTextChunker` chunks the document and attaches citation metadata.
5. The configured embedder embeds chunk text.
6. `ChromaVectorStore` upserts chunk vectors and metadata.
7. A client calls the query API.
8. `Retriever` embeds the query and retrieves vector matches.
9. The retriever deduplicates near-identical chunks.
10. The configured generator builds a grounded answer and citations.

### Current Dependencies

Python runtime dependencies:

- `fastapi`
- `uvicorn[standard]`
- `pydantic`
- `pydantic-settings`
- `chromadb`
- `httpx`
- `pypdf`
- `pytest`

Optional Python dependencies:

- `sentence-transformers` for local embeddings
- `transformers` for local generation

Frontend dependencies:

- React
- TypeScript
- Vite
- lucide-react
- ESLint

### Current Data Sources

Currently supported:

- local `.md` files
- local `.txt` files
- local `.pdf` files

Not yet supported:

- GitHub repositories
- source code repositories as structured sources
- GitHub issues or pull requests
- HTML
- scanned PDFs or image-only documents

### Current Vector DB And Embedding Usage

Vector storage:

- Chroma through `ChromaVectorStore`
- in-memory vector store for tests

Embedding providers:

- deterministic hash embedder
- local sentence-transformers embedder
- external OpenAI-compatible embedding endpoint

The default mode is intentionally lightweight and offline:

- `EMBEDDER_PROVIDER=hash`
- `GENERATOR_PROVIDER=simple`

### Current API Layer

FastAPI exposes:

- health routes
- document ingestion route
- query route

The query route accepts:

- query text
- `top_k`
- optional source path filter

The response includes:

- answer text
- citations
- retrieved chunk metadata

### Current Tests

The current test suite covers:

- text loader behavior
- PDF loader behavior
- chunking behavior
- hash embedder behavior
- embedder factory behavior
- generator factory behavior
- vector store behavior
- retriever deduplication and path filtering
- health route behavior
- use-case behavior
- API ingestion and query flow

Baseline at audit time:

```bash
pytest -q
```

Result:

```text
42 passed
```

### Current Weaknesses

- Offline ingestion and online query service are coupled inside the same `rag` package.
- Chunk IDs are generated from document ID, chunk index, and character offsets rather than content hashes.
- A small edit near the top of a file can churn downstream chunk IDs.
- There is no hash-based incremental indexing.
- There is no embedding cache keyed by content hash and embedding model.
- There is no checkpointing or resumable ingestion.
- There is no ingestion report artifact.
- There is no BM25 lexical index.
- There is no hybrid retrieval or RRF fusion.
- There is no retrieval evaluation dataset or metrics runner.
- External embeddings do not yet have retry, backoff, or rate limiting.
- There is no Go online service yet.
- Generated bytecode artifacts appear in the repository tree and should be cleaned from version control if tracked.

### What Is Reusable

Reusable as-is or with small adaptations:

- loader abstraction
- chunker abstraction
- embedder abstraction
- vector store abstraction
- in-memory store for tests
- Chroma integration
- grounded answer builder
- use-case functions
- FastAPI route shape
- Docker Compose Chroma setup
- existing React workbench for manual inspection

### What Should Be Refactored

Refactor gradually:

- split offline ingestion into a dedicated `docgrab_ingest` package
- keep online query/API code in `rag` until a Go service exists
- extract stable content hashing primitives
- introduce explicit ingestion metadata models
- add interfaces for source loading, parsing, chunking, embedding, index writing, BM25 building, and metadata storage
- add a CLI around ingestion configs after interfaces settle
- add lexical and hybrid retrieval after chunk metadata and index artifacts are reliable

## Target Architecture

### High-Level Direction

Python should own offline ingestion and evaluation.

Go should eventually own the online query service once the ingestion artifacts and retrieval contracts are stable.

The intended architecture is:

```text
Local files / repos / GitHub data
        |
        v
Python ingestion pipeline
        |
        +--> cleaned documents
        +--> structured chunks
        +--> metadata store
        +--> embedding cache
        +--> vector index
        +--> BM25 index
        +--> ingestion report
        |
        v
Evaluation scripts
        |
        v
Online query service
        |
        +--> vector search
        +--> BM25 search
        +--> RRF fusion
        +--> optional reranking
        +--> context building
        +--> streamed answer
```

### Python Ingestion Modules

Proposed package:

```text
src/docgrab_ingest/
  __init__.py
  config.py
  hashing.py
  models.py
  paths.py
  sources/
  parsers/
  chunking/
  embeddings/
  indexing/
  metadata/
  pipeline/
  eval/
```

Responsibilities:

- load source material
- parse documents into normalized internal models
- chunk Markdown and code using structure-aware logic
- generate stable metadata
- compute content hashes
- cache embeddings
- write vector index records
- build BM25 index artifacts
- persist metadata
- write ingestion reports
- run retrieval evaluation

### Go Online Service Modules

Do not add Go immediately. There is no Go code today, and adding a second runtime before the index artifacts are stable would expand scope too early.

When ready, add:

```text
services/query-api/
  go.mod
  cmd/server/main.go
  internal/config/
  internal/http/
  internal/retrieval/
  internal/rrf/
  internal/session/
  internal/llm/
  internal/observability/
```

Eventual responsibilities:

- `/query` endpoint
- SSE streaming
- request timeout and cancellation
- vector search
- BM25 search
- hybrid retrieval
- RRF fusion
- Redis-backed session memory
- clean interfaces and unit tests
- request-level observability

### Storage Choices

Short-term:

- Chroma for vector storage
- local JSONL or SQLite for metadata
- local disk files for embedding cache
- local BM25 artifact files
- local YAML config files

Medium-term:

- SQLite metadata store for local development and deterministic tests
- Chroma or Qdrant for vector storage
- Tantivy-style or SQLite FTS strategy for lexical search if Python-only
- Go service reads stable index artifacts or calls storage backends directly

Avoid adding distributed storage before it is needed.

### Vector DB Integration

Keep Chroma as the first production-style vector DB integration because it already exists in the project.

Improve integration by adding:

- deterministic record IDs
- content hash metadata
- source metadata
- collection versioning conventions
- write reports
- health checks
- tests around metadata normalization and filters

### BM25 Strategy

Start with a simple local BM25 index builder in Python.

Initial design:

- tokenize chunk text
- store chunk ID to token statistics
- persist the index artifact locally
- expose a search interface returning ranked chunk IDs and scores
- add tests for ranking behavior

Later options:

- SQLite FTS
- Tantivy-compatible service
- Bleve in the Go query service

### Metadata Storage

Start with JSONL or SQLite.

Metadata should include:

- `source_type`
- `repo`
- `file_path`
- `heading_path`
- `symbol_name`
- `chunk_index`
- `occurrence_ordinal`
- `content_hash`
- `document_hash`
- `hash_version`
- `embedding_model`
- `created_at`
- `parser_version`
- `chunker_version`

SQLite is a good near-term choice once incremental indexing begins because it supports:

- lookup by content hash
- ingestion run records
- source file state
- chunk state
- simple local inspection
- deterministic tests

### Config Management

Add YAML ingestion configs under `configs/`.

Example future command:

```bash
python -m docgrab_ingest.pipeline.ingest --config configs/django.yaml
```

Config sections:

- source settings
- parser settings
- chunker settings
- embedding settings
- vector index settings
- BM25 settings
- metadata store settings
- output/report settings

### Testing Strategy

Keep the current API and vector tests.

Add focused tests for each new ingestion primitive:

- content hash stability
- metadata generation
- Markdown chunking
- source code chunking
- embedding cache hit/miss behavior
- incremental indexing decisions
- BM25 ranking
- RRF fusion
- ingestion report contents
- evaluation metrics

Use test doubles for embedders and stores. Avoid tests that require live external APIs.

### Local Development Workflow

Current workflow remains valid:

```bash
pytest -q
uvicorn rag.main:app --host 0.0.0.0 --port 8000
docker compose up --build -d
```

Future workflow:

```bash
python -m docgrab_ingest.pipeline.ingest --config configs/sample_ingest.yaml
python -m docgrab_ingest.eval.run_eval --dataset eval/sample_queries.yaml
pytest -q
```

### Docker Compose Plan

Keep existing services:

- `app`
- `web`
- `chroma`
- optional `model`

Add later:

- optional `redis` for session memory
- optional `query-api` Go service
- optional `ingest` profile for batch ingestion jobs

Do not add these services until the corresponding code exists.

## Python Ingestion Refactor Plan

### Proposed Package Structure

```text
src/docgrab_ingest/
  __init__.py
  config.py
  hashing.py
  models.py
  sources/
    __init__.py
    base.py
    local_files.py
    github_repo.py
  parsers/
    __init__.py
    base.py
    markdown.py
    code.py
    plain_text.py
  chunking/
    __init__.py
    base.py
    markdown.py
    code.py
  embeddings/
    __init__.py
    base.py
    cache.py
    retrying.py
  indexing/
    __init__.py
    vector.py
    bm25.py
  metadata/
    __init__.py
    store.py
  pipeline/
    __init__.py
    ingest.py
    report.py
    checkpoint.py
  eval/
    __init__.py
    metrics.py
    run_eval.py
```

### Core Interfaces

`SourceLoader`:

- discovers source items
- returns source records with stable source metadata
- examples: local files, GitHub repo checkout, GitHub issues later

`Parser`:

- converts raw source content into parsed documents
- normalizes text
- preserves structural information such as headings or symbols

`Chunker`:

- converts parsed documents into chunks
- assigns stable chunk metadata
- computes chunk content hashes

`Embedder`:

- embeds text in batches
- exposes model identity
- may be wrapped by retry/rate-limit/cache decorators

`VectorIndexWriter`:

- writes chunk embeddings and metadata to a vector store
- should not know how content was loaded or parsed

`BM25IndexBuilder`:

- builds lexical index artifacts from chunk text
- exposes search-time metadata needed for evaluation and hybrid retrieval

`MetadataStore`:

- persists source, document, chunk, and ingestion run state
- supports incremental indexing decisions

## Engineering Features Roadmap

### Structure-Aware Chunking

Markdown chunking should preserve:

- heading path
- section text boundaries
- source file path
- chunk index within section or file

Source code chunking should preserve:

- language
- symbol name where feasible
- file path
- byte or character offsets
- import/package context when useful

Start with Markdown. Add source code after metadata and hashing are stable.

### Chunk Metadata

Target metadata:

```text
source_type
repo
file_path
heading_path
symbol_name
chunk_index
occurrence_ordinal
content_hash
document_hash
hash_version
embedding_model
parser_version
chunker_version
start_char
end_char
```

### Hash-Based Incremental Indexing

Use:

- `document_hash` for entire source content
- `content_hash` for normalized chunk text plus selected structural metadata
- `chunk_id` for source-scoped content identity plus `occurrence_ordinal`; it intentionally excludes document-wide hashes and absolute offsets
- `embedding_cache_key = content_hash + embedding_model`

Exact duplicate chunks are intentionally disambiguated by occurrence ordinal. Inserting another identical chunk before an existing duplicate can therefore change that duplicate's ID; a later parser may adopt a more contextual duplicate policy if this churn becomes material.

Indexing decisions:

- unchanged source: skip parsing and chunking when safe
- unchanged chunk: reuse embedding
- changed chunk: regenerate embedding and upsert
- deleted source: mark stale records and remove or tombstone chunks

### Embedding Cache

Initial cache:

- local disk or SQLite
- keyed by content hash and embedding model
- stores vector, dimension, created time, and provider metadata

Tests:

- cache hit avoids embedder call
- model change invalidates cache
- content change invalidates cache

### Batch Embedding

Embedder interface should support:

- `embed_texts(texts: list[str])`
- configurable batch size
- deterministic behavior for tests

External provider wrappers should add:

- retries
- backoff
- rate limiting
- clear error messages

### Checkpointing

Checkpoint after each source or batch:

- source path
- source hash
- processed chunk count
- indexed chunk count
- skipped reason
- errors

The first implementation can be a simple JSONL report plus metadata store state.

### Ingestion Reports

Report fields:

- run ID
- started and finished timestamps
- config path
- source count
- parsed document count
- chunk count
- embedded count
- cache hits
- cache misses
- skipped files
- failed files
- deleted or stale chunks

### Retrieval Evaluation

Add dataset format:

```yaml
queries:
  - id: q1
    query: Which database stores document embeddings?
    relevant_chunk_ids:
      - chunk-id
    relevant_source_paths:
      - data/sample_docs/architecture.md
```

Metrics:

- Recall@K
- MRR
- optional Precision@K later

Compare:

- vector-only retrieval
- BM25-only retrieval
- hybrid retrieval

## Retrieval Strategy

### Vector Search

Existing vector search should remain the baseline.

Improve by:

- stable IDs
- richer metadata filters
- score normalization discipline
- evaluation coverage

### BM25 Search

Add a lexical retriever that returns:

- chunk ID
- score
- source path
- metadata

Use it first in Python evaluation before the Go service exists.

### RRF Fusion

Implement reciprocal rank fusion over ranked lists:

```text
score(document) = sum(1 / (k + rank))
```

Start with default `k = 60`.

Tests:

- documents present in both lists rise above single-list documents
- empty inputs are handled
- ties are deterministic

### Optional Reranking

Do not implement reranking first.

Add it later only after:

- vector/BM25/RRF evaluation exists
- there is a measurable quality problem
- a small reranker interface can be tested without live external calls

## Go Service Plan

### Recommendation

Do not implement Go immediately.

Reason:

- there is no Go code in the repo today
- the ingestion artifacts are not yet stable
- adding a Go service now would create a second runtime without clear contracts

### Future Structure

```text
services/query-api/
  cmd/server/main.go
  internal/config/
  internal/http/
  internal/retrieval/
  internal/rrf/
  internal/session/
  internal/llm/
  internal/observability/
```

### Future Capabilities

- `/query` endpoint
- SSE streaming
- request timeout and context cancellation
- vector search client
- BM25 search client or local index reader
- RRF fusion
- context builder
- Redis session memory
- request-level structured logging
- unit tests around retrieval, RRF, timeout behavior, and HTTP handlers

## Recommended First Commits

### Commit 1: Typed Ingestion Models And Stable Content Hashing

Files:

- `src/docgrab_ingest/__init__.py`
- `src/docgrab_ingest/hashing.py`
- `src/docgrab_ingest/models.py`
- `tests/unit/test_docgrab_ingest_hashing.py`
- `tests/unit/test_docgrab_ingest_models.py`

Purpose:

- add stable content hash primitives
- formalize versioned ingestion document and chunk contracts
- validate target chunk metadata fields
- validate document and chunk hash integrity, with a separate deterministic chunk occurrence ID
- avoid touching existing runtime behavior

Status:

- complete

### Commit 2: Ingestion Interfaces And Local-Source Adapters

Files:

- `src/docgrab_ingest/sources/__init__.py`
- `src/docgrab_ingest/sources/base.py`
- `src/docgrab_ingest/sources/local_files.py`
- `tests/unit/test_docgrab_ingest_local_files.py`

Purpose:

- define source discovery contracts
- add local file source records without changing FastAPI ingestion
- keep parsing and chunking out of this step

Status:

- complete

### Commit 3: Markdown Structure-Aware Chunker

Files:

- `src/docgrab_ingest/parsers/markdown.py`
- `src/docgrab_ingest/chunking/markdown.py`
- `tests/unit/test_markdown_chunking.py`

Purpose:

- preserve heading paths
- improve chunk explainability
- avoid changing existing `SimpleTextChunker` until the new path is proven

### Commit 4: In-Process Ingestion Orchestration Seam

Files:

- `src/docgrab_ingest/pipeline/base.py`
- `tests/unit/test_ingestion_orchestrator.py`

Purpose:

- compose source discovery, parsing, and chunking in-process without a CLI or persistence
- integration-test component contracts before additional ingestion stages accumulate

### Commit 5: Source-Code Parsing And Chunking

Files:

- `src/docgrab_ingest/parsers/source_code.py`
- `src/docgrab_ingest/chunking/source_code.py`
- `tests/unit/test_source_code_chunking.py`

Purpose:

- preserve source-code symbol boundaries and paths
- keep code-specific parsing separate from Markdown parsing

### Commit 6: SQLite Metadata Manifest And Change Planning

Files:

- `src/docgrab_ingest/metadata/store.py`
- `src/docgrab_ingest/metadata/change_planner.py`
- `tests/unit/test_incremental_indexing.py`

Purpose:

- track source and chunk state
- decide add, change, and delete work explicitly

### Commit 7: Embedding Cache And Resilient Batch Embedding

Files:

- `src/docgrab_ingest/embeddings/cache.py`
- `src/docgrab_ingest/embeddings/retrying.py`
- `tests/unit/test_embedding_cache.py`
- `tests/unit/test_embedding_retrying.py`

Purpose:

- cache embeddings by `content_hash + embedding_model`
- define retry classification and retry exhaustion behavior

### Commit 8: Checkpointing And Ingestion Reports

Files:

- `src/docgrab_ingest/pipeline/checkpoint.py`
- `src/docgrab_ingest/pipeline/report.py`
- `tests/unit/test_checkpoint.py`
- `tests/unit/test_ingestion_report.py`

Purpose:

- resume interrupted ingestion work
- report ingestion totals and deferred failures

### Commit 9: BM25 Artifact Generation And Lexical Search

Files:

- `src/docgrab_ingest/indexing/bm25.py`
- `tests/unit/test_bm25_index.py`

Purpose:

- add lexical retrieval artifacts
- enable comparison against vector-only retrieval

### Commit 10: RRF Fusion And Retrieval Evaluation

Files:

- `src/docgrab_ingest/eval/metrics.py`
- `src/docgrab_ingest/eval/run_eval.py`
- `tests/unit/test_rrf.py`
- `tests/unit/test_eval_metrics.py`
- `eval/sample_queries.yaml`

Purpose:

- evaluate Recall@K and MRR
- compare vector, BM25, and hybrid retrieval

### Commit 11: Stable Persisted Retrieval Artifact Contracts

Files:

- `docs/retrieval-artifact-contract.md`
- `tests/unit/test_retrieval_artifact_contract.py`

Purpose:

- version portable persisted retrieval records and index metadata
- establish a compatibility boundary before Go work

### Commit 12: Runnable Offline Ingestion Pipeline And Configuration

Files:

- `src/docgrab_ingest/config.py`
- `src/docgrab_ingest/pipeline/ingest.py`
- `configs/sample_ingest.yaml`
- `tests/unit/test_ingest_config.py`
- `tests/integration/test_ingest_pipeline.py`

Purpose:

- connect the implemented ingestion components into an end-to-end offline workflow
- validate configuration and expose the documented `python -m docgrab_ingest.pipeline.ingest` command
- write artifacts that conform to the stable persisted retrieval contracts

### Commit 13: Go Query-Service Skeleton

Files:

- `services/query-api/go.mod`
- `services/query-api/cmd/server/main.go`
- `services/query-api/internal/config/config.go`

Purpose:

- establish the Go service layout after artifact contracts stabilize

### Commit 14: Go Request Handling, Cancellation, Observability, SSE, And Sessions

Files:

- `services/query-api/internal/http/`
- `services/query-api/internal/retrieval/`
- `services/query-api/internal/session/`

Purpose:

- implement online request behavior against stable retrieval artifacts

## Next File Changes Before Coding

The next implementation step should change only Markdown parsing and structure-aware chunking.

Proposed changes:

1. Add `src/docgrab_ingest/parsers/markdown.py`
   - parses Markdown into normalized sections with heading paths

2. Add `src/docgrab_ingest/chunking/markdown.py`
   - splits parsed Markdown sections into stable chunk boundaries

3. Add `tests/unit/test_markdown_chunking.py`
   - covers heading paths, character offsets, and deterministic chunk hashes

## Quality Gates

Every commit should keep:

```bash
pytest -q
```

passing.

Additional checks when touching frontend:

```bash
cd frontend
npm run build
```

Avoid live external API tests in CI. Use test doubles for provider behavior.

## Resume-Oriented Project Highlights

Once the staged work is implemented, the project can honestly demonstrate:

- offline ingestion pipeline design
- source-specific parsing
- structure-aware chunking
- stable metadata design
- hash-based incremental indexing
- embedding cache design
- batch embedding with retry and rate limiting
- vector and lexical retrieval
- hybrid retrieval with RRF
- retrieval quality evaluation
- ingestion observability and reports
- clean separation between offline ingestion and online query serving
- future Go service boundary for low-latency query APIs

## Explicit Non-Goals For Early Stages

Avoid early complexity from:

- adding Kubernetes
- adding multiple vector databases
- adding reranking before evaluation exists
- adding a Go service before index contracts stabilize
- adding GitHub issues and pull requests before local files and repos are solid
- claiming performance improvements without benchmarks
- hardcoding API keys

## Next Step

Implement Commit 3 only:

- Markdown parsing
- structure-aware Markdown chunking
- focused tests for heading paths and stable chunk boundaries

Do not begin source-code parsing, metadata persistence, or embedding cache work in the same commit.
