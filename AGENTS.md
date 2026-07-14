# DocGrab Agent Instructions

These instructions apply to the entire repository.

## Source Of Truth

Before implementing production-RAG work, read:

1. `README.md`
2. `docs/production-rag-migration-plan.md`
3. The relevant current implementation and tests.

The migration plan describes intended architecture, not implemented behavior. Do not present roadmap items as existing features.

## Core Constraints

- Preserve the working Python/FastAPI RAG flow unless the task explicitly changes it.
- Keep offline ingestion separate from online query serving.
- Add new ingestion work under `src/docgrab_ingest/`.
- Do not move the existing `src/rag/` package wholesale.
- Prefer one capability per change.
- Do not introduce Go until persisted retrieval contracts are stable.
- Do not add infrastructure without code that uses it.
- Do not hardcode credentials or commit secrets.
- Do not claim performance or quality gains without measurements.

## Required Workflow

For every implementation task:

1. Inspect the affected implementation, tests, configuration, and documentation.
2. State the proposed files and purpose of each change before editing.
3. Select the smallest incomplete migration step.
4. Add or update tests with the implementation.
5. Preserve compatibility with existing API behavior unless a migration is explicit.
6. Run the relevant focused tests.
7. Run the full Python test suite.
8. Update documentation only for behavior that now exists.
9. Report implemented behavior, verification results, and remaining limitations.

Do not begin the next migration stage automatically unless requested.

## Migration Order

Implement capabilities in this order unless the task explicitly requires otherwise:

1. [x] Typed ingestion models and stable content hashing.
2. [x] Ingestion interfaces and local-source adapters.
3. [ ] Structure-aware Markdown parsing and chunking.
4. [ ] In-process ingestion orchestration seam.
5. [ ] Source-code parsing and chunking.
6. [ ] SQLite metadata manifest and change planning.
7. [ ] Embedding cache and resilient batch embedding.
8. [ ] Checkpointing and ingestion reports.
9. [ ] BM25 artifact generation and lexical search.
10. [ ] RRF fusion and retrieval evaluation.
11. [ ] Stable persisted retrieval artifact contracts.
12. [ ] Runnable offline ingestion pipeline and configuration.
13. [ ] Go query-service skeleton.
14. [ ] Go request handling, cancellation, observability, SSE, and sessions.

Current migration progress:

- Step 1 is implemented under `src/docgrab_ingest/` with versioned ingestion document/chunk models and deterministic `sha256:<hex>` content hashing.
- Step 2 is implemented under `src/docgrab_ingest/sources/` with a source-discovery contract and a local-file adapter that emits deterministic root-relative paths and raw source bytes for supported files.
- Step 3 is the next incomplete migration step. Do not begin it unless requested.

A later stage may not silently invent or duplicate contracts owned by an earlier stage.

## Artifact Contracts

Persisted documents and chunks should use explicit, versioned models.

Target chunk metadata includes:

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
- `parser_version`
- `chunker_version`
- `start_char`
- `end_char`

Content hashes must be deterministic and based on normalized content plus the structural metadata that affects retrieval meaning.

Embedding-cache identity must include both `content_hash` and `embedding_model`.

Do not make Python-specific serialized objects the cross-language storage contract. Prefer documented JSON, JSONL, SQLite schemas, or similarly portable formats.

## Compatibility Rules

- Existing FastAPI endpoints remain operational during the Python migration.
- Existing Chroma data must not be destructively migrated without an explicit versioning or rebuild strategy.
- New ingestion components should initially coexist with the current `rag` implementation.
- Deleted sources and stale chunks require an explicit policy; do not leave deletion behavior implicit.
- Changes to identifiers, metadata, index formats, or API schemas require migration notes and focused tests.

## Testing Requirements

Run the full baseline suite with:

```bash
pytest -q
```

Current audited baseline:

```text
85 passed
```

Add focused tests for each introduced behavior, especially:

- hash normalization and stability
- metadata validation and serialization
- Markdown heading paths
- source-code symbol boundaries
- incremental add/change/delete decisions
- cache hit, miss, and model invalidation
- retry classification and retry exhaustion
- BM25 ranking
- deterministic RRF behavior
- Recall@K and MRR
- checkpoint recovery
- ingestion report totals

External services must use test doubles in unit tests.

When frontend files change, also run:

```bash
cd frontend
npm run build
```

When Go is introduced, require formatting, static analysis, and unit tests for every Go change.

## Definition Of Done

A migration step is complete only when:

- its behavior is implemented;
- relevant tests pass;
- the full existing suite passes;
- configuration contains no secrets;
- persisted-format changes are documented;
- README claims match implemented behavior;
- limitations and deferred work are stated clearly.

A directory skeleton or interface alone does not make a feature complete.

## Documentation Discipline

- Keep `README.md` focused on current capabilities and developer usage.
- Keep architecture and staged migration details in `docs/production-rag-migration-plan.md`.
