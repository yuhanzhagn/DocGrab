# DocGrab

DocGrab is a local-first Retrieval-Augmented Generation project built with FastAPI, Chroma, and a React TypeScript workbench.

The current version is a working RAG MVP: it ingests local documents, chunks them with citation metadata, stores embeddings in Chroma, retrieves relevant chunks, and returns grounded answers with source citations. The next direction is to evolve it into a production-style ingestion and retrieval system with incremental indexing, lexical search, hybrid retrieval, evaluation, and stronger observability.

## Current Capabilities

- Ingest local `.md`, `.txt`, and `.pdf` files.
- Chunk documents while preserving source and citation metadata.
- Generate embeddings through a provider interface.
- Store vectors and metadata in Chroma.
- Query indexed documents through a FastAPI API.
- Return grounded answers with citations and retrieved chunk inspection.
- Run without external API keys using deterministic local fallback providers.
- Use optional local or external embedding and generation providers.
- Explore the system through a React TypeScript document workbench.
- Start the full local stack with Docker Compose.

## Why This Project Exists

DocGrab is intended to grow beyond a toy "chat with your docs" demo. The goal is to make the repository useful for backend and system design discussion by showing clear boundaries between ingestion, indexing, retrieval, API serving, evaluation, and local operations.

The current code favors a small, testable MVP. Future work should keep that discipline: add production-oriented features in commit-sized steps, keep the existing workflow working, and avoid adding infrastructure before it has a concrete role.

## Architecture

Current high-level flow:

```text
Local documents
    |
    v
Document loaders
    |
    v
Text chunker
    |
    v
Embedding provider
    |
    v
Chroma vector store
    |
    v
Retriever
    |
    v
Answer generator
    |
    v
FastAPI response with citations
```

Core modules:

- `src/rag/loaders/`: text, Markdown, and PDF loading
- `src/rag/chunkers/`: chunking logic
- `src/rag/embeddings/`: hash, local, and external embedding providers
- `src/rag/vectorstores/`: Chroma and in-memory vector stores
- `src/rag/retrieval/`: vector retrieval and deduplication
- `src/rag/generation/`: grounded answer generation
- `src/rag/services/`: indexing and query orchestration
- `src/rag/use_cases.py`: framework-agnostic use-case functions
- `src/rag/api/`: FastAPI routes and dependency wiring
- `frontend/`: React TypeScript workbench

## Provider Modes

Embedding providers:

- `hash`: deterministic fallback, no external dependencies
- `local`: in-process local embeddings through `sentence-transformers`
- `external`: OpenAI-compatible external embedding API

Generation providers:

- `simple`: deterministic fallback grounded answer generator
- `local`: in-process local generation through `transformers`
- `ollama`: local model server through the Compose `model` service
- `external`: OpenAI-compatible external generation API

The default mode is intentionally lightweight:

```bash
export EMBEDDER_PROVIDER=hash
export GENERATOR_PROVIDER=simple
```

This lets the project run locally without external credentials or heavyweight model downloads.

## Requirements

- Python `3.11+`
- Node.js `20+` for frontend development
- Docker and Docker Compose for the containerized stack

## Local Python Setup

Create a virtual environment and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Start the API:

```bash
uvicorn rag.main:app --host 0.0.0.0 --port 8000
```

Check health:

```bash
curl -fsS http://localhost:8000/api/health
```

Expected response:

```json
{"status":"ok"}
```

## Optional Local Model Dependencies

Install optional embedding and generation dependencies only when needed:

```bash
pip install -e ".[local-embeddings]"
pip install -e ".[local-generation]"
```

Example local provider configuration:

```bash
export EMBEDDER_PROVIDER=local
export EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
export GENERATOR_PROVIDER=local
export GENERATOR_MODEL_NAME=google/flan-t5-small
```

## External Provider Configuration

External providers use OpenAI-compatible API shapes.

```bash
export EMBEDDER_PROVIDER=external
export EMBEDDING_MODEL_NAME=text-embedding-3-small
export EXTERNAL_EMBEDDING_API_KEY=your-embedding-key

export GENERATOR_PROVIDER=external
export GENERATOR_MODEL_NAME=gpt-4o-mini
export EXTERNAL_GENERATOR_API_KEY=your-generator-key
```

Do not commit real API keys.

## Frontend Setup

Install frontend dependencies and start the Vite dev server:

```bash
cd frontend
npm install
npm run dev
```

The React app runs at:

```text
http://localhost:5173/
```

The Vite dev server proxies `/api` requests to `http://127.0.0.1:8000`.

## Docker Startup

The default Docker stack starts:

- `app`: FastAPI backend
- `web`: React frontend served by nginx
- `chroma`: vector database

Start the default stack:

```bash
docker compose up --build -d
```

Or use the Make target:

```bash
make up
```

Default local URLs:

- FastAPI API: `http://localhost:8000`
- Chroma: `http://localhost:8001`
- React frontend: `http://localhost:8010`

Check service status:

```bash
docker compose ps
```

Stop the stack:

```bash
make down
```

## Ollama Mode

The Compose stack includes an optional Ollama profile for local model serving.

Start the app with Qwen 2.5 1.5B through Ollama:

```bash
make llm-up
```

Override the model:

```bash
make llm-up LLM_MODEL=qwen2.5:3b
```

Before using this mode on Docker Desktop, allocate enough memory for the selected model.

## API Usage

Base URL:

```text
http://localhost:8000/api
```

### Health

```bash
curl -fsS http://localhost:8000/api/health
```

### Ingest Documents

Index a local directory:

```bash
curl -X POST http://localhost:8000/api/documents/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "directory": "/absolute/path/to/your/documents"
  }'
```

Example response:

```json
{
  "indexed_documents": 2,
  "indexed_chunks": 5,
  "skipped_files": []
}
```

### Query Documents

Ask a question:

```bash
curl -X POST http://localhost:8000/api/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Which database stores document embeddings?",
    "top_k": 3
  }'
```

Filter to one source path:

```bash
curl -X POST http://localhost:8000/api/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Which database stores document embeddings?",
    "top_k": 3,
    "source_path": "/absolute/path/to/architecture.md"
  }'
```

The response contains:

- `answer_text`
- `citations`
- `retrieved_chunks`
- chunk scores and source metadata

## Framework-Agnostic Usage

The core pipeline can be called directly from Python without FastAPI:

```python
from rag.api.deps import get_indexing_service, get_query_service
from rag.use_cases import answer_query, ingest_directory

ingest_result = ingest_directory(
    indexing_service=get_indexing_service(),
    directory="/absolute/path/to/documents",
)

answer = answer_query(
    query_service=get_query_service(),
    query="Which database stores document embeddings?",
    top_k=3,
)
```

## Environment Variables

Copy `.env.example` to the ignored `.env` file when local overrides or provider
credentials are needed. Docker Compose defaults remain usable without an `.env`
file.

Common runtime settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_HOST` | `0.0.0.0` | API bind host |
| `APP_PORT` | `8000` | API bind port |
| `WEB_EXTERNAL_PORT` | `8010` | Host port for the React frontend |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:5173,http://localhost:8010` | Browser origins allowed by FastAPI |
| `CHROMA_HOST` | unset locally / `chroma` in Compose | Remote Chroma host |
| `CHROMA_PORT` | `8000` | Remote Chroma port |
| `CHROMA_SSL` | `false` | Use HTTPS for Chroma HTTP client |
| `CHROMA_COLLECTION_NAME` | `documents` | Chroma collection name |
| `EMBEDDER_PROVIDER` | `hash` | `hash`, `local`, or `external` |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model identifier |
| `EMBEDDING_DIMENSION` | `256` | Hash embedder dimension |
| `GENERATOR_PROVIDER` | `simple` | `simple`, `local`, `ollama`, or `external` |
| `GENERATOR_MODEL_NAME` | `google/flan-t5-small` | Generator model identifier |
| `LOCAL_GENERATOR_MAX_NEW_TOKENS` | `160` | Local generator output cap |
| `EXTERNAL_EMBEDDING_API_KEY` | unset | Required for external embeddings |
| `EXTERNAL_EMBEDDING_BASE_URL` | `https://api.openai.com/v1` | External embedding API base URL |
| `EXTERNAL_GENERATOR_API_KEY` | unset | Required for external generation |
| `EXTERNAL_GENERATOR_BASE_URL` | `https://api.openai.com/v1` | External generator API base URL |
| `EXTERNAL_GENERATOR_TIMEOUT_SECONDS` | `30.0` | External generator timeout |
| `EXTERNAL_GENERATOR_TEMPERATURE` | `0.0` | External generator temperature |
| `LOCAL_MODEL_ENDPOINT` | unset | Ollama endpoint used by `GENERATOR_PROVIDER=ollama` |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Model pulled by the Compose `model-init` service |
| `OLLAMA_KEEP_ALIVE` | `24h` | How long Ollama keeps the loaded model warm |
| `CHUNK_SIZE` | `800` | Chunk size |
| `CHUNK_OVERLAP` | `120` | Chunk overlap |

## Testing

Run the full test suite:

```bash
pytest -q
```

Run focused subsets:

```bash
pytest tests/unit -q
pytest tests/integration -q
```

Build the frontend:

```bash
cd frontend
npm run build
```

## Future Targets

The next engineering targets are intentionally staged:

- Separate offline ingestion from the online query API.
- Add richer metadata with stable content hashes.
- Support hash-based incremental indexing.
- Add an embedding cache keyed by content hash and embedding model.
- Add retry, backoff, and rate limiting around external embedding calls.
- Add structure-aware Markdown and source-code chunking.
- Generate ingestion reports and checkpoint resumable runs.
- Add BM25 lexical search.
- Add hybrid retrieval with Reciprocal Rank Fusion.
- Add retrieval evaluation with Recall@K and MRR.
- Keep a future Go query service as a later boundary once ingestion artifacts are stable.

These are future targets, not current claims. The current implementation remains a Python/FastAPI RAG MVP with Chroma-backed vector retrieval.

## Current Limitations

- Chunking is simple and not yet fully structure-aware.
- Incremental indexing and embedding cache are not implemented yet.
- Retrieval is vector-only; BM25 and hybrid retrieval are planned.
- Evaluation datasets and retrieval metrics are planned.
- PDF support depends on text extraction quality; scanned/image-only PDFs are not handled.
- Local model modes may be slow or memory-heavy on smaller machines.

## Project Philosophy

Prefer small, reviewable changes over broad rewrites.

Good next commits should add one clear capability at a time, keep tests passing, and preserve the existing local developer workflow.
