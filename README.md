# RAG MVP

A local-first Retrieval-Augmented Generation MVP built with FastAPI and Chroma.

The project supports:
- document ingestion for `.md`, `.txt`, and `.pdf`
- simple chunking with citation metadata
- provider-based embeddings
- provider-based answer generation
- grounded responses with citations
- FastAPI endpoints
- framework-agnostic use-case functions
- a React TypeScript document workbench
- Docker and Docker Compose local startup

## Overview

This project indexes local documents into Chroma, retrieves the most relevant chunks for a query, and returns a grounded answer with citations.

The default setup is intentionally lightweight:
- embedder: `hash`
- generator: `simple`

That means the app can run locally without external API keys and without a heavyweight local model.

## Architecture

High-level flow:

1. Load documents from disk.
2. Chunk documents while preserving citation metadata.
3. Embed chunks with the configured embedding provider.
4. Store vectors and metadata in Chroma.
5. Retrieve relevant chunks for a query.
6. Generate a grounded answer from retrieved context.
7. Return answer text, citations, and retrieved chunk metadata through the API.

Core modules:
- `src/rag/loaders/`: text and PDF loading
- `src/rag/chunkers/`: chunking logic
- `src/rag/embeddings/`: provider-based embedding layer
- `src/rag/vectorstores/`: Chroma-backed vector storage
- `src/rag/retrieval/`: retrieval logic
- `src/rag/generation/`: provider-based grounded answer generation
- `src/rag/services/`: indexing and query services
- `src/rag/use_cases.py`: framework-agnostic entrypoints
- `src/rag/api/`: FastAPI routes and dependency wiring

## Provider Modes

Embedding providers:
- `hash`: deterministic fallback, no external dependencies
- `local`: in-process local embedding provider using `sentence-transformers`
- `external`: external API-backed embedding provider

Generation providers:
- `simple`: deterministic fallback grounded answer generator
- `local`: in-process local generator using `transformers`
- `ollama`: local model server generator using the Compose `model` service
- `external`: external API-backed generator

Important note:
- Today, `local` provider mode means in-process Python model execution, not a remote model-serving container.
- Use `ollama` provider mode when you want the Python app to call the Dockerized Ollama model service through `LOCAL_MODEL_ENDPOINT`.

## Requirements

- Python `3.11+`
- Node.js `20+` for local frontend development
- Docker and Docker Compose for containerized startup

## Local Python Setup

Create a virtual environment and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Optional local model dependencies:

```bash
pip install -e ".[local-embeddings]"
pip install -e ".[local-generation]"
```

Start the API locally:

```bash
uvicorn rag.main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl -fsS http://localhost:8000/api/health
```

Expected result:

```json
{"status":"ok"}
```

## Local Frontend Setup

Install frontend dependencies and start the Vite dev server:

```bash
cd frontend
npm install
npm run dev
```

The local React app runs at:

```text
http://localhost:5173/
```

The Vite dev server proxies `/api` requests to `http://127.0.0.1:8000`.

## Docker Startup

The default Docker stack starts:
- `app` for the FastAPI backend
- `web` for the React frontend served by nginx
- `chroma` for vector storage

Optional profile:
- `model`

Copy the example environment file if you want to customize settings:

```bash
cp .env.example .env
```

Start the default lightweight stack:

```bash
docker compose up --build -d
```

Or with the helper target:

```bash
make up
```

Start the optional local model service too:

```bash
docker compose --profile local-model up --build -d
```

Start the app with Qwen 2.5 1.5B through Ollama in one command:

```bash
make llm-up
```

Before running this on Docker Desktop, set Docker Desktop's global memory allocation to at least 6 GB. Compose can cap a container, but it cannot increase Docker Desktop's overall memory pool.

This starts the normal app stack plus the Ollama `model` service, pulls `qwen2.5:1.5b`, and configures the API with `GENERATOR_PROVIDER=ollama`.

You can override the model:

```bash
make llm-up LLM_MODEL=qwen2.5:3b
```

Check status:

```bash
docker compose ps
```

FastAPI health:

```bash
curl -fsS http://localhost:8000/api/health
```

Chroma health:

```bash
curl -fsS http://localhost:8001/api/v1/heartbeat
```

React frontend:

```bash
curl -I http://localhost:8010/
```

Default local URLs:
- FastAPI API: `http://localhost:8000`
- Chroma: `http://localhost:8001`
- React frontend: `http://localhost:8010`

Stop the stack:

```bash
docker compose down
```

Or:

```bash
make down
```

## Environment Variables

Common runtime settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_HOST` | `0.0.0.0` | API bind host |
| `APP_PORT` | `8000` | API bind port |
| `WEB_EXTERNAL_PORT` | `8010` | Host port for the React frontend |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:5173,http://localhost:8010` | Comma-separated browser origins allowed by FastAPI |
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

Document ingestion settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CHUNK_SIZE` | `800` | Chunk size |
| `CHUNK_OVERLAP` | `120` | Chunk overlap |

## Configuration Examples

### Default Lightweight Mode

No external credentials required:

```bash
export EMBEDDER_PROVIDER=hash
export GENERATOR_PROVIDER=simple
```

### External Provider Mode

Example:

```bash
export EMBEDDER_PROVIDER=external
export EMBEDDING_MODEL_NAME=text-embedding-3-small
export EXTERNAL_EMBEDDING_API_KEY=your-embedding-key

export GENERATOR_PROVIDER=external
export GENERATOR_MODEL_NAME=gpt-4o-mini
export EXTERNAL_GENERATOR_API_KEY=your-generator-key
```

### Optional Local Provider Mode

Install optional dependencies first:

```bash
pip install -e ".[local-embeddings]"
pip install -e ".[local-generation]"
```

Then configure:

```bash
export EMBEDDER_PROVIDER=local
export EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

export GENERATOR_PROVIDER=local
export GENERATOR_MODEL_NAME=google/flan-t5-small
```

### Ollama Qwen Provider Mode

The easiest path is:

```bash
make llm-up
```

Manual equivalent:

```bash
export EMBEDDER_PROVIDER=hash
export GENERATOR_PROVIDER=ollama
export GENERATOR_MODEL_NAME=qwen2.5:1.5b
export OLLAMA_MODEL=qwen2.5:1.5b
export LOCAL_MODEL_ENDPOINT=http://model:11434

docker compose --profile local-model up --build -d
```

Ollama exposes the model on the host at:

```text
http://localhost:11434
```

## API Usage

Base URL:

```text
http://localhost:8000/api
```

### Health

```bash
curl -fsS http://localhost:8000/api/health
```

### React Workbench

If you are running the Compose stack, open:

```text
http://localhost:8010/
```

The workbench includes document ingestion, query controls, citations, retrieved chunk inspection, and browser-local query history.

### Ingest Documents

Index a directory of local files:

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

### Query the API

Basic query:

```bash
curl -X POST http://localhost:8000/api/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Which database stores document embeddings?",
    "top_k": 3
  }'
```

Query with source-path filtering:

```bash
curl -X POST http://localhost:8000/api/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Which database stores document embeddings?",
    "top_k": 3,
    "source_path": "/absolute/path/to/architecture.md"
  }'
```

Response shape:

```json
{
  "result": {
    "answer_text": "Question: Which database stores document embeddings?\n\nGrounded context: ...",
    "citations": [
      {
        "chunk_id": "chunk-id",
        "document_id": "document-id",
        "source_path": "/absolute/path/to/file.md",
        "document_title": "RAG MVP Architecture",
        "file_name": "file.md",
        "file_type": "md",
        "section_header": "Architecture",
        "page_number": null,
        "chunk_index": 0,
        "start_char": 0,
        "end_char": 120,
        "snippet": "This sample project stores document embeddings in Chroma."
      }
    ],
    "retrieved_chunks": [
      {
        "chunk_id": "chunk-id",
        "score": 0.45,
        "distance": 0.55,
        "relevance": "high",
        "source_path": "/absolute/path/to/file.md",
        "document_title": "RAG MVP Architecture",
        "file_name": "file.md",
        "file_type": "md",
        "section_header": "Architecture",
        "page_number": null,
        "chunk_index": 0,
        "text": "This sample project stores document embeddings in Chroma."
      }
    ]
  }
}
```

## Framework-Agnostic Usage

The core pipeline can also be called directly from Python without going through FastAPI routes.

Use cases live in [`src/rag/use_cases.py`](/home/yuhanzhang/ai-assitant/src/rag/use_cases.py).

Example:

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

## Running Tests

Run the full test suite:

```bash
pytest -q
```

Run a focused subset:

```bash
pytest tests/unit -q
pytest tests/integration -q
```

## Current Limitations

- The default generation path is still intentionally simple and deterministic.
- External provider support is implemented, but this README does not claim live external API validation in every environment.
- Local provider mode runs models in-process and may be slow or too heavy for smaller machines.
- Ollama provider mode requires the `model` service to be running and may need Docker Desktop or the host Docker engine to have enough memory available.
- PDF support depends on text extraction quality; scanned/image-only PDFs are not handled yet.
