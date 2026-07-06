import { FormEvent, useEffect, useState } from "react";
import {
  AlertCircle,
  BookOpenText,
  CheckCircle2,
  DatabaseZap,
  FileSearch,
  FolderInput,
  History,
  Loader2,
  Search,
  Sparkles
} from "lucide-react";
import { getHealth, ingestDocuments, queryDocuments } from "./api";
import type { IngestResponse, QueryHistoryItem, QueryResponse, RetrievedChunk } from "./types";

const HISTORY_KEY = "rag-workbench-history";
const SAMPLE_DOCKER_PATH = "/app/data/sample_docs";
const SAMPLE_LOCAL_PATH = "data/sample_docs";

function loadHistory(): QueryHistoryItem[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? (JSON.parse(raw) as QueryHistoryItem[]) : [];
  } catch {
    return [];
  }
}

function formatScore(score: unknown): string {
  return typeof score === "number" ? score.toFixed(3) : "n/a";
}

function getChunkTitle(chunk: RetrievedChunk): string {
  if (typeof chunk.file_name === "string" && chunk.file_name) {
    return chunk.file_name;
  }
  if (typeof chunk.source_path === "string" && chunk.source_path) {
    return chunk.source_path.split("/").pop() || chunk.source_path;
  }
  return "Retrieved chunk";
}

export function App() {
  const [activePanel, setActivePanel] = useState<"query" | "ingest">("query");
  const [health, setHealth] = useState<"checking" | "online" | "offline">("checking");
  const [directory, setDirectory] = useState(SAMPLE_DOCKER_PATH);
  const [ingestResult, setIngestResult] = useState<IngestResponse | null>(null);
  const [ingestError, setIngestError] = useState<string | null>(null);
  const [isIngesting, setIsIngesting] = useState(false);
  const [query, setQuery] = useState("Which database stores document embeddings?");
  const [sourcePath, setSourcePath] = useState("");
  const [topK, setTopK] = useState(5);
  const [queryResult, setQueryResult] = useState<QueryResponse["result"] | null>(null);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [isQuerying, setIsQuerying] = useState(false);
  const [history, setHistory] = useState<QueryHistoryItem[]>(loadHistory);

  useEffect(() => {
    getHealth()
      .then(() => setHealth("online"))
      .catch(() => setHealth("offline"));
  }, []);

  useEffect(() => {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, 8)));
  }, [history]);

  async function handleIngest(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsIngesting(true);
    setIngestError(null);
    setIngestResult(null);

    try {
      const result = await ingestDocuments(directory.trim());
      setIngestResult(result);
    } catch (error) {
      setIngestError(error instanceof Error ? error.message : "Ingestion failed.");
    } finally {
      setIsIngesting(false);
    }
  }

  async function handleQuery(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsQuerying(true);
    setQueryError(null);
    setQueryResult(null);

    try {
      const response = await queryDocuments(query.trim(), topK, sourcePath.trim());
      setQueryResult(response.result);
      setHistory((current) => [
        {
          id: crypto.randomUUID(),
          query: query.trim(),
          topK,
          answer: response.result.answer_text,
          citationCount: response.result.citations.length,
          createdAt: new Date().toISOString()
        },
        ...current
      ].slice(0, 8));
    } catch (error) {
      setQueryError(error instanceof Error ? error.message : "Query failed.");
    } finally {
      setIsQuerying(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="hero">
        <div>
          <p className="eyebrow">RAG MVP</p>
          <h1>Document retrieval workbench</h1>
          <p className="hero-copy">
            Index local files, ask grounded questions, and inspect exactly which chunks shaped the answer.
          </p>
        </div>
        <div className={`status-pill status-${health}`}>
          {health === "checking" ? <Loader2 className="spin" size={18} /> : health === "online" ? <CheckCircle2 size={18} /> : <AlertCircle size={18} />}
          <span>{health === "checking" ? "Checking API" : health === "online" ? "API online" : "API offline"}</span>
        </div>
      </section>

      <section className="workspace">
        <aside className="sidebar">
          <button className={activePanel === "query" ? "nav-button active" : "nav-button"} onClick={() => setActivePanel("query")}>
            <Search size={18} />
            Query
          </button>
          <button className={activePanel === "ingest" ? "nav-button active" : "nav-button"} onClick={() => setActivePanel("ingest")}>
            <FolderInput size={18} />
            Ingest
          </button>

          <div className="sidebar-block">
            <div className="block-heading">
              <History size={16} />
              Recent
            </div>
            {history.length === 0 ? (
              <p className="muted small">Queries will appear here after the first answer.</p>
            ) : (
              <div className="history-list">
                {history.map((item) => (
                  <button
                    className="history-item"
                    key={item.id}
                    onClick={() => {
                      setActivePanel("query");
                      setQuery(item.query);
                    }}
                  >
                    <span>{item.query}</span>
                    <small>{item.citationCount} citations</small>
                  </button>
                ))}
              </div>
            )}
          </div>
        </aside>

        <section className="panel">
          {activePanel === "query" ? (
            <>
              <form className="tool-surface" onSubmit={handleQuery}>
                <div className="section-title">
                  <FileSearch size={22} />
                  <div>
                    <h2>Ask the index</h2>
                    <p>Answers include citations and retrieved chunks from the FastAPI backend.</p>
                  </div>
                </div>

                <label>
                  <span>Question</span>
                  <textarea value={query} onChange={(event) => setQuery(event.target.value)} rows={5} required />
                </label>

                <div className="form-grid">
                  <label>
                    <span>Top K</span>
                    <input
                      type="number"
                      min={1}
                      max={20}
                      value={topK}
                      onChange={(event) => setTopK(Number(event.target.value))}
                    />
                  </label>
                  <label>
                    <span>Source path filter</span>
                    <input
                      value={sourcePath}
                      onChange={(event) => setSourcePath(event.target.value)}
                      placeholder="/app/data/sample_docs/architecture.md"
                    />
                  </label>
                </div>

                <button className="primary-action" disabled={isQuerying || !query.trim()}>
                  {isQuerying ? <Loader2 className="spin" size={18} /> : <Sparkles size={18} />}
                  {isQuerying ? "Searching" : "Run query"}
                </button>
              </form>

              {queryError && <Notice tone="error" message={queryError} />}

              {queryResult && (
                <div className="results-grid">
                  <article className="answer-surface">
                    <div className="section-title compact">
                      <BookOpenText size={20} />
                      <h2>Answer</h2>
                    </div>
                    <p className="answer-text">{queryResult.answer_text}</p>
                  </article>

                  <article className="tool-surface">
                    <h2>Citations</h2>
                    {queryResult.citations.length === 0 ? (
                      <p className="muted">No citations returned.</p>
                    ) : (
                      <div className="citation-list">
                        {queryResult.citations.map((citation) => (
                          <div className="citation-item" key={citation.chunk_id}>
                            <strong>{citation.file_name || citation.source_path}</strong>
                            <span>
                              {citation.section_header ? `${citation.section_header} · ` : ""}
                              {citation.page_number ? `page ${citation.page_number} · ` : ""}
                              chunk {citation.chunk_index}
                            </span>
                            <p>{citation.snippet}</p>
                          </div>
                        ))}
                      </div>
                    )}
                  </article>

                  <article className="tool-surface wide">
                    <h2>Retrieved chunks</h2>
                    {queryResult.retrieved_chunks.length === 0 ? (
                      <p className="muted">No retrieved chunks returned.</p>
                    ) : (
                      <div className="chunk-list">
                        {queryResult.retrieved_chunks.map((chunk, index) => (
                          <details key={`${chunk.chunk_id ?? index}-${index}`} open={index === 0}>
                            <summary>
                              <span>{getChunkTitle(chunk)}</span>
                              <small>
                                score {formatScore(chunk.score)} {chunk.relevance ? `· ${chunk.relevance}` : ""}
                              </small>
                            </summary>
                            <pre>{typeof chunk.text === "string" ? chunk.text : JSON.stringify(chunk, null, 2)}</pre>
                          </details>
                        ))}
                      </div>
                    )}
                  </article>
                </div>
              )}
            </>
          ) : (
            <>
              <form className="tool-surface" onSubmit={handleIngest}>
                <div className="section-title">
                  <DatabaseZap size={22} />
                  <div>
                    <h2>Ingest documents</h2>
                    <p>The path must be readable by the FastAPI backend container or local process.</p>
                  </div>
                </div>

                <label>
                  <span>Directory path</span>
                  <input value={directory} onChange={(event) => setDirectory(event.target.value)} required />
                </label>

                <div className="path-actions">
                  <button type="button" onClick={() => setDirectory(SAMPLE_DOCKER_PATH)}>
                    Docker sample
                  </button>
                  <button type="button" onClick={() => setDirectory(SAMPLE_LOCAL_PATH)}>
                    Local sample
                  </button>
                </div>

                <button className="primary-action" disabled={isIngesting || !directory.trim()}>
                  {isIngesting ? <Loader2 className="spin" size={18} /> : <FolderInput size={18} />}
                  {isIngesting ? "Indexing" : "Run ingestion"}
                </button>
              </form>

              {ingestError && <Notice tone="error" message={ingestError} />}

              {ingestResult && (
                <article className="tool-surface result-summary">
                  <h2>Indexing result</h2>
                  <div className="metric-row">
                    <Metric label="Documents" value={ingestResult.indexed_documents} />
                    <Metric label="Chunks" value={ingestResult.indexed_chunks} />
                    <Metric label="Skipped" value={ingestResult.skipped_files.length} />
                  </div>
                  {ingestResult.skipped_files.length > 0 && (
                    <div className="skipped-list">
                      {ingestResult.skipped_files.map((path) => (
                        <code key={path}>{path}</code>
                      ))}
                    </div>
                  )}
                </article>
              )}
            </>
          )}
        </section>
      </section>
    </main>
  );
}

function Notice({ tone, message }: { tone: "error" | "success"; message: string }) {
  return (
    <div className={`notice ${tone}`}>
      {tone === "error" ? <AlertCircle size={18} /> : <CheckCircle2 size={18} />}
      <span>{message}</span>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div className="metric">
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}
