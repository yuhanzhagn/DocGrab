export type IngestResponse = {
  indexed_documents: number;
  indexed_chunks: number;
  skipped_files: string[];
};

export type Citation = {
  chunk_id: string;
  document_id: string;
  source_path: string;
  document_title?: string | null;
  file_name?: string | null;
  file_type?: string | null;
  section_header?: string | null;
  page_number?: number | null;
  chunk_index: number;
  start_char: number;
  end_char: number;
  snippet: string;
};

export type RetrievedChunk = {
  chunk_id?: string;
  document_id?: string;
  source_path?: string;
  file_name?: string;
  text?: string;
  score?: number;
  distance?: number | null;
  relevance?: string;
  page_number?: number | null;
  section_header?: string | null;
  [key: string]: unknown;
};

export type QueryResponse = {
  result: {
    answer_text: string;
    citations: Citation[];
    retrieved_chunks: RetrievedChunk[];
  };
};

export type QueryHistoryItem = {
  id: string;
  query: string;
  topK: number;
  answer: string;
  citationCount: number;
  createdAt: string;
};
