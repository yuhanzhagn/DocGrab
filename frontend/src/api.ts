import type { IngestResponse, QueryResponse } from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "/api";

type ApiErrorPayload = {
  detail?: string | { msg?: string }[] | unknown;
};

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers
    }
  });

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;
    try {
      const payload = (await response.json()) as ApiErrorPayload;
      if (typeof payload.detail === "string") {
        message = payload.detail;
      } else if (Array.isArray(payload.detail)) {
        message = payload.detail
          .map((item) => (typeof item === "object" && item && "msg" in item ? String(item.msg) : "Validation error"))
          .join(", ");
      }
    } catch {
      message = response.statusText || message;
    }
    throw new Error(message);
  }

  return response.json() as Promise<T>;
}

export function ingestDocuments(directory: string): Promise<IngestResponse> {
  return requestJson<IngestResponse>("/documents/ingest", {
    method: "POST",
    body: JSON.stringify({ directory })
  });
}

export function queryDocuments(query: string, topK: number, sourcePath?: string): Promise<QueryResponse> {
  return requestJson<QueryResponse>("/query/", {
    method: "POST",
    body: JSON.stringify({
      query,
      top_k: topK,
      source_path: sourcePath || null
    })
  });
}

export function getHealth(): Promise<{ status: string }> {
  return requestJson<{ status: string }>("/health");
}
