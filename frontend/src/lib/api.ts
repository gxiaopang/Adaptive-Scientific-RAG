import type { QueryResponse } from "../types";

const API_ORIGIN = (import.meta.env.VITE_API_ORIGIN ?? "").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function parseError(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (typeof body === "object" && body !== null && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      if (typeof detail === "string") return detail;
    }
  } catch {
    // Fall through to the stable status text; never render arbitrary response HTML.
  }
  return response.statusText || "Request failed";
}

type QueryStreamEvent =
  | { type: "answer_start" }
  | { type: "answer_delta"; text: string }
  | { type: "result"; response: QueryResponse }
  | { type: "error"; status: number; detail: string };

function parseStreamEvent(line: string): QueryStreamEvent {
  let value: unknown;
  try {
    value = JSON.parse(line);
  } catch {
    throw new ApiError(502, "The backend returned an invalid stream event.");
  }
  if (typeof value !== "object" || value === null || !("type" in value)) {
    throw new ApiError(502, "The backend returned an invalid stream event.");
  }
  return value as QueryStreamEvent;
}

export async function streamQuery(
  query: string,
  conversationId: string | null,
  signal: AbortSignal,
  onAnswerStart: () => void,
  onAnswerDelta: (text: string) => void
): Promise<QueryResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_ORIGIN}/api/v1/query/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, conversation_id: conversationId }),
      signal
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new ApiError(0, "The backend could not be reached.");
  }
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  if (response.body === null) {
    throw new ApiError(502, "The backend returned an empty response stream.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result: QueryResponse | null = null;

  function handleLine(line: string) {
    if (!line.trim()) return;
    const event = parseStreamEvent(line);
    if (event.type === "answer_start") onAnswerStart();
    if (event.type === "answer_delta") onAnswerDelta(event.text);
    if (event.type === "result") result = event.response;
    if (event.type === "error") throw new ApiError(event.status, event.detail);
  }

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) handleLine(line);
    if (done) break;
  }
  handleLine(buffer);
  if (result === null) {
    throw new ApiError(502, "The backend stream ended before the final result.");
  }
  return result;
}

export async function checkHealth(signal: AbortSignal): Promise<boolean> {
  try {
    const response = await fetch(`${API_ORIGIN}/health`, { signal });
    return response.ok;
  } catch {
    return false;
  }
}
