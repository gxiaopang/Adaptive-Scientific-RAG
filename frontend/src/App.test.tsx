import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import App from "./App";
import type { QueryResponse } from "./types";

const scientificResponse: QueryResponse = {
  conversation_id: "12345678-1234-5678-1234-567812345678",
  original_query: "Does A cause B?",
  response_mode: "scientific_evidence",
  routing_reason: "Scientific claim",
  final_query: "A B causal evidence",
  query_history: ["Does A cause B?", "A B causal evidence"],
  answer: "The available study supports a limited association.",
  evidence_sufficient: true,
  verification_reason: "The evidence directly addresses the association.",
  retrieval_attempts: 2,
  termination_reason: "evidence_sufficient",
  evidence: [
    { doc_id: "42", title: "A controlled study", excerpt: "A concise evidence passage.", score: 0.912, rank: 1 }
  ]
};

function response(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" }
  });
}

function streamResponse(result: QueryResponse): Response {
  const midpoint = Math.floor(result.answer.length / 2);
  const events = [
    { type: "answer_start" },
    { type: "answer_delta", text: result.answer.slice(0, midpoint) },
    { type: "answer_delta", text: result.answer.slice(midpoint) },
    { type: "result", response: result }
  ];
  return new Response(`${events.map((event) => JSON.stringify(event)).join("\n")}\n`, {
    status: 200,
    headers: { "Content-Type": "application/x-ndjson" }
  });
}

function mockFetch(queryResponse: Response): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(response({ status: "ok" }))
    .mockResolvedValueOnce(queryResponse);
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("research workspace", () => {
  it("renders a scientific answer, verification, and evidence", async () => {
    const user = userEvent.setup();
    mockFetch(streamResponse(scientificResponse));
    render(<App />);

    await user.type(screen.getByLabelText("Ask a scientific question"), "Does A cause B?");
    await user.click(screen.getByRole("button", { name: "Ask" }));

    expect(await screen.findByText(scientificResponse.answer)).toBeInTheDocument();
    expect(screen.getByText("Evidence verified")).toBeInTheDocument();
    expect(screen.getByText("A controlled study")).toBeInTheDocument();
    expect(screen.getByText("A B causal evidence")).toBeInTheDocument();
  });

  it("opens the mobile evidence sheet for the latest scientific answer", async () => {
    const user = userEvent.setup();
    mockFetch(streamResponse(scientificResponse));
    render(<App />);

    await user.type(screen.getByLabelText("Ask a scientific question"), "Does A cause B?");
    await user.click(screen.getByRole("button", { name: "Ask" }));
    await screen.findByText(scientificResponse.answer);
    await user.click(screen.getByRole("button", { name: "Evidence" }));

    const dialog = screen.getByRole("dialog", { name: "Evidence trace" });
    expect(within(dialog).getByText("A controlled study")).toBeInTheDocument();
    await user.click(within(dialog).getByRole("button", { name: "Close evidence panel" }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("keeps general chat free of retrieval metadata", async () => {
    const user = userEvent.setup();
    mockFetch(streamResponse({
      ...scientificResponse,
      response_mode: "general_chat",
      answer: "Hello — how can I help?",
      evidence: [],
      evidence_sufficient: null,
      verification_reason: null,
      retrieval_attempts: 0,
      query_history: ["Hello"]
    } satisfies QueryResponse));
    render(<App />);

    await user.type(screen.getByLabelText("Ask a scientific question"), "Hello");
    await user.click(screen.getByRole("button", { name: "Ask" }));

    expect(await screen.findByText("Hello — how can I help?")).toBeInTheDocument();
    expect(screen.queryByText("Evidence verified")).not.toBeInTheDocument();
    expect(screen.getByText("Conversation")).toBeInTheDocument();
  });

  it("prevents duplicate submission and supports cancellation", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.fn().mockResolvedValueOnce(response({ status: "ok" }));
    fetchMock.mockImplementationOnce((_url: string, options: RequestInit) =>
      new Promise((_resolve, reject) => {
        options.signal?.addEventListener("abort", () => reject(new DOMException("cancelled", "AbortError")));
      })
    );
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    await user.type(screen.getByLabelText("Ask a scientific question"), "Long research request");
    await user.click(screen.getByRole("button", { name: "Ask" }));
    expect(screen.getByRole("button", { name: "Cancel" })).toBeInTheDocument();
    expect(screen.getByLabelText("Ask a scientific question")).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Request cancelled");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("shows conversation expiry and requires an explicit new conversation", async () => {
    const user = userEvent.setup();
    mockFetch(response({ detail: "not found" }, 404));
    render(<App />);

    await user.type(screen.getByLabelText("Ask a scientific question"), "Follow up");
    await user.click(screen.getByRole("button", { name: "Ask" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("no longer available");
    expect(screen.getByRole("button", { name: "Start new" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Retry" })).not.toBeInTheDocument();
  });

  it("restores local messages and clears them with New conversation", async () => {
    localStorage.setItem("asr:conversation:v1", JSON.stringify({
      conversationId: scientificResponse.conversation_id,
      messages: [{ id: "saved", role: "assistant", text: "Restored answer", result: scientificResponse }]
    }));
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response({ status: "ok" })));
    const user = userEvent.setup();
    render(<App />);

    expect(screen.getByText("Restored answer")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /New conversation/ }));

    expect(screen.queryByText("Restored answer")).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /Ask a claim.*Trace the evidence/ })).toBeInTheDocument();
  });

  it("submits with Enter while Shift+Enter preserves a newline", async () => {
    mockFetch(streamResponse({ ...scientificResponse, evidence: [], evidence_sufficient: false }));
    render(<App />);
    const textarea = screen.getByLabelText("Ask a scientific question");
    fireEvent.change(textarea, { target: { value: "Question" } });
    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: true });
    expect(textarea).toHaveValue("Question");
    fireEvent.keyDown(textarea, { key: "Enter" });
    await waitFor(() => expect(screen.getByText(scientificResponse.answer)).toBeInTheDocument());
    expect(screen.getByText("Evidence remains limited")).toBeInTheDocument();
    expect(screen.getByText("No evidence passages were returned for this answer.")).toBeInTheDocument();
  });

  it("renders answer text before the final stream result arrives", async () => {
    const user = userEvent.setup();
    const encoder = new TextEncoder();
    let streamController: ReadableStreamDefaultController<Uint8Array> | null = null;
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        streamController = controller;
      }
    });
    vi.stubGlobal(
      "fetch",
      vi.fn()
        .mockResolvedValueOnce(response({ status: "ok" }))
        .mockResolvedValueOnce(new Response(stream, {
          status: 200,
          headers: { "Content-Type": "application/x-ndjson" }
        }))
    );
    render(<App />);

    await user.type(screen.getByLabelText("Ask a scientific question"), "Does A cause B?");
    await user.click(screen.getByRole("button", { name: "Ask" }));
    act(() => {
      streamController?.enqueue(encoder.encode(
        `${JSON.stringify({ type: "answer_start" })}\n${JSON.stringify({ type: "answer_delta", text: "Partial answer" })}\n`
      ));
    });

    expect(await screen.findByText("Partial answer")).toBeInTheDocument();
    expect(screen.getByText("Responding")).toBeInTheDocument();

    act(() => {
      streamController?.enqueue(encoder.encode(
        `${JSON.stringify({ type: "result", response: scientificResponse })}\n`
      ));
      streamController?.close();
    });
    expect(await screen.findByText(scientificResponse.answer)).toBeInTheDocument();
  });

  it.each([
    [422, "question was not accepted"],
    [502, "model provider could not complete"],
    [503, "research service is not ready"]
  ])("renders a retryable HTTP %s error", async (status, expected) => {
    const user = userEvent.setup();
    mockFetch(response({ detail: "provider details must not be rendered" }, status));
    render(<App />);

    await user.type(screen.getByLabelText("Ask a scientific question"), "Question");
    await user.click(screen.getByRole("button", { name: "Ask" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(expected);
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
    expect(screen.queryByText("provider details must not be rendered")).not.toBeInTheDocument();
  });

  it("renders a retryable network error", async () => {
    const user = userEvent.setup();
    vi.stubGlobal(
      "fetch",
      vi.fn()
        .mockResolvedValueOnce(response({ status: "ok" }))
        .mockRejectedValueOnce(new TypeError("network failed"))
    );
    render(<App />);

    await user.type(screen.getByLabelText("Ask a scientific question"), "Question");
    await user.click(screen.getByRole("button", { name: "Ask" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("backend could not be reached");
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });
});
