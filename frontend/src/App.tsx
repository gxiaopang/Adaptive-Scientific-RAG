import { BookMarked, Menu, Plus, RotateCcw, Send, Square } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { EvidencePanel } from "./components/evidence-panel";
import { Timeline } from "./components/timeline";
import { Badge } from "./components/ui/badge";
import { Button } from "./components/ui/button";
import { Sheet } from "./components/ui/sheet";
import { ApiError, checkHealth, streamQuery } from "./lib/api";
import type { QueryResponse, StoredConversation, TimelineMessage } from "./types";

const STORAGE_KEY = "asr:conversation:v1";

type RequestError = { message: string; retryable: boolean; expired: boolean };

function readStoredConversation(): StoredConversation {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw === null) return { conversationId: null, messages: [] };
    const parsed = JSON.parse(raw) as Partial<StoredConversation>;
    return {
      conversationId: typeof parsed.conversationId === "string" ? parsed.conversationId : null,
      messages: Array.isArray(parsed.messages) ? parsed.messages : []
    };
  } catch {
    return { conversationId: null, messages: [] };
  }
}

function friendlyError(error: unknown): RequestError {
  if (error instanceof DOMException && error.name === "AbortError") {
    return { message: "Request cancelled. Your question was not automatically retried.", retryable: true, expired: false };
  }
  if (error instanceof ApiError) {
    if (error.status === 404) return { message: "This conversation is no longer available. Start a new conversation to continue.", retryable: false, expired: true };
    if (error.status === 422) return { message: "The question was not accepted. Check its length and try again.", retryable: true, expired: false };
    if (error.status === 502) return { message: "A model provider could not complete the workflow. You can retry when it recovers.", retryable: true, expired: false };
    if (error.status === 503) return { message: "The research service is not ready yet. Check the backend and retry.", retryable: true, expired: false };
    if (error.status === 0) return { message: "The backend could not be reached. Check your connection and API origin.", retryable: true, expired: false };
  }
  return { message: "Something unexpected interrupted the request.", retryable: true, expired: false };
}

export default function App() {
  const initial = useMemo(readStoredConversation, []);
  const [conversationId, setConversationId] = useState<string | null>(initial.conversationId);
  const [messages, setMessages] = useState<TimelineMessage[]>(initial.messages);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<RequestError | null>(null);
  const [lastQuestion, setLastQuestion] = useState<string | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [sheetOpen, setSheetOpen] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const latestScientific = useMemo<QueryResponse | null>(() => {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      const message = messages[index];
      if (message?.role === "assistant" && message.result?.response_mode === "scientific_evidence") return message.result;
    }
    return null;
  }, [messages]);

  useEffect(() => {
    const completedMessages = messages.filter(
      (message) => message.role === "user" || message.result !== null
    );
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ conversationId, messages: completedMessages }));
  }, [conversationId, messages]);

  useEffect(() => {
    const controller = new AbortController();
    void checkHealth(controller.signal).then(setBackendOnline);
    return () => controller.abort();
  }, []);

  async function runQuery(rawQuestion: string) {
    const normalized = rawQuestion.trim();
    if (!normalized || loading) return;
    setLoading(true);
    setError(null);
    setLastQuestion(normalized);
    setQuestion("");
    const userMessage: TimelineMessage = { id: crypto.randomUUID(), role: "user", text: normalized };
    const assistantMessageId = crypto.randomUUID();
    setMessages((current) => [...current, userMessage]);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const result = await streamQuery(
        normalized,
        conversationId,
        controller.signal,
        () => {
          setMessages((current) => {
            const existing = current.find((message) => message.id === assistantMessageId);
            if (existing !== undefined) {
              return current.map((message) =>
                message.id === assistantMessageId && message.role === "assistant"
                  ? { ...message, text: "", result: null }
                  : message
              );
            }
            return [
              ...current,
              { id: assistantMessageId, role: "assistant", text: "", result: null }
            ];
          });
        },
        (text) => {
          setMessages((current) => {
            const existing = current.find((message) => message.id === assistantMessageId);
            if (existing === undefined) {
              return [
                ...current,
                { id: assistantMessageId, role: "assistant", text, result: null }
              ];
            }
            return current.map((message) =>
              message.id === assistantMessageId && message.role === "assistant"
                ? { ...message, text: message.text + text }
                : message
            );
          });
        }
      );
      setConversationId(result.conversation_id);
      setMessages((current) => {
        const existing = current.find((message) => message.id === assistantMessageId);
        if (existing === undefined) {
          return [...current, { id: assistantMessageId, role: "assistant", text: result.answer, result }];
        }
        return current.map((message) =>
          message.id === assistantMessageId && message.role === "assistant"
            ? { ...message, text: result.answer, result }
            : message
        );
      });
      setBackendOnline(true);
    } catch (caught) {
      setMessages((current) => current.filter((message) => message.id !== assistantMessageId));
      setError(friendlyError(caught));
    } finally {
      abortRef.current = null;
      setLoading(false);
      requestAnimationFrame(() => textareaRef.current?.focus());
    }
  }

  function newConversation() {
    abortRef.current?.abort();
    setConversationId(null);
    setMessages([]);
    setQuestion("");
    setLastQuestion(null);
    setError(null);
    localStorage.removeItem(STORAGE_KEY);
    requestAnimationFrame(() => textareaRef.current?.focus());
  }

  return (
    <div className="min-h-screen bg-paper text-ink">
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(circle_at_18%_10%,rgba(217,239,135,0.28),transparent_30%),radial-gradient(circle_at_86%_90%,rgba(65,100,90,0.12),transparent_28%)]" />
      <header className="sticky top-0 z-30 border-b border-ink/10 bg-paper/85 backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-[1500px] items-center justify-between px-4 sm:px-7">
          <div className="flex items-center gap-3">
            <span className="flex h-9 w-9 items-center justify-center rounded-2xl bg-ink text-citron"><BookMarked className="h-4 w-4" /></span>
            <div><p className="font-serif text-lg leading-5">Evidence Desk</p><p className="text-[10px] uppercase tracking-[0.18em] text-ink/40">Adaptive Scientific RAG</p></div>
          </div>
          <div className="flex items-center gap-2 sm:gap-4">
            <div className="hidden items-center gap-2 text-xs text-ink/55 sm:flex" role="status" aria-label="Backend service status">
              <span className={`h-2 w-2 rounded-full ${backendOnline === true ? "bg-emerald-500" : backendOnline === false ? "bg-rose-500" : "animate-pulse bg-amber-400"}`} />
              {backendOnline === true ? "Backend online" : backendOnline === false ? "Backend offline" : "Checking backend"}
            </div>
            {latestScientific !== null && <Button type="button" variant="outline" className="lg:hidden" onClick={() => setSheetOpen(true)}><Menu className="h-4 w-4" />Evidence</Button>}
            <Button type="button" variant="ghost" onClick={newConversation}><Plus className="h-4 w-4" /><span className="hidden sm:inline">New conversation</span></Button>
          </div>
        </div>
      </header>

      <main className="relative mx-auto grid max-w-[1500px] grid-cols-1 lg:grid-cols-[minmax(0,1fr)_390px]">
        <section className="min-w-0 px-4 pb-48 sm:px-8 lg:border-r lg:border-ink/10 lg:px-12" aria-label="Conversation">
          <Timeline messages={messages} loading={loading} onSuggestion={(value) => { setQuestion(value); textareaRef.current?.focus(); }} />
        </section>
        <aside className="sticky top-16 hidden h-[calc(100vh-4rem)] overflow-y-auto px-7 py-8 lg:block" aria-label="Scientific evidence">
          <div className="mb-5 flex items-center justify-between"><h2 className="font-serif text-2xl">Evidence trace</h2><Badge>Live result</Badge></div>
          <EvidencePanel result={latestScientific} />
        </aside>
      </main>

      <div className="fixed inset-x-0 bottom-0 z-20 bg-gradient-to-t from-paper via-paper/95 to-transparent px-4 pb-5 pt-12 sm:px-8 lg:right-[390px]">
        <div className="mx-auto max-w-3xl">
          {error !== null && (
            <div className="mb-3 flex items-center justify-between gap-3 rounded-2xl border border-rose-900/10 bg-rose-50 px-4 py-3 text-sm text-rose-900" role="alert">
              <span>{error.message}</span>
              {error.expired ? <Button type="button" variant="ghost" onClick={newConversation}>Start new</Button> : error.retryable && lastQuestion !== null ? <Button type="button" variant="ghost" onClick={() => void runQuery(lastQuestion)}><RotateCcw className="h-4 w-4" />Retry</Button> : null}
            </div>
          )}
          <form className="rounded-[1.8rem] border border-ink/15 bg-white/85 p-2 shadow-panel backdrop-blur-xl" onSubmit={(event) => { event.preventDefault(); void runQuery(question); }}>
            <label htmlFor="research-question" className="sr-only">Ask a scientific question</label>
            <textarea
              ref={textareaRef}
              id="research-question"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void runQuery(question); } }}
              placeholder="Ask a scientific question or continue the conversation…"
              rows={2}
              maxLength={2000}
              disabled={loading}
              className="max-h-40 min-h-14 w-full resize-none bg-transparent px-4 py-3 text-[15px] leading-6 text-ink outline-none placeholder:text-ink/35 disabled:opacity-60"
            />
            <div className="flex items-center justify-between px-2 pb-1">
              <span className="text-[11px] text-ink/35">Enter to send · Shift + Enter for a new line</span>
              {loading ? <Button type="button" variant="outline" onClick={() => abortRef.current?.abort()}><Square className="h-3.5 w-3.5 fill-current" />Cancel</Button> : <Button type="submit" disabled={!question.trim()}><Send className="h-4 w-4" />Ask</Button>}
            </div>
          </form>
          <div className="sr-only" aria-live="polite">{loading ? "Research request in progress" : error?.message ?? ""}</div>
        </div>
      </div>

      <Sheet open={sheetOpen} onOpenChange={setSheetOpen} title="Evidence trace"><EvidencePanel result={latestScientific} /></Sheet>
    </div>
  );
}
