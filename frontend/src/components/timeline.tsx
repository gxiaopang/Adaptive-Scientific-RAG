import { FlaskConical, MessageSquareText, Search } from "lucide-react";

import type { TimelineMessage } from "../types";
import { Badge } from "./ui/badge";

const suggestions = [
  "What evidence links vitamin D to respiratory outcomes?",
  "How does scientific retrieval differ from a web search?",
  "Summarize what makes a claim well-supported."
];

export function Timeline({
  messages,
  loading,
  onSuggestion
}: {
  messages: TimelineMessage[];
  loading: boolean;
  onSuggestion: (text: string) => void;
}) {
  if (messages.length === 0) {
    return (
      <div className="mx-auto flex min-h-[58vh] max-w-3xl flex-col justify-center py-16">
        <Badge className="mb-5 w-fit bg-citron/60 text-ink">SciFact evidence workspace</Badge>
        <h1 className="max-w-2xl font-serif text-5xl leading-[1.02] tracking-[-0.04em] text-ink sm:text-6xl">
          Ask a claim.<br />Trace the evidence.
        </h1>
        <p className="mt-6 max-w-xl text-base leading-7 text-ink/60">
          A focused research desk for scientific questions, retrieval transparency, and answers that show their work.
        </p>
        <div className="mt-9 grid gap-3 sm:grid-cols-3">
          {suggestions.map((suggestion, index) => (
            <button
              key={suggestion}
              type="button"
              onClick={() => onSuggestion(suggestion)}
              className="group rounded-3xl border border-ink/10 bg-white/55 p-4 text-left text-sm leading-6 text-ink/70 transition hover:-translate-y-0.5 hover:bg-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-moss"
            >
              {index === 0 ? <FlaskConical className="mb-4 h-5 w-5 text-moss" /> : index === 1 ? <Search className="mb-4 h-5 w-5 text-moss" /> : <MessageSquareText className="mb-4 h-5 w-5 text-moss" />}
              {suggestion}
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-3xl space-y-8 py-10">
      {messages.map((message) =>
        message.role === "user" ? (
          <article key={message.id} className="ml-auto max-w-[85%] rounded-[1.6rem] rounded-br-md bg-ink px-5 py-4 text-white shadow-panel">
            <p className="whitespace-pre-wrap text-sm leading-6">{message.text}</p>
          </article>
        ) : (
          <article key={message.id} className="max-w-[95%]">
            <div className="mb-3 flex items-center gap-2">
              <span className="flex h-8 w-8 items-center justify-center rounded-full bg-citron text-ink">
                <FlaskConical className="h-4 w-4" />
              </span>
              <span className="text-xs font-bold uppercase tracking-[0.14em] text-ink/45">Evidence Desk</span>
              <Badge>{message.result === null ? "Responding" : message.result.response_mode === "general_chat" ? "Conversation" : "Scientific answer"}</Badge>
            </div>
            <p className="whitespace-pre-wrap font-serif text-[1.35rem] leading-8 text-ink">
              {message.text}
              {message.result === null && <span className="ml-1 inline-block h-5 w-0.5 animate-pulse bg-moss align-middle" aria-hidden="true" />}
            </p>
            {message.result?.response_mode === "scientific_evidence" && (
              <div className="mt-4 flex flex-wrap gap-2 lg:hidden">
                <Badge>{message.result.evidence.length} evidence sources</Badge>
                <Badge>{message.result.evidence_sufficient ? "Verified" : "Verification limited"}</Badge>
              </div>
            )}
          </article>
        )
      )}
      {loading && (
        <div className="flex items-center gap-3 text-sm text-ink/50" role="status">
          <span className="flex gap-1" aria-hidden="true">
            <i className="h-2 w-2 animate-pulse rounded-full bg-moss" />
            <i className="h-2 w-2 animate-pulse rounded-full bg-moss [animation-delay:150ms]" />
            <i className="h-2 w-2 animate-pulse rounded-full bg-moss [animation-delay:300ms]" />
          </span>
          Retrieving and verifying evidence…
        </div>
      )}
    </div>
  );
}
