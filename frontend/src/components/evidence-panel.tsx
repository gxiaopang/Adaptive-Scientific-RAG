import { BookOpen, CheckCircle2, CircleAlert } from "lucide-react";

import type { QueryResponse } from "../types";
import { Badge } from "./ui/badge";

export function EvidencePanel({ result }: { result: QueryResponse | null }) {
  if (result === null || result.response_mode !== "scientific_evidence") {
    return (
      <div className="rounded-3xl border border-dashed border-ink/15 p-6 text-sm leading-6 text-ink/55">
        Evidence appears here after a scientific question. General conversation stays uncluttered.
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <section className="rounded-3xl bg-ink p-5 text-white shadow-panel">
        <div className="flex items-start gap-3">
          {result.evidence_sufficient ? (
            <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-citron" />
          ) : (
            <CircleAlert className="mt-0.5 h-5 w-5 shrink-0 text-amber-300" />
          )}
          <div>
            <p className="text-sm font-semibold">
              {result.evidence_sufficient ? "Evidence verified" : "Evidence remains limited"}
            </p>
            <p className="mt-2 text-sm leading-6 text-white/65">
              {result.verification_reason ?? "No verification reason was returned."}
            </p>
          </div>
        </div>
        <div className="mt-4 flex flex-wrap gap-2">
          <Badge className="border-white/10 bg-white/10 text-white/70">
            {result.retrieval_attempts} retrieval {result.retrieval_attempts === 1 ? "round" : "rounds"}
          </Badge>
          <Badge className="border-white/10 bg-white/10 text-white/70">
            {result.termination_reason.replaceAll("_", " ")}
          </Badge>
        </div>
      </section>

      {result.query_history.length > 1 && (
        <section aria-labelledby="query-history-title">
          <p id="query-history-title" className="mb-2 text-xs font-bold uppercase tracking-[0.16em] text-ink/45">
            Query path
          </p>
          <ol className="space-y-2 border-l border-ink/15 pl-4 text-sm text-ink/65">
            {result.query_history.map((query, index) => (
              <li key={`${index}-${query}`}>
                <span className="mr-2 font-mono text-[10px] text-moss">0{index + 1}</span>
                {query}
              </li>
            ))}
          </ol>
        </section>
      )}

      <section aria-labelledby="sources-title">
        <div className="mb-3 flex items-center justify-between">
          <p id="sources-title" className="text-xs font-bold uppercase tracking-[0.16em] text-ink/45">
            Evidence library
          </p>
          <span className="font-mono text-xs text-ink/40">{result.evidence.length} sources</span>
        </div>
        {result.evidence.length === 0 ? (
          <div className="rounded-3xl border border-dashed border-ink/15 p-5 text-sm text-ink/55">
            No evidence passages were returned for this answer.
          </div>
        ) : (
          <ol className="space-y-3">
            {result.evidence.map((evidence) => (
              <li key={`${evidence.rank}-${evidence.doc_id}`} className="rounded-3xl border border-ink/10 bg-white/65 p-4 shadow-sm">
                <div className="flex items-start gap-3">
                  <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-citron/70 font-mono text-xs font-bold text-ink">
                    {evidence.rank}
                  </span>
                  <div className="min-w-0">
                    <h3 className="font-serif text-lg leading-5 text-ink">{evidence.title || `Document ${evidence.doc_id}`}</h3>
                    <p className="mt-2 text-sm leading-6 text-ink/60">{evidence.excerpt}</p>
                    <div className="mt-3 flex items-center gap-2 text-[11px] text-ink/40">
                      <BookOpen className="h-3.5 w-3.5" />
                      <span>{evidence.doc_id}</span>
                      <span>·</span>
                      <span>score {evidence.score.toFixed(3)}</span>
                    </div>
                  </div>
                </div>
              </li>
            ))}
          </ol>
        )}
      </section>
    </div>
  );
}
