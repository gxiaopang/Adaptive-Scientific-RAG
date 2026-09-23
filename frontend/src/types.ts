export type Evidence = {
  doc_id: string;
  title: string;
  excerpt: string;
  score: number;
  rank: number;
};

export type QueryResponse = {
  conversation_id: string;
  original_query: string;
  response_mode: "scientific_evidence" | "general_chat";
  routing_reason: string;
  final_query: string;
  query_history: string[];
  answer: string;
  evidence_sufficient: boolean | null;
  verification_reason: string | null;
  retrieval_attempts: number;
  termination_reason: string;
  evidence: Evidence[];
};

export type TimelineMessage =
  | { id: string; role: "user"; text: string }
  | { id: string; role: "assistant"; text: string; result: QueryResponse | null };

export type StoredConversation = {
  conversationId: string | null;
  messages: TimelineMessage[];
};
