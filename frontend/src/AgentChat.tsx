import { useState, useRef, useEffect, useCallback } from "react";
import type { RouteDecision, ChunkRef } from "./lib/types";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  route?: RouteDecision;
  path?: string;
  latencyMs?: number;
  chunks?: ChunkRef[];
  streaming?: boolean;
}

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

const ROUTE_META: Record<RouteDecision, { label: string; color: string }> = {
  SIMPLE: { label: "local fast path", color: "route-simple" },
  COMPLEX: { label: "LangGraph agent", color: "route-complex" },
  SENSITIVE: { label: "local-only lock", color: "route-sensitive" },
};

function RouteTag({ route, latencyMs }: { route: RouteDecision; latencyMs?: number }) {
  const meta = ROUTE_META[route];
  return (
    <div className="route-tag-row">
      <span className={`route-tag ${meta.color}`}>
        <span className="route-dot" />
        {meta.label}
      </span>
      {latencyMs !== undefined && (
        <span className="latency">{latencyMs.toLocaleString()} ms</span>
      )}
    </div>
  );
}

function ChunkCard({ chunk }: { chunk: ChunkRef }) {
  return (
    <div className="chunk-card">
      <div className="chunk-source">
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
        </svg>
        {chunk.source} · p.{chunk.page}
      </div>
      <div className="chunk-text">{chunk.text}</div>
      {chunk.score !== undefined && (
        <div className="chunk-score">relevance {(chunk.score * 100).toFixed(0)}%</div>
      )}
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="typing-indicator">
      <span /><span /><span />
    </div>
  );
}

export default function AgentChat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "init",
      role: "assistant",
      content: "Documents indexed. Ask me anything across your knowledge base.",
      route: "SIMPLE",
      path: "local fast path",
      latencyMs: 0,
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput("");
    setLoading(true);

    const userMsg: Message = { id: crypto.randomUUID(), role: "user", content: text };
    const asstId = crypto.randomUUID();
    const asstMsg: Message = {
      id: asstId,
      role: "assistant",
      content: "",
      streaming: true,
    };

    setMessages((m) => [...m, userMsg, asstMsg]);

    const startTime = Date.now();

    try {
      abortRef.current = new AbortController();
      const res = await fetch(`${API_BASE}/query/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text }),
        signal: abortRef.current.signal,
      });

      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6).trim();
          if (raw === "[DONE]") continue;
          try {
            const ev = JSON.parse(raw);
            setMessages((prev) =>
              prev.map((m) => {
                if (m.id !== asstId) return m;
                // handle backend SSE format: {type, result, ...}
                if (ev.type === "done" && ev.result) {
                  return {
                    ...m,
                    content: ev.result.final_answer ?? m.content,
                    route: ev.result.route_badge?.label ?? m.route,
                    path: ev.result.used_backend === "local" ? "local fast path" : ev.result.used_backend,
                    chunks: ev.result.citations ?? m.chunks,
                    latencyMs: Date.now() - startTime,
                    streaming: false,
                  };
                }
                if (ev.type === "route") {
                  return { ...m, route: ev.route_badge?.label ?? m.route };
                }
                return {
                  ...m,
                  content: ev.token ? m.content + ev.token : m.content,
                  streaming: true,
                };
              })
            );
          } catch {
            // partial JSON, skip
          }
        }
      }
    } catch (err: unknown) {
      if ((err as Error)?.name === "AbortError") return;
      // offline demo: simulate a response
      const fakeResponses: Partial<Message>[] = [
        {
          content:
            "Revenue grew 14% YoY to $4.2M, driven by enterprise contracts. Operating margin compressed ~200bps due to infrastructure spend.",
          route: "COMPLEX",
          path: "LangGraph agent",
          chunks: [{ source: "Q4_Financials_2024.pdf", page: 4, text: "Enterprise ARR reached $3.1M, up from $2.7M in Q3.", score: 0.94 }],
        },
        {
          content:
            "Full-time employees receive 20 days PTO annually. Note: PII redaction applied — employee IDs stripped from context.",
          route: "SENSITIVE",
          path: "local-only lock",
          chunks: [{ source: "Employee_Records.docx", page: 2, text: "Annual leave entitlement: 20 days (FTE).", score: 0.91 }],
        },
        {
          content:
            "Section 3 of the Onboarding Guide covers dev environment setup: clone the repo, run `make dev`, then visit localhost:3000.",
          route: "SIMPLE",
          path: "local fast path",
          chunks: [{ source: "Onboarding_Guide.pdf", page: 7, text: "Run make dev to start all services locally.", score: 0.97 }],
        },
      ];
      const fake = fakeResponses[Math.floor(Math.random() * fakeResponses.length)];
      // stream fake token by token
      let streamed = "";
      for (const char of (fake.content ?? "")) {
        streamed += char;
        const snap = streamed;
        setMessages((prev) =>
          prev.map((m) => (m.id === asstId ? { ...m, content: snap } : m))
        );
        await new Promise((r) => setTimeout(r, 12));
      }
      setMessages((prev) =>
        prev.map((m) =>
          m.id === asstId
            ? { ...m, ...fake, content: fake.content ?? "", streaming: false, latencyMs: Date.now() - startTime }
            : m
        )
      );
    } finally {
      setLoading(false);
    }
  }, [input, loading]);

  const onKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div className="chat-panel">
      <div className="chat-messages">
        {messages.map((msg) => (
          <div key={msg.id} className={`chat-msg ${msg.role}`}>
            <div className="msg-avatar">
              {msg.role === "assistant" ? "AI" : "U"}
            </div>
            <div className="msg-body">
              <div className="msg-bubble">
                {msg.content || (msg.streaming && <TypingIndicator />)}
                {!msg.content && !msg.streaming && (
                  <span className="empty-bubble">…</span>
                )}
              </div>
              {msg.chunks && msg.chunks.length > 0 && (
                <div className="chunk-list">
                  {msg.chunks.map((c, i) => (
                    <ChunkCard key={i} chunk={c} />
                  ))}
                </div>
              )}
              {msg.route && !msg.streaming && (
                <RouteTag route={msg.route} latencyMs={msg.latencyMs} />
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-bar">
        <textarea
          ref={inputRef}
          className="chat-textarea"
          rows={1}
          placeholder="Ask a question… (Enter to send, Shift+Enter for newline)"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKey}
          disabled={loading}
        />
        <button
          className={`send-btn ${loading ? "loading" : ""}`}
          onClick={send}
          disabled={loading || !input.trim()}
          aria-label="Send message"
        >
          {loading ? (
            <span className="send-spinner" />
          ) : (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          )}
        </button>
      </div>
    </div>
  );
}
