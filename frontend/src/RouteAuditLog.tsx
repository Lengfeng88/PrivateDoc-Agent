import { useState, useEffect, useRef } from "react";
import type { RouteDecision } from "./lib/types";

export interface AuditEntry {
  id: string;
  ts: string;
  doc: string;
  query_snippet: string;
  action: string;
  route: RouteDecision;
  pii_types?: string[];
  cloud_blocked: boolean;
  latency_ms: number;
}

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
const POLL_MS = 5000;

// ─── Seed entries (shown while offline) ─────────────────────────────────────
const SEED: AuditEntry[] = [
  {
    id: "a1", ts: "09:14:02", doc: "Employee_Records.docx",
    query_snippet: "How many vacation days…",
    action: "PII redaction pass", route: "SENSITIVE",
    pii_types: ["PERSON", "EMAIL"], cloud_blocked: true, latency_ms: 312,
  },
  {
    id: "a2", ts: "09:14:03", doc: "Employee_Records.docx",
    query_snippet: "How many vacation days…",
    action: "Cloud edge blocked", route: "SENSITIVE",
    pii_types: [], cloud_blocked: true, latency_ms: 0,
  },
  {
    id: "a3", ts: "09:22:47", doc: "HR_Policy_2024.pdf",
    query_snippet: "What are the termination procedures…",
    action: "PII redaction pass", route: "SENSITIVE",
    pii_types: ["PERSON"], cloud_blocked: true, latency_ms: 289,
  },
];

// ─── Badge helpers ───────────────────────────────────────────────────────────
function PIIBadge({ type }: { type: string }) {
  return <span className="pii-badge">{type}</span>;
}

function BlockedBadge({ blocked }: { blocked: boolean }) {
  return blocked ? (
    <span className="blocked-badge">
      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
        <circle cx="12" cy="12" r="10" /><line x1="4.93" y1="4.93" x2="19.07" y2="19.07" />
      </svg>
      cloud blocked
    </span>
  ) : (
    <span className="allowed-badge">
      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
        <polyline points="20 6 9 17 4 12" />
      </svg>
      local only
    </span>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────
export default function RouteAuditLog() {
  const [entries, setEntries] = useState<AuditEntry[]>(SEED);
  const [filter, setFilter] = useState<"all" | "SENSITIVE">("all");
  const [search, setSearch] = useState("");
  const [flash, setFlash] = useState<Set<string>>(new Set());
  const prevIds = useRef<Set<string>>(new Set(SEED.map((e) => e.id)));
  const [exporting, setExporting] = useState(false);

  // poll for new entries
  useEffect(() => {
    const poll = async () => {
      try {
        const r = await fetch(`${API_BASE}/audit/log`);
        if (!r.ok) return;
        const data: AuditEntry[] = await r.json();
        const newIds = new Set<string>();
        data.forEach((e) => {
          if (!prevIds.current.has(e.id)) newIds.add(e.id);
        });
        if (newIds.size > 0) {
          setFlash(newIds);
          setEntries(data);
          newIds.forEach((id) => prevIds.current.add(id));
          setTimeout(() => setFlash(new Set()), 2000);
        }
      } catch {
        // offline — no-op
      }
    };
    const iv = setInterval(poll, POLL_MS);
    return () => clearInterval(iv);
  }, []);

  const exportCSV = async () => {
    setExporting(true);
    const rows = [
      ["time", "document", "query", "action", "route", "pii_types", "cloud_blocked", "latency_ms"],
      ...filtered.map((e) => [
        e.ts, e.doc, e.query_snippet, e.action, e.route,
        (e.pii_types ?? []).join("|"), String(e.cloud_blocked), String(e.latency_ms),
      ]),
    ];
    const csv = rows.map((r) => r.map((v) => `"${v}"`).join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "audit_log.csv"; a.click();
    URL.revokeObjectURL(url);
    setTimeout(() => setExporting(false), 800);
  };

  const filtered = entries.filter((e) => {
    if (filter === "SENSITIVE" && e.route !== "SENSITIVE") return false;
    if (search && !e.doc.toLowerCase().includes(search.toLowerCase()) &&
        !e.query_snippet.toLowerCase().includes(search.toLowerCase()) &&
        !e.action.toLowerCase().includes(search.toLowerCase())) return false;
    return true;
  });

  const sensitiveCount = entries.filter((e) => e.route === "SENSITIVE").length;
  const blockedCount = entries.filter((e) => e.cloud_blocked).length;

  return (
    <div className="audit-panel">
      <div className="audit-header">
        <div className="audit-title-row">
          <h2 className="audit-title">privacy route audit log</h2>
          <div className="audit-stats">
            <span className="audit-stat sensitive">{sensitiveCount} sensitive</span>
            <span className="audit-stat blocked">{blockedCount} cloud blocked</span>
            <span className="audit-stat total">{entries.length} total</span>
          </div>
        </div>

        <div className="audit-controls">
          <input
            className="audit-search"
            type="text"
            placeholder="Search doc, query, action…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <div className="filter-tabs">
            {(["all", "SENSITIVE"] as const).map((f) => (
              <button
                key={f}
                className={`filter-tab ${filter === f ? "active" : ""}`}
                onClick={() => setFilter(f)}
              >
                {f}
              </button>
            ))}
          </div>
          <button className="export-btn" onClick={exportCSV} disabled={exporting}>
            {exporting ? (
              <span className="btn-spinner small" />
            ) : (
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg>
            )}
            export CSV
          </button>
        </div>
      </div>

      {filtered.length === 0 ? (
        <div className="audit-empty">No entries match the current filter.</div>
      ) : (
        <div className="audit-table-wrap">
          <table className="audit-table">
            <thead>
              <tr>
                <th>time</th>
                <th>document</th>
                <th>query</th>
                <th>action</th>
                <th>PII types</th>
                <th>cloud</th>
                <th>latency</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((entry) => (
                <tr key={entry.id} className={flash.has(entry.id) ? "row-flash" : ""}>
                  <td className="td-ts">{entry.ts}</td>
                  <td className="td-doc">
                    <div className="doc-cell">
                      <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                        <polyline points="14 2 14 8 20 8" />
                      </svg>
                      {entry.doc}
                    </div>
                  </td>
                  <td className="td-query">{entry.query_snippet}</td>
                  <td className="td-action">{entry.action}</td>
                  <td className="td-pii">
                    {(entry.pii_types ?? []).map((p) => <PIIBadge key={p} type={p} />)}
                  </td>
                  <td className="td-blocked"><BlockedBadge blocked={entry.cloud_blocked} /></td>
                  <td className="td-latency">
                    {entry.latency_ms > 0 ? `${entry.latency_ms} ms` : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
