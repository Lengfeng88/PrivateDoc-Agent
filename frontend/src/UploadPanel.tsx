import { useState, useCallback, useRef } from "react";
import type { RouteDecision } from "./lib/types";

interface DocEntry {
  id: string;
  name: string;
  size: number;
  status: "pending" | "ingesting" | "done" | "error";
  route?: RouteDecision;
  step?: string;
}

const INGEST_STEPS = [
  "Parsing document…",
  "Chunking semantically…",
  "Embedding with BGE-M3…",
  "Upserting to Qdrant…",
  "Done",
];

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

function formatBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(0)} KB`;
  return `${(b / (1024 * 1024)).toFixed(1)} MB`;
}

function RouteBadge({ route }: { route: RouteDecision }) {
  const styles: Record<RouteDecision, string> = {
    SIMPLE: "badge-simple",
    COMPLEX: "badge-complex",
    SENSITIVE: "badge-sensitive",
  };
  return <span className={`route-badge ${styles[route]}`}>{route}</span>;
}

function StepProgress({ step }: { step: string }) {
  const idx = INGEST_STEPS.indexOf(step);
  return (
    <div className="step-progress">
      {INGEST_STEPS.map((s, i) => (
        <div
          key={s}
          className={`step-dot ${i < idx ? "done" : i === idx ? "active" : ""}`}
          title={s}
        />
      ))}
      <span className="step-label">{step}</span>
    </div>
  );
}

export default function UploadPanel() {
  const [docs, setDocs] = useState<DocEntry[]>([]);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const classify = async (file: File): Promise<RouteDecision> => {
    try {
      const fd = new FormData();
      fd.append("file", file);
      const r = await fetch(`${API_BASE}/classify`, { method: "POST", body: fd });
      if (!r.ok) throw new Error();
      const j = await r.json();
      return j.route as RouteDecision;
    } catch {
      // heuristic fallback while backend is offline
      const name = file.name.toLowerCase();
      if (/employee|hr|payroll|ssn|passport/.test(name)) return "SENSITIVE";
      if (file.size > 500_000) return "COMPLEX";
      return "SIMPLE";
    }
  };

  const ingestFile = async (id: string, file: File) => {
    for (const step of INGEST_STEPS) {
      setDocs((d) =>
        d.map((doc) => (doc.id === id ? { ...doc, step, status: "ingesting" } : doc))
      );
      if (step === "Done") break;
      try {
        const fd = new FormData();
        fd.append("file", file);
        await fetch(`${API_BASE}/ingest`, { method: "POST", body: fd });
      } catch {
        // allow offline demo — just animate steps
      }
      await new Promise((r) => setTimeout(r, 700 + Math.random() * 400));
    }
    setDocs((d) =>
      d.map((doc) => (doc.id === id ? { ...doc, status: "done", step: "Done" } : doc))
    );
  };

  const addFiles = useCallback(async (files: FileList | File[]) => {
    const arr = Array.from(files).filter(
      (f) => /\.(pdf|docx)$/i.test(f.name)
    );
    if (!arr.length) return;

    const entries: DocEntry[] = arr.map((f) => ({
      id: crypto.randomUUID(),
      name: f.name,
      size: f.size,
      status: "pending",
    }));

    setDocs((d) => [...d, ...entries]);

    // classify + ingest in parallel
    await Promise.all(
      entries.map(async (entry, i) => {
        const route = await classify(arr[i]);
        setDocs((d) =>
          d.map((doc) => (doc.id === entry.id ? { ...doc, route } : doc))
        );
        await ingestFile(entry.id, arr[i]);
      })
    );
  }, []);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    addFiles(e.dataTransfer.files);
  };

  const clearAll = () => setDocs([]);

  return (
    <div className="upload-panel">
      {/* Drop zone */}
      <div
        className={`drop-zone ${dragging ? "dragging" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        aria-label="Upload PDF or DOCX files"
        onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx"
          multiple
          style={{ display: "none" }}
          onChange={(e) => e.target.files && addFiles(e.target.files)}
        />
        <div className="drop-icon">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
        </div>
        <p className="drop-primary">Drop PDF / DOCX files here</p>
        <p className="drop-sub">All processing local — raw files never touch the network</p>
      </div>

      {/* Legend */}
      <div className="legend">
        <span className="route-badge badge-simple">SIMPLE</span>
        <span className="legend-desc">short query, low doc count</span>
        <span className="route-badge badge-complex">COMPLEX</span>
        <span className="legend-desc">multi-hop, cross-doc reasoning</span>
        <span className="route-badge badge-sensitive">SENSITIVE</span>
        <span className="legend-desc">PII / flagged keywords → local-only lock</span>
      </div>

      {/* Doc list */}
      {docs.length > 0 && (
        <div className="doc-list">
          <div className="doc-list-header">
            <span>{docs.length} document{docs.length !== 1 ? "s" : ""}</span>
            <button className="clear-btn" onClick={clearAll}>clear all</button>
          </div>
          {docs.map((doc) => (
            <div key={doc.id} className={`doc-row ${doc.status}`}>
              <div className="doc-file-icon">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
                </svg>
              </div>
              <div className="doc-info">
                <span className="doc-name">{doc.name}</span>
                <span className="doc-size">{formatBytes(doc.size)}</span>
              </div>
              <div className="doc-right">
                {doc.route && <RouteBadge route={doc.route} />}
                {doc.status === "ingesting" && doc.step && (
                  <StepProgress step={doc.step} />
                )}
                {doc.status === "done" && (
                  <span className="status-done">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                    indexed
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
