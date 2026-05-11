import { useState } from "react";
import UploadPanel from "./UploadPanel";
import AgentChat from "./AgentChat";
import EvalDashboard from "./EvalDashboard";
import RouteAuditLog from "./RouteAuditLog";
import "./app.css";

type Tab = "upload" | "chat" | "eval" | "audit";

const TABS: { id: Tab; label: string }[] = [
  { id: "upload", label: "Upload + route" },
  { id: "chat",   label: "Agent chat" },
  { id: "eval",   label: "Eval dashboard" },
  { id: "audit",  label: "Audit log" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("upload");

  return (
    <div className="app-root">
      <header className="app-header">
        <div className="header-inner">
          <div className="header-brand">
            <span className="brand-name">PrivateDoc Agent</span>
            <span className="brand-tag">local-first RAG</span>
          </div>
        </div>
      </header>
      <nav className="app-nav">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={`nav-tab ${tab === t.id ? "active" : ""}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>
      <main className="app-main">
        {tab === "upload" && <UploadPanel />}
        {tab === "chat"   && <AgentChat />}
        {tab === "eval"   && <EvalDashboard />}
        {tab === "audit"  && <RouteAuditLog />}
      </main>
    </div>
  );
}
