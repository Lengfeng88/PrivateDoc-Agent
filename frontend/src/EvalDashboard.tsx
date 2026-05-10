import { useState, useEffect, useRef } from "react";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";

interface EvalPoint {
  query: string;
  faithfulness: number;
  answer_relevancy: number;
  context_recall: number;
}

interface CostPoint {
  ts: string;
  local: number;
  cloud: number;
}

interface RouteStats {
  SIMPLE: number;
  COMPLEX: number;
  SENSITIVE: number;
}

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

// ─── Static seed data (offline demo) ───────────────────────────────────────
const SEED_EVAL: EvalPoint[] = Array.from({ length: 12 }, (_, i) => ({
  query: `Q${i + 1}`,
  faithfulness: +(0.82 + Math.sin(i * 0.7) * 0.06 + Math.random() * 0.04).toFixed(2),
  answer_relevancy: +(0.78 + Math.cos(i * 0.5) * 0.07 + Math.random() * 0.04).toFixed(2),
  context_recall: +(0.85 + Math.sin(i * 0.9) * 0.05 + Math.random() * 0.03).toFixed(2),
}));

const SEED_COST: CostPoint[] = Array.from({ length: 8 }, (_, i) => ({
  ts: `${9 + i}:00`,
  local: 0,
  cloud: parseFloat((i * 0.007 + Math.random() * 0.003).toFixed(3)),
}));

const SEED_ROUTES: RouteStats = { SIMPLE: 7, COMPLEX: 4, SENSITIVE: 1 };

// ─── Stat card ──────────────────────────────────────────────────────────────
function StatCard({ label, value, sub, accent }: {
  label: string; value: string; sub?: string; accent?: string
}) {
  return (
    <div className="stat-card">
      <div className="stat-label">{label}</div>
      <div className="stat-value" style={accent ? { color: accent } : {}}>{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

// ─── Route donut (pure SVG) ──────────────────────────────────────────────────
function RouteDonut({ stats }: { stats: RouteStats }) {
  const total = stats.SIMPLE + stats.COMPLEX + stats.SENSITIVE;
  if (total === 0) return null;
  const segments = [
    { key: "SIMPLE", val: stats.SIMPLE, color: "#5DCAA5" },
    { key: "COMPLEX", val: stats.COMPLEX, color: "#85B7EB" },
    { key: "SENSITIVE", val: stats.SENSITIVE, color: "#F09595" },
  ];
  const r = 40, cx = 50, cy = 50, gap = 2;
  let angle = -90;
  const arcs = segments.map((s) => {
    const sweep = (s.val / total) * 360 - gap;
    const start = angle;
    angle += (s.val / total) * 360;
    const a1 = (start * Math.PI) / 180;
    const a2 = ((start + sweep) * Math.PI) / 180;
    const x1 = cx + r * Math.cos(a1), y1 = cy + r * Math.sin(a1);
    const x2 = cx + r * Math.cos(a2), y2 = cy + r * Math.sin(a2);
    const large = sweep > 180 ? 1 : 0;
    return { ...s, d: `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}` };
  });

  return (
    <div className="donut-wrap">
      <svg viewBox="0 0 100 100" width="100" height="100">
        {arcs.map((a) => (
          <path
            key={a.key}
            d={a.d}
            fill="none"
            stroke={a.color}
            strokeWidth="14"
            strokeLinecap="round"
          />
        ))}
        <text x="50" y="47" textAnchor="middle" fontSize="13" fontWeight="500"
          fill="var(--color-text-primary)">{total}</text>
        <text x="50" y="59" textAnchor="middle" fontSize="8"
          fill="var(--color-text-secondary)">queries</text>
      </svg>
      <div className="donut-legend">
        {segments.map((s) => (
          <div key={s.key} className="donut-legend-row">
            <span className="donut-dot" style={{ background: s.color }} />
            <span className="donut-key">{s.key}</span>
            <span className="donut-pct">{((s.val / total) * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Custom tooltip ──────────────────────────────────────────────────────────
function ChartTooltip({ active, payload, label }: {
  active?: boolean; payload?: { color: string; name: string; value: number }[]; label?: string
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <div className="tooltip-label">{label}</div>
      {payload.map((p) => (
        <div key={p.name} className="tooltip-row">
          <span className="tooltip-dot" style={{ background: p.color }} />
          <span className="tooltip-name">{p.name}</span>
          <span className="tooltip-val">{p.value.toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}

// ─── Main component ──────────────────────────────────────────────────────────
export default function EvalDashboard() {
  const [evalData, setEvalData] = useState<EvalPoint[]>(SEED_EVAL);
  const [costData, setCostData] = useState<CostPoint[]>(SEED_COST);
  const [routes, setRoutes] = useState<RouteStats>(SEED_ROUTES);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const avg = (key: keyof EvalPoint) =>
    (evalData.reduce((s, d) => s + (d[key] as number), 0) / evalData.length).toFixed(2);

  const totalCloud = costData.reduce((s, d) => s + d.cloud, 0);
  const totalLocal = 0;

  const runEval = async () => {
    if (running) return;
    setRunning(true);
    setProgress(0);

    // animate progress bar
    let p = 0;
    intervalRef.current = setInterval(() => {
      p += Math.random() * 8 + 2;
      if (p >= 100) { p = 100; clearInterval(intervalRef.current!); }
      setProgress(Math.min(p, 100));
    }, 300);

    try {
      const r = await fetch(`${API_BASE}/eval/run`, { method: "POST" });
      if (r.ok) {
        const j = await r.json();
        setEvalData(j.ragas_scores);
        setCostData(j.cost_history);
        setRoutes(j.route_stats);
      }
    } catch {
      // offline: generate fresh seed data
      await new Promise((r) => setTimeout(r, 3000));
      setEvalData(Array.from({ length: 12 }, (_, i) => ({
        query: `Q${i + 1}`,
        faithfulness: +(0.80 + Math.sin(i * 0.8) * 0.08 + Math.random() * 0.04).toFixed(2),
        answer_relevancy: +(0.76 + Math.cos(i * 0.6) * 0.09 + Math.random() * 0.03).toFixed(2),
        context_recall: +(0.83 + Math.sin(i) * 0.06 + Math.random() * 0.04).toFixed(2),
      })));
      setRoutes({ SIMPLE: 8, COMPLEX: 3, SENSITIVE: 1 });
    } finally {
      clearInterval(intervalRef.current!);
      setProgress(100);
      setTimeout(() => { setRunning(false); setProgress(0); }, 600);
    }
  };

  useEffect(() => () => { if (intervalRef.current) clearInterval(intervalRef.current); }, []);

  return (
    <div className="eval-panel">
      {/* Header row */}
      <div className="eval-header">
        <h2 className="eval-title">RAGAS evaluation</h2>
        <button className={`run-eval-btn ${running ? "running" : ""}`} onClick={runEval} disabled={running}>
          {running ? (
            <><span className="btn-spinner" />running eval…</>
          ) : (
            <><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polygon points="5 3 19 12 5 21 5 3" />
            </svg> run eval</>
          )}
        </button>
      </div>

      {running && (
        <div className="eval-progress-bar">
          <div className="eval-progress-fill" style={{ width: `${progress}%` }} />
        </div>
      )}

      {/* Stat cards */}
      <div className="stat-grid">
        <StatCard label="faithfulness" value={avg("faithfulness")} sub="RAGAS avg" accent="#0F6E56" />
        <StatCard label="answer relevancy" value={avg("answer_relevancy")} sub="RAGAS avg" accent="#185FA5" />
        <StatCard label="context recall" value={avg("context_recall")} sub="RAGAS avg" accent="#854F0B" />
        <StatCard
          label="cloud spend"
          value={`$${totalCloud.toFixed(3)}`}
          sub="vs $0.000 local"
          accent="#185FA5"
        />
      </div>

      {/* RAGAS line chart */}
      <div className="chart-section">
        <div className="chart-title">score per query</div>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={evalData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)" />
            <XAxis dataKey="query" tick={{ fontSize: 11, fill: "var(--color-text-secondary)" }} />
            <YAxis domain={[0.6, 1]} tick={{ fontSize: 11, fill: "var(--color-text-secondary)" }} />
            <Tooltip content={<ChartTooltip />} />
            <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12 }} />
            <Line type="monotone" dataKey="faithfulness" stroke="#0F6E56" strokeWidth={1.5} dot={false} />
            <Line type="monotone" dataKey="answer_relevancy" stroke="#185FA5" strokeWidth={1.5} dot={false} />
            <Line type="monotone" dataKey="context_recall" stroke="#854F0B" strokeWidth={1.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Cost bar chart + route donut */}
      <div className="bottom-row">
        <div className="chart-section" style={{ flex: 2 }}>
          <div className="chart-title">local vs cloud cost ($)</div>
          <ResponsiveContainer width="100%" height={140}>
            <BarChart data={costData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }} barSize={10}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)" />
              <XAxis dataKey="ts" tick={{ fontSize: 11, fill: "var(--color-text-secondary)" }} />
              <YAxis tick={{ fontSize: 11, fill: "var(--color-text-secondary)" }} />
              <Tooltip content={<ChartTooltip />} />
              <Legend iconType="square" iconSize={8} wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="local" fill="#5DCAA5" radius={[2, 2, 0, 0]} />
              <Bar dataKey="cloud" fill="#85B7EB" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="chart-section donut-section" style={{ flex: 1 }}>
          <div className="chart-title">route distribution</div>
          <RouteDonut stats={routes} />
        </div>
      </div>
    </div>
  );
}
