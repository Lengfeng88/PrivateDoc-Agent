# PrivateDoc Agent

**Local-first, privacy-preserving RAG with cost-aware routing.**  
Raw documents never touch the network at any stage.

---

## What it does

Upload PDFs and DOCX files. Ask questions. Every query is classified into one of three routes before a single token is generated:

| Route | When | Execution | Cost |
|---|---|---|---|
| **SIMPLE** | Short, single-hop query | Llama 3.1 8B via llama.cpp | $0.000 |
| **COMPLEX** | Multi-hop, cross-doc reasoning | LangGraph Retrieve→Reason→Critic→Report | ~$0.008 (Azure OAI fallback) |
| **SENSITIVE** | PII / flagged keywords detected | Local-only lock — cloud edge blocked, PII redacted | $0.000 |

The classifier is rule-based — zero ML training needed, works offline, deterministic.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Ingestion layer — all processing local                  │
│  PDF/DOCX → semantic chunker → BGE-M3 embed → Qdrant    │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  Sensitivity classifier  (rule-based, zero ML)           │
│  SIMPLE          COMPLEX          SENSITIVE              │
└──────┬──────────────┬────────────────────┬──────────────┘
       │              │                    │
  ┌────▼────┐   ┌─────▼──────┐   ┌────────▼────────┐
  │  Local  │   │  LangGraph │   │  Local-only lock │
  │ fast    │   │  agent     │   │  PII redacted    │
  │ path    │   │  3-hop max │   │  cloud blocked   │
  └────┬────┘   └─────┬──────┘   └────────┬────────┘
       └──────────────┴────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  Observability + delivery                                │
│  LangSmith traces · RAGAS eval · cost tracker · FastAPI │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  React + Vite demo UI (Azure Static Web Apps)            │
│  Upload+route · Agent chat (SSE) · Eval dashboard        │
└─────────────────────────────────────────────────────────┘
```

---

## Quickstart — local (Docker)

**Prerequisites:** Docker, NVIDIA GPU + drivers (CPU fallback works without GPU).

```bash
# 1. Clone
git clone https://github.com/yourhandle/privatedoc-agent.git
cd privatedoc-agent

# 2. Configure
cp .env.example .env
# Edit .env — set LLAMA_MODEL_FILE, optionally AZURE_OPENAI_* for cloud fallback

# 3. Download model (once, ~4.7 GB)
mkdir -p models
wget -P models https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

# 4. Start
docker compose up -d

# 5. Verify
curl http://localhost:8000/health
# → {"status":"ok","qdrant":"connected","llm":"connected"}

# 6. (Optional) Start Vite dev server
docker compose --profile frontend up frontend
# → http://localhost:5173
```

**Services started:**

| Service | URL | Notes |
|---|---|---|
| FastAPI API | `http://localhost:8000` | `/docs` for Swagger UI |
| Qdrant | `http://localhost:6333` | Web UI at `/dashboard` |
| llama.cpp server | `http://localhost:8080` | OpenAI-compatible `/v1/` |
| Vite frontend | `http://localhost:5173` | profile: frontend only |

---

## Quickstart — Python (no Docker)

```bash
# Requires: Python 3.11+, running Qdrant and llama-server
pip install -e ".[dev,eval,notebooks]"
python -m spacy download en_core_web_sm

export QDRANT_URL=http://localhost:6333
export LLAMA_SERVER_URL=http://localhost:8080

uvicorn api.main:app --reload --port 8000
```

---

## Project layout

```
privatedoc-agent/
│
├── ingest/
│   ├── loader.py          # Unstructured.io: PDF/DOCX → raw text + metadata
│   ├── chunker.py         # Semantic chunking: heading boundaries + 512-token sliding window
│   ├── embedder.py        # BGE-M3 batch embed, GPU-accelerated, Qdrant upsert
│   └── pii_detector.py   # Regex + spaCy NER: SSN/email/names → sensitivity flag
│
├── router/
│   ├── classifier.py      # Core routing logic — zero ML, fully deterministic
│   ├── rules.py           # Configurable: keyword lists, length thresholds, PII triggers
│   └── schemas.py         # RouteDecision dataclass: {route, confidence, reason}
│
├── agent/
│   ├── graph.py           # LangGraph StateGraph — 3 paths share the same graph
│   ├── nodes/
│   │   ├── retrieve.py    # Hybrid search: Qdrant dense + BM25, RRF rerank
│   │   ├── reason.py      # LLM reasoning node — local/cloud dual-backend
│   │   ├── critic.py      # Self-check: is the answer grounded? Re-retrieve if not
│   │   └── report.py      # Final answer + cited chunk IDs + confidence score
│   ├── backends/
│   │   ├── local_llm.py   # llama.cpp HTTP client (Llama 3.1 8B AWQ, ~80 tok/s)
│   │   └── azure_oai.py   # Azure OpenAI client with retry + token cost tracking
│   ├── tools.py           # calculator, date_parser, doc_lookup
│   └── prompts.py         # System + user prompt templates for all nodes
│
├── api/
│   ├── main.py            # FastAPI: /ingest /query /stream /classify /health /metrics
│   ├── schemas.py         # QueryRequest / QueryResponse / RouteInfo Pydantic models
│   └── cost_logger.py     # Per-request log: route, tokens, latency, $ cost
│
├── eval/
│   ├── ground_truth.json  # 25 hand-written Q&A pairs (Apple/Shopify public filings)
│   ├── ragas_eval.py      # RAGAS: faithfulness / answer_relevancy / context_recall
│   ├── route_eval.py      # Classifier accuracy tests (pytest, ≥ 90% gate in CI)
│   └── benchmark.py      # Latency + token cost profiling → CSV output
│
├── frontend/              # React + Vite (TypeScript)
│   └── src/
│       ├── UploadPanel.tsx     # Drag-drop upload + live route badge
│       ├── AgentChat.tsx       # SSE streaming + cited chunk cards
│       ├── EvalDashboard.tsx   # Recharts: RAGAS scores + cost charts
│       └── RouteAuditLog.tsx   # SENSITIVE route audit table + CSV export
│
├── infra/
│   ├── docker-compose.yml  # Local: Qdrant + llama-server + FastAPI + Vite
│   ├── Dockerfile          # Multi-stage: builder → runtime (~350 MB, non-root)
│   └── main.bicep          # Azure: Container Apps + Static Web Apps + Key Vault
│
├── .github/workflows/
│   ├── ci.yml              # lint → route_eval (≥90%) → RAGAS gate → docker build
│   └── deploy.yml          # Triggered on CI pass: ACR push → Bicep deploy → SWA
│
├── notebooks/
│   └── demo.ipynb          # End-to-end demo: ingest → classify → query → RAGAS viz
│
├── .env.example
├── pyproject.toml
└── README.md
```

---

## API reference

All endpoints served at `http://localhost:8000`. Interactive docs at `/docs`.

### `POST /ingest`

Upload a PDF or DOCX file for ingestion.

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@Q4_Financials.pdf"
```

```json
{
  "status": "ok",
  "chunks_indexed": 24,
  "source": "Q4_Financials.pdf",
  "pii_detected": false,
  "route_hint": "COMPLEX"
}
```

### `POST /classify`

Classify a query (or file) without executing it.

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare Apple and Shopify revenue growth"}'
```

```json
{
  "route": "COMPLEX",
  "confidence": 0.91,
  "reason": "2 complexity signals, multi_doc=true"
}
```

### `POST /query`

Blocking query — returns full answer when complete.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What was Apple revenue in 2024?", "top_k": 3}'
```

```json
{
  "answer": "Apple's total net sales were $391.0 billion in fiscal 2024...",
  "route": {"route": "SIMPLE", "confidence": 0.91, "reason": "Short single-hop query"},
  "chunks": [
    {"source": "Apple_10K_2024.pdf", "page": 4, "text": "...", "score": 0.97}
  ],
  "latency_ms": 218,
  "cost_usd": 0.0
}
```

### `POST /stream`

SSE streaming — tokens arrive as `data:` events.

```bash
curl -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze R&D spending across both companies"}' \
  --no-buffer
```

```
data: {"token": "Apple"}
data: {"token": " spent"}
data: {"token": " $31.4B"}
...
data: {"route": "COMPLEX", "path": "LangGraph agent", "chunks": [...], "done": true}
data: [DONE]
```

### `GET /audit/log`

Returns the SENSITIVE route audit log (JSON array).

### `GET /metrics`

Prometheus metrics endpoint — route counts, latency histograms, token totals.

### `GET /health`

Liveness check — returns 200 when Qdrant and llama-server are reachable.

---

## Evaluation

### Run locally

```bash
# Route classifier accuracy
pytest eval/route_eval.py -v
# → must pass ≥ 90% accuracy to unblock CI

# RAGAS eval (requires Qdrant + LLM)
python eval/ragas_eval.py --output-json reports/ragas_scores.json

# Latency + cost benchmark
python eval/benchmark.py --n-queries 50 --output-csv reports/benchmark.csv
```

### CI gates

CI in `.github/workflows/ci.yml` enforces hard thresholds before any deploy:

| Metric | Threshold |
|---|---|
| Route classifier accuracy | ≥ 90% |
| RAGAS faithfulness | ≥ 0.70 |
| RAGAS answer relevancy | ≥ 0.65 |
| RAGAS context recall | ≥ 0.70 |

### Current scores (demo notebook)

| Metric | Score |
|---|---|
| Route classifier accuracy | 100% (8/8 test queries) |
| RAGAS faithfulness | 0.91 |
| RAGAS answer relevancy | 0.88 |
| RAGAS context recall | 0.93 |

---

## Deploy to Azure

```bash
# 1. Deploy infrastructure (first time)
az group create -n privatedoc-rg -l eastus
az deployment group create \
  -g privatedoc-rg \
  -f infra/main.bicep \
  -p @infra/params.json

# 2. Configure GitHub Secrets (see infra/README_secrets.md)
#    AZURE_CREDENTIALS, ACR_LOGIN_SERVER, ACR_USERNAME, ACR_PASSWORD,
#    AZURE_STATIC_WEB_APPS_API_TOKEN, API_BASE_URL

# 3. Push to main — CI runs, then deploy.yml triggers automatically
git push origin main
```

The deploy pipeline:

```
push to main
  → ci.yml: lint → route_eval → RAGAS gate → docker build smoke test
  → deploy.yml: ACR push → Bicep deploy → SWA deploy → health check
```

---

## Configuration

All configuration is via environment variables (see `.env.example`).

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `LLAMA_SERVER_URL` | `http://localhost:8080` | llama.cpp server endpoint |
| `AZURE_OPENAI_ENDPOINT` | _(empty)_ | Azure OAI URL — leave blank to disable cloud fallback |
| `AZURE_OPENAI_API_KEY` | _(empty)_ | Azure OAI key |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-4o` | Deployment name |
| `LANGCHAIN_TRACING_V2` | `false` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | _(empty)_ | LangSmith API key |
| `PII_REDACT_ENABLED` | `true` | Run PII redaction pass on SENSITIVE route |
| `MAX_RETRIEVAL_LOOPS` | `3` | LangGraph critic max re-retrieval iterations |
| `AUDIT_LOG_PATH` | `./logs/audit.jsonl` | SENSITIVE route audit log location |
| `LOG_LEVEL` | `INFO` | Uvicorn log level |

---

## Hardware

Developed and benchmarked on:

| Component | Spec |
|---|---|
| CPU | Intel Core i9-13900 |
| GPU | NVIDIA RTX 4080 12 GB |
| RAM | 64 GB DDR5 |
| OS | Ubuntu 24.04 |
| Inference | Llama 3.1 8B Q4\_K\_M — ~80 tok/s on 4080 |
| Embedding | BGE-M3 — batch of 32 in ~0.4s on 4080 |

---

## License

MIT — see [LICENSE](LICENSE).
