"""
api/main.py
-----------
FastAPI application for the PrivateDoc Intelligence Agent.

Endpoints
---------
  POST /ingest            Upload a PDF or DOCX, chunk and embed it.
  POST /query             Blocking query — returns complete answer + citations.
  POST /query/stream      SSE streaming query — emits node-level events.
  GET  /health            Backend reachability check.
  GET  /metrics           In-memory usage statistics.

Architecture notes
------------------
- The blocking /query and the streaming /query/stream share the same
  `_build_initial_state` and router logic.  The only difference is the
  delivery mechanism: blocking collects the full result then returns it;
  streaming emits an SSE event after each LangGraph node completes.

- SSE streaming uses LangGraph's `.stream()` method which yields one
  dict per node (e.g. {"retrieve": {...state fields...}}).  The API
  translates each node output into a typed SSEEvent and serialises it
  as `data: <json>\n\n`.

- The ingest endpoint is synchronous but runs in a thread pool via
  FastAPI's default executor — embedding 100+ chunks at 12k tokens/s
  on the RTX 4080 takes < 2 s for a typical 30-page document.

- No authentication: this is a demo API.  Production hardening would
  add an API key header middleware and rate limiting.

Running locally
---------------
    docker run -p 6333:6333 qdrant/qdrant          # start Qdrant
    ./llama-server -m model.gguf --port 8080        # start local LLM
    uvicorn api.main:app --reload --port 8000

Environment variables (all optional with safe defaults)
-------------------------------------------------------
    QDRANT_URL              http://localhost:6333
    QDRANT_COLLECTION       privatedoc
    LOCAL_LLM_URL           http://localhost:8080
    AZURE_OAI_ENDPOINT      (required for cloud fallback)
    AZURE_OAI_KEY           (required for cloud fallback)
    AZURE_OAI_DEPLOYMENT    gpt-4o
    AGENT_MAX_LOOPS         3
    LOG_PATH                logs/queries.jsonl
    CORS_ORIGINS            * (restrict in production)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from sse_starlette.sse import EventSourceResponse

from api.schemas import (
    CitationCard,
    HealthResponse,
    BackendStatus,
    IngestRequest,
    IngestResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
    RouteBadge,
    SSEChunksEvent,
    SSECriticEvent,
    SSEDoneEvent,
    SSEErrorEvent,
    SSENodeStartEvent,
    SSERouteEvent,
)
from api.cost_logger import init_metrics, metrics
from router.classifier import SensitivityClassifier
from router.schemas import RouteDecision

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

from contextlib import asynccontextmanager

@asynccontextmanager
async def _lifespan(app: FastAPI):
    log_path = os.getenv("LOG_PATH", "logs/queries.jsonl")
    init_metrics(log_path=log_path)
    logger.info("PrivateDoc API started")
    yield


_CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

app = FastAPI(lifespan=_lifespan,
    title="PrivateDoc Intelligence Agent",
    description=(
        "Privacy-preserving agentic RAG with local/cloud cost-aware routing. "
        "Raw documents never leave the machine."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module-level classifier — stateless, safe to share across requests
_classifier = SensitivityClassifier(
    audit_log_path=os.getenv("AUDIT_LOG_PATH", "logs/audit.jsonl")
)


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------

@app.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Upload and index a PDF or DOCX file",
    status_code=status.HTTP_200_OK,
)
async def ingest(
    file: UploadFile = File(..., description="PDF or DOCX file to ingest"),
    collection_name: str = Form(default="privatedoc"),
    run_pii_detection: bool = Form(default=True),
) -> IngestResponse:
    """
    Ingestion pipeline:
      1. Save the upload to a temp file.
      2. Run DocumentLoader → SemanticChunker.
      3. Embed chunks with BGE-M3 (GPU-accelerated on local machine).
      4. Upsert into Qdrant with full payload.

    The file is deleted from disk after ingestion — only vectors persist.
    """
    t0 = time.perf_counter()
    filename = file.filename or "upload"

    # Validate extension early
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".docx", ".doc"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type '{suffix}'. Accepted: .pdf, .docx",
        )

    # Write to a temp path
    tmp_dir = Path("/tmp/privatedoc")
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / f"{time.time_ns()}{suffix}"

    try:
        content = await file.read()
        tmp_path.write_bytes(content)
        logger.info(f"[ingest] saved {filename} ({len(content):,} bytes) → {tmp_path}")

        # Run ingest pipeline (sync — runs in FastAPI's thread pool)
        result = _run_ingest_pipeline(
            tmp_path=tmp_path,
            filename=filename,
            collection_name=collection_name,
            run_pii_detection=run_pii_detection,
        )

    finally:
        if tmp_path.exists():
            tmp_path.unlink()   # never keep raw files on disk

    duration = time.perf_counter() - t0
    result["duration_seconds"] = round(duration, 3)
    logger.success(
        f"[ingest] {filename}: {result['num_chunks']} chunks in {duration:.2f}s"
    )
    return IngestResponse(**result)


def _run_ingest_pipeline(
    tmp_path: Path,
    filename: str,
    collection_name: str,
    run_pii_detection: bool,
) -> dict:
    """
    Synchronous ingest — called in FastAPI's default thread pool executor.
    Returns a partial IngestResponse dict (without duration_seconds).
    """
    from ingest.loader import DocumentLoader
    from ingest.chunker import SemanticChunker

    loader = DocumentLoader(run_pii_detection=run_pii_detection)
    elements = loader.load(tmp_path)

    chunker = SemanticChunker()
    chunks = chunker.chunk(elements)

    pii_count = sum(1 for c in chunks if c.pii_flagged)
    file_hash = chunks[0].file_hash if chunks else "unknown"

    # Upsert to Qdrant (graceful degradation if Qdrant is unreachable)
    _upsert_to_qdrant(chunks, collection_name)

    return {
        "status": "ok",
        "filename": filename,
        "file_hash": file_hash,
        "num_elements": len(elements),
        "num_chunks": len(chunks),
        "pii_flagged_chunks": pii_count,
        "collection_name": collection_name,
        "error": "",
    }


def _upsert_to_qdrant(chunks, collection_name: str) -> None:
    """
    Embed chunks with BGE-M3 and upsert into Qdrant.
    Logs a warning and continues if Qdrant is unreachable — ingest response
    still returns 200 so the caller knows chunking succeeded.
    """
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance, VectorParams,
            PointStruct, OptimizersConfigDiff,
        )
        from sentence_transformers import SentenceTransformer

        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        client = QdrantClient(url=qdrant_url, timeout=30)

        # Create collection if it doesn't exist
        existing = [c.name for c in client.get_collections().collections]
        if collection_name not in existing:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
            logger.info(f"[ingest] created Qdrant collection '{collection_name}'")

        # Embed
        model = SentenceTransformer("BAAI/bge-m3")
        texts = [c.text for c in chunks]
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=len(chunks) > 50,
        ).tolist()

        # Build Qdrant points
        points = []
        for chunk, vector in zip(chunks, vectors):
            payload = chunk.to_qdrant_payload()
            payload["text"] = chunk.text          # store text in payload for retrieval
            points.append(PointStruct(
                id=chunk.chunk_id,
                vector=vector,
                payload=payload,
            ))

        client.upsert(collection_name=collection_name, points=points)
        logger.success(f"[ingest] upserted {len(points)} points to '{collection_name}'")

    except ImportError as exc:
        logger.warning(f"[ingest] qdrant/sentence-transformers not available: {exc}")
    except Exception as exc:
        logger.warning(f"[ingest] Qdrant upsert failed (non-fatal): {exc}")


# ---------------------------------------------------------------------------
# Shared helpers: routing + initial state
# ---------------------------------------------------------------------------

def _route_query(req: QueryRequest) -> RouteDecision:
    """Run the sensitivity classifier and return a RouteDecision."""
    doc_meta = {
        "doc_count":   req.doc_count,
        "pii_flagged": req.pii_flagged,
        "pii_types":   req.pii_types,
    }
    return _classifier.classify(req.query, doc_meta)


def _decision_to_badge(decision: RouteDecision) -> RouteBadge:
    badge_data = decision.to_ui_badge()
    return RouteBadge(
        label=badge_data["label"],
        color=badge_data["color"],
        cloud_allowed=badge_data["cloud_allowed"],
        tooltip=badge_data["tooltip"],
    )


def _build_agent_state(req: QueryRequest, decision: RouteDecision) -> dict:
    """Translate QueryRequest + RouteDecision into the initial AgentState dict."""
    max_loops = 1 if decision.route == "SIMPLE" else req.max_loops
    return {
        "query":            req.query,
        "cloud_allowed":    decision.cloud_allowed,
        "route_decision":   decision.to_dict(),
        "loop_count":       0,
        "cost_usd":         0.0,
        "retrieved_chunks": [],
        "answer_draft":     "",
        "critique":         "",
        "needs_more_retrieval": False,
        "final_answer":     "",
        "citations":        [],
        "confidence":       0.0,
        "used_backend":     "",
        "retrieval_query":  "",
        "error":            "",
        "_max_loops":       max_loops,   # consumed by run_agent()
    }


def _state_to_query_response(
    state: dict,
    req: QueryRequest,
    decision: RouteDecision,
    duration: float,
) -> QueryResponse:
    """Convert final AgentState into a QueryResponse for the client."""
    citations = [
        CitationCard(
            chunk_id=c["chunk_id"],
            source_file=c["source_file"],
            page_start=c["page_start"],
            page_end=c["page_end"],
            section_heading=c["section_heading"],
            excerpt=c["excerpt"],
        )
        for c in state.get("citations", [])
    ]
    used_backend = state.get("used_backend", "none") or "none"
    return QueryResponse(
        query=req.query,
        final_answer=state.get("final_answer", ""),
        citations=citations,
        route_badge=_decision_to_badge(decision),
        confidence=float(state.get("confidence", 0.0)),
        used_backend=used_backend,
        loop_count=int(state.get("loop_count", 0)),
        cost_usd=float(state.get("cost_usd", 0.0)),
        duration_seconds=round(duration, 3),
        error=state.get("error", ""),
    )


def _record_metrics(resp: QueryResponse) -> None:
    metrics.record(
        route=resp.route_badge.label,
        backend=resp.used_backend,
        cost_usd=resp.cost_usd,
        duration_seconds=resp.duration_seconds,
        loop_count=resp.loop_count,
        num_citations=len(resp.citations),
        query_len=len(resp.query),
    )


# ---------------------------------------------------------------------------
# POST /query  (blocking)
# ---------------------------------------------------------------------------

@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Submit a query and receive the complete answer (blocking)",
)
async def query_blocking(req: QueryRequest) -> QueryResponse:
    """
    Classify the query, run the LangGraph agent, return the full result.
    Typical latency: 2–8 s on local LLM, 1–3 s on Azure GPT-4o.
    Use /query/stream for a progressive UX during longer queries.
    """
    t0 = time.perf_counter()
    decision = _route_query(req)
    logger.info(
        f"[query] route={decision.route} "
        f"cloud={decision.cloud_allowed} "
        f"query='{req.query[:60]}'"
    )

    try:
        from agent.graph import run_agent
        max_loops = 1 if decision.route == "SIMPLE" else req.max_loops
        state = run_agent(
            query=req.query,
            cloud_allowed=decision.cloud_allowed,
            route_decision=decision.to_dict(),
            max_loops=max_loops,
        )
    except Exception as exc:
        logger.error(f"[query] agent error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {exc}",
        )

    duration = time.perf_counter() - t0
    resp = _state_to_query_response(state, req, decision, duration)
    _record_metrics(resp)
    return resp


# ---------------------------------------------------------------------------
# POST /query/stream  (SSE)
# ---------------------------------------------------------------------------

@app.post(
    "/query/stream",
    summary="Submit a query and receive a Server-Sent Events stream",
    response_description="SSE stream of typed JSON events",
)
async def query_stream(req: QueryRequest) -> EventSourceResponse:
    """
    Same as /query but delivers results progressively over SSE.

    Event sequence:
      route       → immediately after classification
      node_start  → before each LangGraph node executes
      chunks      → after retrieve node (which sources were found)
      critic      → after critic node (GROUNDED / INSUFFICIENT verdict)
      done        → final event containing the complete QueryResponse

    If an unrecoverable error occurs, an `error` event is emitted and
    the stream closes.  The React client should always listen for `error`.

    React usage example:
        const es = new EventSource('/query/stream', {...})   // via fetch + ReadableStream
        es.addEventListener('chunks', e => setChunks(JSON.parse(e.data).sources))
        es.addEventListener('done',   e => setAnswer(JSON.parse(e.data).result))
    """
    decision = _route_query(req)
    return EventSourceResponse(
        _stream_agent(req, decision),
        media_type="text/event-stream",
    )


async def _stream_agent(
    req: QueryRequest,
    decision: RouteDecision,
) -> AsyncGenerator[dict, None]:
    """
    Async generator that drives the LangGraph stream and yields SSE events.

    LangGraph's .stream() is synchronous; we run it in a thread via
    asyncio.to_thread so the event loop stays responsive.
    """
    import asyncio

    t0 = time.perf_counter()

    # ── Event 1: route badge ───────────────────────────────────────────
    yield _sse(SSERouteEvent(route_badge=_decision_to_badge(decision)))

    try:
        from agent.graph import agent_graph
        import os
        max_loops = 1 if decision.route == "SIMPLE" else req.max_loops
        os.environ["AGENT_MAX_LOOPS"] = str(max_loops)

        initial_state = {
            "query":            req.query,
            "cloud_allowed":    decision.cloud_allowed,
            "route_decision":   decision.to_dict(),
            "loop_count":       0,
            "cost_usd":         0.0,
            "retrieved_chunks": [],
            "answer_draft":     "",
            "critique":         "",
            "needs_more_retrieval": False,
            "final_answer":     "",
            "citations":        [],
            "confidence":       0.0,
            "used_backend":     "",
            "retrieval_query":  "",
            "error":            "",
        }

        final_state: dict = {}

        # Run LangGraph stream in a thread (it's synchronous)
        def _run_stream():
            nonlocal final_state
            loop_tracker: dict[str, int] = {}
            for chunk in agent_graph.stream(initial_state):
                # chunk is {"node_name": {partial_state_fields}}
                for node_name, node_state in chunk.items():
                    loop_tracker[node_name] = loop_tracker.get(node_name, 0) + 1
                    loop_num = node_state.get("loop_count", 0)
                    yield (node_name, node_state, loop_num)
                    final_state.update(node_state)

        # Consume the synchronous generator in the async context
        for node_name, node_state, loop_num in await asyncio.to_thread(
            lambda: list(_run_stream())
        ):
            event = _node_to_sse_event(node_name, node_state, loop_num)
            if event:
                yield _sse(event)

        # ── Final done event ──────────────────────────────────────────
        duration = time.perf_counter() - t0
        resp = _state_to_query_response(final_state, req, decision, duration)
        _record_metrics(resp)
        yield _sse(SSEDoneEvent(result=resp))

    except Exception as exc:
        logger.error(f"[stream] unhandled error: {exc}")
        yield _sse(SSEErrorEvent(
            message="Agent execution failed",
            detail=str(exc),
        ))


def _node_to_sse_event(node_name: str, node_state: dict, loop_num: int):
    """
    Translate a LangGraph node output into a typed SSE event.
    Returns None for nodes that don't need a dedicated event.
    """
    if node_name == "retrieve":
        chunks = node_state.get("retrieved_chunks", [])
        sources = list({c["source_file"] for c in chunks})
        return SSEChunksEvent(
            loop=loop_num,
            num_chunks=len(chunks),
            sources=sources,
        )

    if node_name == "critic":
        critique: str = node_state.get("critique", "")
        needs_more: bool = node_state.get("needs_more_retrieval", False)
        verdict = "INSUFFICIENT" if needs_more else "GROUNDED"
        reason = critique.split(":", 1)[1].strip() if ":" in critique else ""
        return SSECriticEvent(
            loop=loop_num,
            verdict=verdict,
            reason=reason,
        )

    # node_start events for reason and report
    if node_name in ("reason", "report"):
        return SSENodeStartEvent(node=node_name, loop=loop_num)

    return None


def _sse(event) -> dict:
    """
    Serialise a Pydantic SSE event model into the dict format that
    sse_starlette expects: {"data": "<json>", "event": "<type>"}.
    """
    payload = event.model_dump()
    event_type = payload.get("type", "message")
    return {
        "event": event_type,
        "data": json.dumps(payload),
    }


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Check reachability of all backends",
)
async def health() -> HealthResponse:
    """
    Pings Qdrant, the local LLM server, and Azure OpenAI.
    Returns 200 regardless of backend status — the `status` field
    in the response body reflects overall health.
    """
    import asyncio

    async def _check_qdrant() -> BackendStatus:
        try:
            import httpx
            t0 = time.perf_counter()
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{qdrant_url}/healthz")
            ok = r.status_code == 200
            return BackendStatus(
                name="qdrant",
                reachable=ok,
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            )
        except Exception:
            return BackendStatus(name="qdrant", reachable=False)

    async def _check_local_llm() -> BackendStatus:
        try:
            import httpx
            t0 = time.perf_counter()
            local_url = os.getenv("LOCAL_LLM_URL", "http://localhost:8080")
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{local_url}/health")
            ok = r.status_code == 200
            return BackendStatus(
                name="local_llm",
                reachable=ok,
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            )
        except Exception:
            return BackendStatus(name="local_llm", reachable=False)

    async def _check_azure() -> BackendStatus:
        endpoint = os.getenv("AZURE_OAI_ENDPOINT", "")
        if not endpoint:
            return BackendStatus(name="azure_oai", reachable=False, latency_ms=-1.0)
        try:
            import httpx
            t0 = time.perf_counter()
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(
                    endpoint.rstrip("/"),
                    headers={"api-key": os.getenv("AZURE_OAI_KEY", "")},
                )
            # Azure returns 401 for wrong key but 404 for unreachable — either
            # means the endpoint is live.
            ok = r.status_code in (200, 401, 404)
            return BackendStatus(
                name="azure_oai",
                reachable=ok,
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            )
        except Exception:
            return BackendStatus(name="azure_oai", reachable=False)

    qdrant, local, azure = await asyncio.gather(
        _check_qdrant(), _check_local_llm(), _check_azure()
    )

    # Overall status
    if qdrant.reachable and local.reachable:
        overall = "healthy"
    elif qdrant.reachable or local.reachable:
        overall = "degraded"
    else:
        overall = "unhealthy"

    return HealthResponse(
        status=overall,
        qdrant=qdrant.reachable,
        local_llm=local.reachable,
        azure_oai=azure.reachable,
        backends=[qdrant, local, azure],
    )


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------

@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="In-memory query statistics (resets on restart)",
)
async def get_metrics() -> MetricsResponse:
    return MetricsResponse(**metrics.snapshot())


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unhandled exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )