"""
api/schemas.py
--------------
Pydantic models that define the HTTP contract between the FastAPI layer
and its clients (the React frontend, the eval harness, curl).

Design notes
------------
- All models are strict: extra fields are rejected so the API surface
  doesn't silently absorb typos in client code.
- Request models mirror the agent's internal AgentState where appropriate,
  but are decoupled — the API layer translates between the two.
- Response models are designed for UI consumption: they include the route
  badge payload, per-query cost, and citation cards in a single response.
- SSE events use a discriminated union on the `type` field so the React
  client can switch on event type without parsing the full payload.

Endpoint summary
----------------
  POST /ingest          IngestRequest  → IngestResponse
  POST /query           QueryRequest   → QueryResponse         (blocking)
  POST /query/stream    QueryRequest   → SSE stream of SSEEvent
  GET  /health                         → HealthResponse
  GET  /metrics                        → MetricsResponse
"""

from __future__ import annotations

from typing import Literal, Any
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    """
    Sent as multipart/form-data from the frontend.
    The file bytes arrive via UploadFile; this model covers the metadata fields.
    """
    collection_name: str = Field(
        default="privatedoc",
        description="Qdrant collection to upsert into. Defaults to 'privatedoc'.",
        min_length=1,
        max_length=64,
    )
    run_pii_detection: bool = Field(
        default=True,
        description="Whether to run PII detection during ingestion.",
    )

    model_config = {"extra": "forbid"}


class IngestResponse(BaseModel):
    status: Literal["ok", "error"]
    filename: str
    file_hash: str
    num_elements: int       # raw elements from loader
    num_chunks: int         # chunks after semantic splitting
    pii_flagged_chunks: int
    collection_name: str
    duration_seconds: float
    error: str = ""

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Query (shared between blocking and streaming endpoints)
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """
    Body sent to POST /query and POST /query/stream.
    """
    query: str = Field(
        ...,
        description="The user's question in plain text.",
        min_length=1,
        max_length=2000,
    )
    collection_name: str = Field(
        default="privatedoc",
        description="Qdrant collection to search.",
    )
    doc_count: int = Field(
        default=1,
        description="Number of distinct documents in scope (used by the router).",
        ge=0,
    )
    # These are populated by the API layer from the ingested doc's metadata —
    # the frontend can pass them if it cached the ingest response.
    pii_flagged: bool = Field(
        default=False,
        description="Whether any chunk in scope was PII-flagged at ingest time.",
    )
    pii_types: list[str] = Field(
        default_factory=list,
        description="PII types found at ingest time (from IngestResponse).",
    )
    # Override the automatic max_loops (mainly used by the eval harness)
    max_loops: int = Field(default=3, ge=1, le=5)

    model_config = {"extra": "forbid"}


class CitationCard(BaseModel):
    """One source reference — rendered as a card in the React UI."""
    chunk_id: str
    source_file: str
    page_start: int
    page_end: int
    section_heading: str
    excerpt: str            # first ~120 chars of the chunk text

    model_config = {"extra": "forbid"}


class RouteBadge(BaseModel):
    """Minimal payload consumed by the React route badge component."""
    label: Literal["SIMPLE", "COMPLEX", "SENSITIVE"]
    color: Literal["green", "blue", "red"]
    cloud_allowed: bool
    tooltip: str            # human-readable reason string

    model_config = {"extra": "forbid"}


class QueryResponse(BaseModel):
    """
    Blocking response from POST /query.
    Contains the complete answer, citations, and routing metadata.
    """
    query: str
    final_answer: str
    citations: list[CitationCard]
    route_badge: RouteBadge
    confidence: float = Field(ge=0.0, le=1.0)
    used_backend: Literal["local", "azure", "none", "blocked"]
    loop_count: int
    cost_usd: float
    duration_seconds: float
    error: str = ""

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# SSE streaming events
# ---------------------------------------------------------------------------
# The React client switches on `type` and renders accordingly.
# Each event is serialised as JSON and sent as the `data:` field of an
# SSE frame.  The stream always ends with a `done` or `error` event.

class SSENodeStartEvent(BaseModel):
    type: Literal["node_start"] = "node_start"
    node: Literal["retrieve", "reason", "critic", "report"]
    loop: int = 0

    model_config = {"extra": "forbid"}


class SSERouteEvent(BaseModel):
    """Emitted immediately after routing, before the first node runs."""
    type: Literal["route"] = "route"
    route_badge: RouteBadge

    model_config = {"extra": "forbid"}


class SSEChunksEvent(BaseModel):
    """Emitted after retrieve node — shows which sources were found."""
    type: Literal["chunks"] = "chunks"
    loop: int
    num_chunks: int
    sources: list[str]      # source_file values for the UI source list

    model_config = {"extra": "forbid"}


class SSECriticEvent(BaseModel):
    """Emitted after critic node — shows verdict without looping detail."""
    type: Literal["critic"] = "critic"
    loop: int
    verdict: Literal["GROUNDED", "INSUFFICIENT"]
    reason: str = ""

    model_config = {"extra": "forbid"}


class SSETokenEvent(BaseModel):
    """
    Streamed token-by-token from the reason node when streaming is active.
    (In practice the local llama.cpp server supports token streaming;
    this event type allows the UI to show text appearing progressively.)
    """
    type: Literal["token"] = "token"
    text: str

    model_config = {"extra": "forbid"}


class SSEDoneEvent(BaseModel):
    """Final event — carries the complete QueryResponse payload."""
    type: Literal["done"] = "done"
    result: QueryResponse

    model_config = {"extra": "forbid"}


class SSEErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str
    detail: str = ""

    model_config = {"extra": "forbid"}


# Union type used in type hints (not serialised directly)
SSEEvent = (
    SSENodeStartEvent
    | SSERouteEvent
    | SSEChunksEvent
    | SSECriticEvent
    | SSETokenEvent
    | SSEDoneEvent
    | SSEErrorEvent
)


# ---------------------------------------------------------------------------
# Health + Metrics
# ---------------------------------------------------------------------------

class BackendStatus(BaseModel):
    name: str
    reachable: bool
    latency_ms: float = -1.0

    model_config = {"extra": "forbid"}


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    qdrant: bool
    local_llm: bool
    azure_oai: bool
    backends: list[BackendStatus]

    model_config = {"extra": "forbid"}


class MetricsResponse(BaseModel):
    """Lightweight in-memory metrics — resets on restart."""
    total_queries: int
    total_cost_usd: float
    route_counts: dict[str, int]    # {"SIMPLE": N, "COMPLEX": N, "SENSITIVE": N}
    backend_counts: dict[str, int]  # {"local": N, "azure": N, "none": N}
    avg_duration_seconds: float
    avg_loops: float

    model_config = {"extra": "forbid"}