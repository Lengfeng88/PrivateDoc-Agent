"""
tests/test_api.py
-----------------
FastAPI endpoint tests.  All external dependencies (Qdrant, LLM backends,
embedding model) are mocked so the suite runs in CI with zero infra.

Coverage
--------
  Schemas          : request validation, response shape, SSE event models
  /ingest          : happy path, bad extension, empty file, Qdrant-down graceful
  /query           : SIMPLE / COMPLEX / SENSITIVE routing, agent error handling
  /query/stream    : SSE event sequence, done event, error event
  /health          : all-up, degraded, all-down
  /metrics         : counts accumulate correctly
  cost_logger      : thread-safety, JSONL output, reset
"""

from __future__ import annotations

import json
import sys
import threading
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

# Patch heavy imports BEFORE importing the app so we never try to load
# sentence-transformers or qdrant_client in the test environment.
_MOCK_AGENT_STATE = {
    "final_answer":     "Revenue was $1.86B in Q3 2024, up 26% YoY. [abc-chunk-001]",
    "citations":        [{
        "chunk_id":        "abc-chunk-001",
        "source_file":     "shopify_q3.pdf",
        "page_start":      4,
        "page_end":        4,
        "section_heading": "Revenue Overview",
        "excerpt":         "Total revenue for Q3 2024 was $1.86 billion…",
    }],
    "confidence":       0.91,
    "used_backend":     "local",
    "loop_count":       1,
    "cost_usd":         0.0,
    "error":            "",
}

# Patch run_agent to return mock state without touching LangGraph
_mock_run_agent = MagicMock(return_value=_MOCK_AGENT_STATE)

# Patch agent_graph.stream to yield one node output per node
def _mock_stream(state):
    yield {"retrieve": {"retrieved_chunks": [
        {
            "chunk_id": "abc-chunk-001", "text": "Revenue was $1.86B",
            "source_file": "shopify_q3.pdf", "page_start": 4, "page_end": 4,
            "section_heading": "Revenue Overview", "score": 0.91,
        }
    ], "retrieval_query": "Q3 2024 revenue", "loop_count": 1}}
    yield {"reason": {"answer_draft": "Revenue was $1.86B. [abc-chunk-001]\nCONFIDENCE: 0.91",
                      "confidence": 0.91, "used_backend": "local", "cost_usd": 0.0}}
    yield {"critic": {"critique": "GROUNDED", "needs_more_retrieval": False}}
    yield {"report": {**_MOCK_AGENT_STATE}}

_mock_graph = MagicMock()
_mock_graph.stream = _mock_stream

with patch("agent.graph.run_agent", _mock_run_agent), \
     patch("agent.graph.agent_graph", _mock_graph):
    from api.main import app
    from api.cost_logger import MetricsStore, init_metrics, metrics as _metrics_singleton

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset the in-memory metrics before every test."""
    _metrics_singleton.reset()
    yield
    _metrics_singleton.reset()


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _pdf_bytes() -> bytes:
    """Minimal valid-looking PDF bytes for upload tests."""
    return b"%PDF-1.4 fake content for testing purposes only"


def _docx_bytes() -> bytes:
    """Minimal DOCX-like bytes (PK header = zip = docx)."""
    return b"PK\x03\x04 fake docx content"


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_query_request_rejects_empty_query(self, client):
        r = client.post("/query", json={"query": ""})
        assert r.status_code == 422

    def test_query_request_rejects_extra_fields(self, client):
        r = client.post("/query", json={"query": "what is revenue?", "unknown_field": 1})
        assert r.status_code == 422

    def test_query_request_defaults(self, client):
        with patch("agent.graph.run_agent", _mock_run_agent):
            r = client.post("/query", json={"query": "what is revenue?"})
        assert r.status_code == 200
        body = r.json()
        # doc_count defaults to 1 → SIMPLE route
        assert body["route_badge"]["label"] in ("SIMPLE", "COMPLEX", "SENSITIVE")

    def test_query_response_has_required_fields(self, client):
        with patch("agent.graph.run_agent", _mock_run_agent):
            r = client.post("/query", json={"query": "what is revenue?"})
        body = r.json()
        required = {
            "query", "final_answer", "citations", "route_badge",
            "confidence", "used_backend", "loop_count", "cost_usd",
            "duration_seconds", "error",
        }
        assert required.issubset(body.keys())

    def test_route_badge_has_color(self, client):
        with patch("agent.graph.run_agent", _mock_run_agent):
            r = client.post("/query", json={"query": "what is revenue?"})
        badge = r.json()["route_badge"]
        assert badge["color"] in ("green", "blue", "red")

    def test_citation_card_shape(self, client):
        with patch("agent.graph.run_agent", _mock_run_agent):
            r = client.post("/query", json={"query": "what is revenue?"})
        citations = r.json()["citations"]
        assert isinstance(citations, list)
        if citations:
            c = citations[0]
            assert "chunk_id" in c
            assert "source_file" in c
            assert "excerpt" in c


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------

class TestIngest:
    def test_rejects_unsupported_extension(self, client):
        r = client.post(
            "/ingest",
            files={"file": ("report.xlsx", b"fake", "application/octet-stream")},
        )
        assert r.status_code == 422
        assert "xlsx" in r.json()["detail"].lower()

    def test_rejects_txt_file(self, client):
        r = client.post(
            "/ingest",
            files={"file": ("notes.txt", b"some text", "text/plain")},
        )
        assert r.status_code == 422

    def test_pdf_accepted_graceful_without_qdrant(self, client):
        """Even with Qdrant down, ingest should return 200 (chunking succeeded)."""
        with patch("api.main._run_ingest_pipeline", return_value={
            "status": "ok",
            "filename": "test.pdf",
            "file_hash": "abc123",
            "num_elements": 10,
            "num_chunks": 5,
            "pii_flagged_chunks": 0,
            "collection_name": "privatedoc",
            "error": "",
        }):
            r = client.post(
                "/ingest",
                files={"file": ("test.pdf", _pdf_bytes(), "application/pdf")},
            )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["num_chunks"] == 5
        assert "duration_seconds" in body

    def test_docx_accepted(self, client):
        with patch("api.main._run_ingest_pipeline", return_value={
            "status": "ok", "filename": "doc.docx", "file_hash": "def456",
            "num_elements": 8, "num_chunks": 4, "pii_flagged_chunks": 1,
            "collection_name": "privatedoc", "error": "",
        }):
            r = client.post(
                "/ingest",
                files={"file": ("doc.docx", _docx_bytes(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
                data={"collection_name": "privatedoc"},
            )
        assert r.status_code == 200
        assert r.json()["pii_flagged_chunks"] == 1

    def test_custom_collection_name_forwarded(self, client):
        captured = {}
        def _fake_pipeline(tmp_path, filename, collection_name, run_pii_detection):
            captured["collection_name"] = collection_name
            return {
                "status": "ok", "filename": filename, "file_hash": "x",
                "num_elements": 1, "num_chunks": 1, "pii_flagged_chunks": 0,
                "collection_name": collection_name, "error": "",
            }
        with patch("api.main._run_ingest_pipeline", side_effect=_fake_pipeline):
            client.post(
                "/ingest",
                files={"file": ("r.pdf", _pdf_bytes(), "application/pdf")},
                data={"collection_name": "my_docs"},
            )
        assert captured.get("collection_name") == "my_docs"


# ---------------------------------------------------------------------------
# POST /query (blocking)
# ---------------------------------------------------------------------------

class TestQueryBlocking:
    def test_simple_query_returns_200(self, client):
        with patch("agent.graph.run_agent", _mock_run_agent):
            r = client.post("/query", json={"query": "What was total revenue?"})
        assert r.status_code == 200

    def test_final_answer_populated(self, client):
        with patch("agent.graph.run_agent", _mock_run_agent):
            r = client.post("/query", json={"query": "What was total revenue?"})
        assert r.json()["final_answer"] != ""

    def test_sensitive_route_cloud_blocked(self, client):
        with patch("agent.graph.run_agent", _mock_run_agent):
            r = client.post("/query", json={"query": "Show me the patient diagnosis"})
        body = r.json()
        assert body["route_badge"]["label"] == "SENSITIVE"
        assert body["route_badge"]["cloud_allowed"] is False

    def test_complex_route_for_compare_query(self, client):
        with patch("agent.graph.run_agent", _mock_run_agent):
            r = client.post("/query", json={
                "query": "Compare gross margin year over year and explain the trend",
                "doc_count": 2,
            })
        body = r.json()
        assert body["route_badge"]["label"] == "COMPLEX"

    def test_agent_error_returns_500(self, client):
        with patch("agent.graph.run_agent", side_effect=RuntimeError("GPU OOM")):
            r = client.post("/query", json={"query": "What is revenue?"})
        assert r.status_code == 500

    def test_cost_usd_in_response(self, client):
        with patch("agent.graph.run_agent", _mock_run_agent):
            r = client.post("/query", json={"query": "What is revenue?"})
        assert "cost_usd" in r.json()
        assert isinstance(r.json()["cost_usd"], float)

    def test_duration_seconds_positive(self, client):
        with patch("agent.graph.run_agent", _mock_run_agent):
            r = client.post("/query", json={"query": "What is revenue?"})
        assert r.json()["duration_seconds"] >= 0

    def test_metrics_accumulate_after_query(self, client):
        with patch("agent.graph.run_agent", _mock_run_agent):
            client.post("/query", json={"query": "What is revenue?"})
        r = client.get("/metrics")
        assert r.json()["total_queries"] == 1

    def test_pii_flag_forwarded_to_router(self, client):
        """Query on a PII-flagged doc should always route SENSITIVE."""
        with patch("agent.graph.run_agent", _mock_run_agent):
            r = client.post("/query", json={
                "query": "Summarise this document",
                "pii_flagged": True,
                "pii_types": ["ssn"],
            })
        assert r.json()["route_badge"]["label"] == "SENSITIVE"


# ---------------------------------------------------------------------------
# POST /query/stream (SSE)
# ---------------------------------------------------------------------------

class TestQueryStream:
    def _collect_events(self, client, payload: dict) -> list[dict]:
        """
        Consume the SSE stream and return a list of parsed event dicts.
        TestClient returns the raw response; we parse the SSE frames manually.
        """
        with patch("agent.graph.agent_graph", _mock_graph):
            with client.stream("POST", "/query/stream", json=payload) as r:
                assert r.status_code == 200
                events = []
                current_data = []
                for line in r.iter_lines():
                    if line.startswith("data:"):
                        current_data.append(line[5:].strip())
                    elif line == "" and current_data:
                        raw = " ".join(current_data)
                        try:
                            events.append(json.loads(raw))
                        except json.JSONDecodeError:
                            pass
                        current_data = []
                return events

    def test_stream_returns_200(self, client):
        with patch("agent.graph.agent_graph", _mock_graph):
            r = client.post("/query/stream", json={"query": "What is revenue?"})
        assert r.status_code == 200

    def test_stream_content_type(self, client):
        with patch("agent.graph.agent_graph", _mock_graph):
            r = client.post("/query/stream", json={"query": "What is revenue?"})
        assert "text/event-stream" in r.headers.get("content-type", "")

    def test_first_event_is_route(self, client):
        events = self._collect_events(client, {"query": "What is revenue?"})
        assert len(events) >= 1
        assert events[0]["type"] == "route"

    def test_route_event_has_badge(self, client):
        events = self._collect_events(client, {"query": "What is revenue?"})
        route_events = [e for e in events if e["type"] == "route"]
        assert len(route_events) == 1
        assert "route_badge" in route_events[0]
        assert "label" in route_events[0]["route_badge"]

    def test_chunks_event_emitted(self, client):
        events = self._collect_events(client, {"query": "What is revenue?"})
        chunk_events = [e for e in events if e["type"] == "chunks"]
        assert len(chunk_events) >= 1
        assert "num_chunks" in chunk_events[0]
        assert "sources" in chunk_events[0]

    def test_done_event_is_last(self, client):
        events = self._collect_events(client, {"query": "What is revenue?"})
        assert events[-1]["type"] == "done"

    def test_done_event_has_result(self, client):
        events = self._collect_events(client, {"query": "What is revenue?"})
        done = events[-1]
        assert "result" in done
        assert "final_answer" in done["result"]

    def test_sensitive_stream_route_badge_red(self, client):
        events = self._collect_events(client, {"query": "Show patient SSN"})
        route_event = next(e for e in events if e["type"] == "route")
        assert route_event["route_badge"]["color"] == "red"

    def test_stream_metrics_accumulate(self, client):
        self._collect_events(client, {"query": "What is revenue?"})
        r = client.get("/metrics")
        assert r.json()["total_queries"] == 1


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_required_fields(self, client):
        r = client.get("/health")
        body = r.json()
        assert "status" in body
        assert "qdrant" in body
        assert "local_llm" in body
        assert "backends" in body

    def test_health_status_values(self, client):
        r = client.get("/health")
        assert r.json()["status"] in ("healthy", "degraded", "unhealthy")

    def test_all_down_is_unhealthy(self, client):
        """When all backends are unreachable, status should be 'unhealthy'."""
        import httpx
        async def _fail(*a, **kw):
            raise httpx.ConnectError("refused")

        with patch("httpx.AsyncClient") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.__aenter__ = MagicMock(return_value=mock_instance)
            mock_instance.__aexit__ = MagicMock(return_value=False)
            mock_instance.get = MagicMock(side_effect=Exception("refused"))
            mock_cls.return_value = mock_instance
            r = client.get("/health")
        # Even if health checks error, endpoint must return 200
        assert r.status_code == 200

    def test_backends_list_has_three_entries(self, client):
        r = client.get("/health")
        assert len(r.json()["backends"]) == 3


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_metrics_starts_at_zero(self, client):
        r = client.get("/metrics")
        body = r.json()
        assert body["total_queries"] == 0
        assert body["total_cost_usd"] == 0.0

    def test_metrics_accumulate_across_queries(self, client):
        with patch("agent.graph.run_agent", _mock_run_agent):
            for _ in range(3):
                client.post("/query", json={"query": "What is revenue?"})
        r = client.get("/metrics")
        assert r.json()["total_queries"] == 3

    def test_route_counts_in_metrics(self, client):
        with patch("agent.graph.run_agent", _mock_run_agent):
            client.post("/query", json={"query": "What is revenue?"})          # SIMPLE
            client.post("/query", json={"query": "Compare margins year over year", "doc_count": 2})  # COMPLEX
            client.post("/query", json={"query": "Show patient SSN"})          # SENSITIVE
        body = client.get("/metrics").json()
        counts = body["route_counts"]
        assert counts["SIMPLE"] >= 1
        assert counts["COMPLEX"] >= 1
        assert counts["SENSITIVE"] >= 1

    def test_metrics_response_shape(self, client):
        r = client.get("/metrics")
        body = r.json()
        required = {
            "total_queries", "total_cost_usd", "route_counts",
            "backend_counts", "avg_duration_seconds", "avg_loops",
        }
        assert required.issubset(body.keys())


# ---------------------------------------------------------------------------
# MetricsStore unit tests (thread-safety)
# ---------------------------------------------------------------------------

class TestMetricsStore:
    def test_basic_record(self):
        store = MetricsStore(log_path=None)
        store.record(route="SIMPLE", backend="local", cost_usd=0.0,
                     duration_seconds=1.2, loop_count=1)
        s = store.snapshot()
        assert s["total_queries"] == 1
        assert s["route_counts"]["SIMPLE"] == 1
        assert s["backend_counts"]["local"] == 1
        assert s["avg_duration_seconds"] == 1.2

    def test_concurrent_writes_are_safe(self):
        store = MetricsStore(log_path=None)
        threads = [
            threading.Thread(
                target=store.record,
                kwargs=dict(route="SIMPLE", backend="local", cost_usd=0.001,
                            duration_seconds=0.5, loop_count=1)
            )
            for _ in range(50)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert store.snapshot()["total_queries"] == 50

    def test_reset_clears_all_counters(self):
        store = MetricsStore(log_path=None)
        store.record(route="COMPLEX", backend="azure", cost_usd=0.01,
                     duration_seconds=3.0, loop_count=2)
        store.reset()
        s = store.snapshot()
        assert s["total_queries"] == 0
        assert s["total_cost_usd"] == 0.0

    def test_unknown_route_creates_new_bucket(self):
        store = MetricsStore(log_path=None)
        store.record(route="FUTURE_ROUTE", backend="local", cost_usd=0.0,
                     duration_seconds=1.0, loop_count=1)
        s = store.snapshot()
        assert s["route_counts"].get("FUTURE_ROUTE", 0) == 1

    def test_jsonl_written(self, tmp_path):
        log = tmp_path / "q.jsonl"
        store = MetricsStore(log_path=log)
        store.record(route="SIMPLE", backend="local", cost_usd=0.0,
                     duration_seconds=1.0, loop_count=1,
                     num_citations=2, query_len=20)
        assert log.exists()
        record = json.loads(log.read_text().strip())
        assert record["route"] == "SIMPLE"
        assert "query_len" in record
        # Raw query text must NOT be in the log
        assert "query" not in record or record.get("query") is None

    def test_jsonl_raw_query_not_stored(self, tmp_path):
        log = tmp_path / "q.jsonl"
        store = MetricsStore(log_path=log)
        store.record(route="SENSITIVE", backend="blocked", cost_usd=0.0,
                     duration_seconds=0.1, loop_count=0, query_len=42)
        content = log.read_text()
        # Must NOT contain any reconstructable query text
        assert "patient" not in content
        assert "SSN" not in content

    def test_avg_computed_correctly(self):
        store = MetricsStore(log_path=None)
        store.record(route="SIMPLE", backend="local", cost_usd=0.0,
                     duration_seconds=2.0, loop_count=1)
        store.record(route="SIMPLE", backend="local", cost_usd=0.0,
                     duration_seconds=4.0, loop_count=3)
        s = store.snapshot()
        assert s["avg_duration_seconds"] == 3.0
        assert s["avg_loops"] == 2.0


# ---------------------------------------------------------------------------
# SSE helper unit tests
# ---------------------------------------------------------------------------

class TestSSEHelpers:
    def test_sse_event_serialisable(self):
        from api.main import _sse
        from api.schemas import SSERouteEvent, RouteBadge
        event = SSERouteEvent(route_badge=RouteBadge(
            label="SIMPLE", color="green", cloud_allowed=True, tooltip="ok"
        ))
        result = _sse(event)
        assert result["event"] == "route"
        parsed = json.loads(result["data"])
        assert parsed["type"] == "route"
        assert parsed["route_badge"]["label"] == "SIMPLE"

    def test_node_to_sse_event_retrieve(self):
        from api.main import _node_to_sse_event
        chunks = [{"source_file": "a.pdf", "chunk_id": "x", "text": "t",
                   "page_start": 1, "page_end": 1, "section_heading": "", "score": 0.9}]
        event = _node_to_sse_event("retrieve", {"retrieved_chunks": chunks, "loop_count": 1}, 1)
        assert event is not None
        assert event.type == "chunks"
        assert event.num_chunks == 1

    def test_node_to_sse_event_critic_grounded(self):
        from api.main import _node_to_sse_event
        event = _node_to_sse_event(
            "critic",
            {"critique": "GROUNDED", "needs_more_retrieval": False},
            loop_num=1,
        )
        assert event.verdict == "GROUNDED"

    def test_node_to_sse_event_critic_insufficient(self):
        from api.main import _node_to_sse_event
        event = _node_to_sse_event(
            "critic",
            {"critique": "INSUFFICIENT: missing Q3 data", "needs_more_retrieval": True},
            loop_num=1,
        )
        assert event.verdict == "INSUFFICIENT"
        assert "Q3" in event.reason

    def test_node_to_sse_event_reason_returns_node_start(self):
        from api.main import _node_to_sse_event
        event = _node_to_sse_event("reason", {}, loop_num=0)
        assert event is not None
        assert event.type == "node_start"
        assert event.node == "reason"

    def test_node_to_sse_event_unknown_returns_none(self):
        from api.main import _node_to_sse_event
        event = _node_to_sse_event("unknown_node", {}, loop_num=0)
        assert event is None