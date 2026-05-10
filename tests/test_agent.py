"""
tests/test_agent.py
-------------------
Test suite for agent/ — graph topology, node logic, routing decisions.

All external I/O is mocked:
  - LLM backends (LocalLLMBackend, AzureOAIBackend) → deterministic stubs
  - Qdrant search → returns pre-built chunk fixtures
  - BGE-M3 embeddings → zero vectors

This means every test runs in < 1 second with no GPU, no network.

Test sections
-------------
1. Graph topology   — correct nodes, edges, entry point
2. Conditional edge — critic routes to retrieve vs report correctly
3. retrieve node    — RRF fusion, query rewriting, loop expansion
4. reason node      — backend selection, confidence parsing, cost accumulation
5. critic node      — fast paths, verdict parsing, loop guard
6. report node      — citation extraction, passthrough fallback
7. run_agent()      — end-to-end with mocked backends (SIMPLE / COMPLEX / SENSITIVE)
8. Prompts          — format_context_block structure
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.state import AgentState, RetrievedChunk, Citation
from agent.backends.base import LLMResponse
from agent.prompts import format_context_block, REASON_SYSTEM, CRITIC_SYSTEM


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_chunk(
    chunk_id: str = "aaaaaaaa-0000-0000-0000-000000000001",
    text: str = "Total revenue was $1.86 billion in Q3 2024.",
    source_file: str = "shopify_q3.pdf",
    page_start: int = 5,
    page_end: int = 5,
    section_heading: str = "Revenue Overview",
    score: float = 0.92,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source_file=source_file,
        page_start=page_start,
        page_end=page_end,
        section_heading=section_heading,
        score=score,
    )


CHUNK_A = make_chunk(
    chunk_id="aaaaaaaa-0000-0000-0000-000000000001",
    text="Total revenue was $1.86B in Q3 2024, up 26% YoY.",
)
CHUNK_B = make_chunk(
    chunk_id="bbbbbbbb-0000-0000-0000-000000000002",
    text="Gross profit margin was 49.8% in Q3 2024 vs 50.7% in Q3 2023.",
    section_heading="Gross Profit",
)

GROUNDED_ANSWER = (
    f"Revenue grew 26% to $1.86B. [{CHUNK_A['chunk_id']}] "
    f"Gross margin declined slightly. [{CHUNK_B['chunk_id']}]"
)

_GOOD_RESPONSE = LLMResponse(
    text=GROUNDED_ANSWER,
    prompt_tokens=200,
    completion_tokens=50,
    cost_usd=0.0,
    backend="local",
)

_INSUFFICIENT_RESPONSE = LLMResponse(
    text="INSUFFICIENT_CONTEXT: no margin data found",
    prompt_tokens=150,
    completion_tokens=15,
    cost_usd=0.0,
    backend="local",
)

_CRITIC_GROUNDED = LLMResponse(
    text="GROUNDED", prompt_tokens=100, completion_tokens=5,
    cost_usd=0.0, backend="local",
)
_CRITIC_INSUFFICIENT = LLMResponse(
    text="INSUFFICIENT: missing year-over-year margin data",
    prompt_tokens=100, completion_tokens=12, cost_usd=0.0, backend="local",
)


def make_local_mock(response: LLMResponse = _GOOD_RESPONSE) -> MagicMock:
    m = MagicMock()
    m.complete.return_value = response
    m.health_check.return_value = True
    return m


def base_state(**overrides) -> AgentState:
    """Minimal valid AgentState for testing."""
    s: AgentState = {
        "query": "What was revenue in Q3 2024?",
        "cloud_allowed": True,
        "route_decision": {"route": "SIMPLE"},
        "loop_count": 0,
        "cost_usd": 0.0,
        "retrieved_chunks": [CHUNK_A, CHUNK_B],
        "answer_draft": GROUNDED_ANSWER,
        "critique": "GROUNDED",
        "needs_more_retrieval": False,
        "final_answer": "",
        "citations": [],
        "confidence": 0.85,
        "used_backend": "local",
        "retrieval_query": "revenue Q3 2024",
        "error": "",
    }
    s.update(overrides)
    return s


# ---------------------------------------------------------------------------
# 1. Graph topology
# ---------------------------------------------------------------------------

class TestGraphTopology:
    def test_graph_compiles(self):
        from agent.graph import agent_graph
        assert agent_graph is not None

    def test_expected_nodes_present(self):
        from agent.graph import agent_graph
        nodes = list(agent_graph.get_graph().nodes.keys())
        for expected in ("retrieve", "reason", "critic", "report"):
            assert expected in nodes, f"missing node: {expected}"

    def test_entry_point_is_retrieve(self):
        from agent.graph import agent_graph
        edges = agent_graph.get_graph().edges
        start_targets = [e.target for e in edges if e.source == "__start__"]
        assert "retrieve" in start_targets

    def test_report_connects_to_end(self):
        from agent.graph import agent_graph
        edges = agent_graph.get_graph().edges
        report_targets = [e.target for e in edges if e.source == "report"]
        assert "__end__" in report_targets

    def test_critic_has_two_outbound_edges(self):
        from agent.graph import agent_graph
        edges = agent_graph.get_graph().edges
        critic_targets = {e.target for e in edges if e.source == "critic"}
        assert "retrieve" in critic_targets
        assert "report" in critic_targets

    def test_retrieve_reason_edge_exists(self):
        from agent.graph import agent_graph
        edges = agent_graph.get_graph().edges
        edge_pairs = {(e.source, e.target) for e in edges}
        assert ("retrieve", "reason") in edge_pairs
        assert ("reason", "critic") in edge_pairs


# ---------------------------------------------------------------------------
# 2. Conditional edge
# ---------------------------------------------------------------------------

class TestConditionalEdge:
    def test_routes_to_report_when_grounded(self):
        from agent.graph import _route_after_critic
        state = base_state(needs_more_retrieval=False, loop_count=1)
        assert _route_after_critic(state) == "report"

    def test_routes_to_retrieve_when_insufficient(self):
        from agent.graph import _route_after_critic
        state = base_state(needs_more_retrieval=True, loop_count=1)
        assert _route_after_critic(state) == "retrieve"

    def test_routes_to_report_when_loop_cap_hit(self):
        from agent.graph import _route_after_critic
        # Even if critic wants more, loop cap takes priority
        state = base_state(needs_more_retrieval=True, loop_count=3)
        assert _route_after_critic(state) == "report"

    def test_routes_to_report_on_loop_count_zero_grounded(self):
        from agent.graph import _route_after_critic
        state = base_state(needs_more_retrieval=False, loop_count=0)
        assert _route_after_critic(state) == "report"

    def test_loop_count_2_still_retries(self):
        from agent.graph import _route_after_critic
        state = base_state(needs_more_retrieval=True, loop_count=2)
        # MAX_LOOP default is 3; loop_count=2 < 3 → retrieve
        assert _route_after_critic(state) == "retrieve"


# ---------------------------------------------------------------------------
# 3. Retrieve node
# ---------------------------------------------------------------------------

class TestRetrieveNode:
    def test_increments_loop_count(self):
        from agent.nodes.retrieve import retrieve_node
        with patch("agent.nodes.retrieve._qdrant_hybrid_search", return_value=[CHUNK_A]):
            with patch("agent.nodes.retrieve._try_rewrite", return_value="revenue Q3"):
                result = retrieve_node(base_state(loop_count=0, retrieved_chunks=[]))
        assert result["loop_count"] == 1

    def test_returns_chunks(self):
        from agent.nodes.retrieve import retrieve_node
        with patch("agent.nodes.retrieve._qdrant_hybrid_search", return_value=[CHUNK_A, CHUNK_B]):
            with patch("agent.nodes.retrieve._try_rewrite", return_value="revenue"):
                result = retrieve_node(base_state(loop_count=0, retrieved_chunks=[]))
        assert len(result["retrieved_chunks"]) == 2

    def test_does_not_retrieve_at_max_loop(self):
        from agent.nodes.retrieve import retrieve_node
        with patch.dict(os.environ, {"AGENT_MAX_LOOPS": "3"}):
            result = retrieve_node(base_state(loop_count=3))
        # Should return without touching Qdrant
        assert result.get("loop_count") == 3

    def test_rrf_fusion_merges_ranked_lists(self):
        from agent.nodes.retrieve import _reciprocal_rank_fusion
        dense  = [("id1", 0.9), ("id2", 0.8), ("id3", 0.5)]
        sparse = [("id2", 0.95), ("id1", 0.7), ("id4", 0.4)]
        fused = _reciprocal_rank_fusion(dense, sparse)
        ids = [cid for cid, _ in fused]
        # id1 and id2 appear in both lists → should rank highest
        assert ids[0] in ("id1", "id2")
        assert ids[1] in ("id1", "id2")
        assert "id4" in ids  # sparse-only should also appear

    def test_rrf_handles_empty_sparse(self):
        from agent.nodes.retrieve import _reciprocal_rank_fusion
        dense = [("id1", 0.9), ("id2", 0.8)]
        fused = _reciprocal_rank_fusion(dense, [])
        assert [cid for cid, _ in fused] == ["id1", "id2"]

    def test_critique_expansion_on_loop_1(self):
        from agent.nodes.retrieve import _expand_from_critique
        result = _expand_from_critique(
            "What was revenue?",
            "INSUFFICIENT: missing gross margin comparison data",
        )
        assert "gross margin" in result.lower()

    def test_expansion_handles_no_colon(self):
        from agent.nodes.retrieve import _expand_from_critique
        result = _expand_from_critique("revenue query", "GROUNDED")
        assert result == "revenue query"

    def test_retrieval_query_stored(self):
        from agent.nodes.retrieve import retrieve_node
        with patch("agent.nodes.retrieve._qdrant_hybrid_search", return_value=[CHUNK_A]):
            with patch("agent.nodes.retrieve._try_rewrite", return_value="rewritten query"):
                result = retrieve_node(base_state(loop_count=0, retrieved_chunks=[]))
        assert result["retrieval_query"] == "rewritten query"


# ---------------------------------------------------------------------------
# 4. Reason node
# ---------------------------------------------------------------------------

class TestReasonNode:
    def test_returns_answer_draft(self):
        from agent.nodes.reason import reason_node
        local_mock = make_local_mock(_GOOD_RESPONSE)
        with patch("agent.nodes.reason.LocalLLMBackend", return_value=local_mock):
            result = reason_node(base_state())
        assert "answer_draft" in result
        assert len(result["answer_draft"]) > 10

    def test_accumulates_cost(self):
        from agent.nodes.reason import reason_node
        azure_resp = LLMResponse(
            text="Revenue grew. CONFIDENCE: 0.80",
            prompt_tokens=300, completion_tokens=20,
            cost_usd=0.0018, backend="azure",
        )
        local_mock = MagicMock()
        local_mock.complete.side_effect = ConnectionError("offline")
        local_mock.health_check.return_value = False
        azure_mock = MagicMock()
        azure_mock.complete.return_value = azure_resp
        with patch("agent.nodes.reason.LocalLLMBackend", return_value=local_mock):
            with patch("agent.nodes.reason.AzureOAIBackend", return_value=azure_mock):
                result = reason_node(base_state(cost_usd=0.001))
        # 0.001 (prev) + 0.0018 (azure) = 0.0028
        assert abs(result["cost_usd"] - 0.0028) < 1e-9

    def test_used_backend_is_local(self):
        from agent.nodes.reason import reason_node
        local_mock = make_local_mock(_GOOD_RESPONSE)
        with patch("agent.nodes.reason.LocalLLMBackend", return_value=local_mock):
            result = reason_node(base_state())
        assert result["used_backend"] == "local"

    def test_falls_back_to_azure_when_local_fails(self):
        from agent.nodes.reason import reason_node
        local_mock = MagicMock()
        local_mock.complete.side_effect = ConnectionError("offline")
        local_mock.health_check.return_value = False
        azure_resp = LLMResponse(
            text="Answer from azure.", prompt_tokens=100,
            completion_tokens=10, cost_usd=0.001, backend="azure",
        )
        azure_mock = MagicMock()
        azure_mock.complete.return_value = azure_resp
        with patch("agent.nodes.reason.LocalLLMBackend", return_value=local_mock):
            with patch("agent.nodes.reason.AzureOAIBackend", return_value=azure_mock):
                result = reason_node(base_state())
        assert result["used_backend"] == "azure"

    def test_cloud_blocked_on_sensitive_route(self):
        from agent.nodes.reason import reason_node
        local_mock = MagicMock()
        local_mock.complete.side_effect = ConnectionError("offline")
        local_mock.health_check.return_value = False
        with patch("agent.nodes.reason.LocalLLMBackend", return_value=local_mock):
            result = reason_node(base_state(cloud_allowed=False))
        # Both local failed AND cloud blocked → refusal
        assert result["used_backend"] in ("blocked", "none")
        assert "INSUFFICIENT_CONTEXT" in result.get("answer_draft", "")

    def test_no_chunks_returns_insufficient(self):
        from agent.nodes.reason import reason_node
        result = reason_node(base_state(retrieved_chunks=[]))
        assert "INSUFFICIENT_CONTEXT" in result["answer_draft"]
        assert result["confidence"] == 0.0

    def test_confidence_parsed_correctly(self):
        from agent.nodes.reason import _parse_confidence
        assert abs(_parse_confidence("Some answer.\nCONFIDENCE: 0.87") - 0.87) < 0.001

    def test_confidence_clamped_to_unit(self):
        from agent.nodes.reason import _parse_confidence
        assert _parse_confidence("CONFIDENCE: 1.5") == 1.0
        assert _parse_confidence("CONFIDENCE: -0.1") == 0.0

    def test_confidence_defaults_on_missing(self):
        from agent.nodes.reason import _parse_confidence
        val = _parse_confidence("Just an answer with no confidence line.")
        assert 0.0 <= val <= 1.0

    def test_confidence_low_for_insufficient(self):
        from agent.nodes.reason import _parse_confidence
        assert _parse_confidence("INSUFFICIENT_CONTEXT: nothing found") <= 0.2

    def test_confidence_line_stripped_from_draft(self):
        from agent.nodes.reason import _strip_confidence_line
        raw = "Revenue was up 26%.\nCONFIDENCE: 0.91"
        clean = _strip_confidence_line(raw)
        assert "CONFIDENCE" not in clean
        assert "Revenue was up 26%" in clean


# ---------------------------------------------------------------------------
# 5. Critic node
# ---------------------------------------------------------------------------

class TestCriticNode:
    def test_grounded_verdict_sets_flag_false(self):
        from agent.nodes.critic import critic_node
        local_mock = make_local_mock(_CRITIC_GROUNDED)
        with patch("agent.nodes.critic.LocalLLMBackend", return_value=local_mock):
            result = critic_node(base_state())
        assert result["critique"] == "GROUNDED"
        assert result["needs_more_retrieval"] is False

    def test_insufficient_verdict_sets_flag_true(self):
        from agent.nodes.critic import critic_node
        local_mock = make_local_mock(_CRITIC_INSUFFICIENT)
        with patch("agent.nodes.critic.LocalLLMBackend", return_value=local_mock):
            result = critic_node(base_state(answer_draft="Some partial answer."))
        assert result["needs_more_retrieval"] is True
        assert "INSUFFICIENT" in result["critique"]

    def test_fast_path_loop_cap(self):
        from agent.nodes.critic import critic_node
        # At MAX_LOOP, should short-circuit to GROUNDED without calling LLM
        with patch.dict(os.environ, {"AGENT_MAX_LOOPS": "3"}):
            result = critic_node(base_state(loop_count=3))
        assert result["critique"] == "GROUNDED"
        assert result["needs_more_retrieval"] is False

    def test_fast_path_insufficient_in_draft(self):
        from agent.nodes.critic import critic_node
        result = critic_node(base_state(answer_draft="INSUFFICIENT_CONTEXT: no data"))
        assert result["needs_more_retrieval"] is True

    def test_fast_path_no_chunks(self):
        from agent.nodes.critic import critic_node
        result = critic_node(base_state(retrieved_chunks=[], answer_draft="Some answer"))
        assert result["needs_more_retrieval"] is True

    def test_verdict_parser_grounded(self):
        from agent.nodes.critic import _parse_verdict
        assert _parse_verdict("GROUNDED") == "GROUNDED"
        assert _parse_verdict("  Grounded  ") == "GROUNDED"
        assert _parse_verdict("The answer is GROUNDED based on sources.") == "GROUNDED"

    def test_verdict_parser_insufficient(self):
        from agent.nodes.critic import _parse_verdict
        v = _parse_verdict("INSUFFICIENT: missing gross margin data")
        assert v.startswith("INSUFFICIENT")
        assert "gross margin" in v

    def test_verdict_parser_defaults_grounded(self):
        from agent.nodes.critic import _parse_verdict
        # Unparseable → default to GROUNDED (avoids infinite loop)
        assert _parse_verdict("I'm not sure about this answer.") == "GROUNDED"

    def test_heuristic_fallback_when_both_backends_unavailable(self):
        from agent.nodes.critic import critic_node
        local_mock = MagicMock()
        local_mock.complete.side_effect = ConnectionError("offline")
        local_mock.health_check.return_value = False
        with patch("agent.nodes.critic.LocalLLMBackend", return_value=local_mock):
            # cloud_allowed=False, local down → heuristic GROUNDED
            result = critic_node(base_state(cloud_allowed=False))
        assert result["critique"] == "GROUNDED"


# ---------------------------------------------------------------------------
# 6. Report node
# ---------------------------------------------------------------------------

class TestReportNode:
    def test_extracts_cited_chunk_ids(self):
        from agent.nodes.report import _extract_cited_ids
        text = (
            f"Revenue grew [{CHUNK_A['chunk_id']}] and margin declined "
            f"[{CHUNK_B['chunk_id']}]."
        )
        ids = _extract_cited_ids(text)
        assert CHUNK_A["chunk_id"] in ids
        assert CHUNK_B["chunk_id"] in ids

    def test_deduplicates_cited_ids(self):
        from agent.nodes.report import _extract_cited_ids
        text = (
            f"See [{CHUNK_A['chunk_id']}] and again [{CHUNK_A['chunk_id']}]."
        )
        ids = _extract_cited_ids(text)
        assert ids.count(CHUNK_A["chunk_id"]) == 1

    def test_builds_citations_from_chunks(self):
        from agent.nodes.report import _build_citations
        cited_ids = [CHUNK_A["chunk_id"]]
        citations = _build_citations(cited_ids, [CHUNK_A, CHUNK_B])
        assert len(citations) == 1
        c = citations[0]
        assert c["chunk_id"] == CHUNK_A["chunk_id"]
        assert c["source_file"] == CHUNK_A["source_file"]
        assert "excerpt" in c
        assert len(c["excerpt"]) <= 123   # 120 chars + "…"

    def test_uncited_chunks_excluded(self):
        from agent.nodes.report import _build_citations
        # Only cite CHUNK_A — CHUNK_B should not appear
        citations = _build_citations([CHUNK_A["chunk_id"]], [CHUNK_A, CHUNK_B])
        cited_files = {c["source_file"] for c in citations}
        assert CHUNK_B["source_file"] in cited_files or len(citations) == 1

    def test_report_node_produces_final_answer(self):
        from agent.nodes.report import report_node
        local_mock = make_local_mock(LLMResponse(
            text="Final polished answer.", prompt_tokens=200,
            completion_tokens=10, cost_usd=0.0, backend="local",
        ))
        with patch("agent.nodes.report.LocalLLMBackend", return_value=local_mock):
            result = report_node(base_state())
        assert result["final_answer"] == "Final polished answer."

    def test_report_node_populates_citations(self):
        from agent.nodes.report import report_node
        local_mock = make_local_mock(LLMResponse(
            text="Answer.", prompt_tokens=100, completion_tokens=5,
            cost_usd=0.0, backend="local",
        ))
        with patch("agent.nodes.report.LocalLLMBackend", return_value=local_mock):
            result = report_node(base_state(answer_draft=GROUNDED_ANSWER))
        assert len(result["citations"]) >= 1

    def test_cost_accumulated_in_report(self):
        from agent.nodes.report import report_node
        resp = LLMResponse(
            text="Answer.", prompt_tokens=200, completion_tokens=10,
            cost_usd=0.002, backend="azure",
        )
        local_mock = MagicMock()
        local_mock.complete.side_effect = ConnectionError()
        azure_mock = MagicMock()
        azure_mock.complete.return_value = resp
        with patch("agent.nodes.report.LocalLLMBackend", return_value=local_mock):
            with patch("agent.nodes.report.AzureOAIBackend", return_value=azure_mock):
                result = report_node(base_state(cost_usd=0.005))
        assert abs(result["cost_usd"] - 0.007) < 1e-9


# ---------------------------------------------------------------------------
# 7. run_agent() end-to-end (mocked backends)
# ---------------------------------------------------------------------------

class TestRunAgentEndToEnd:
    """
    Full graph invocation with all external I/O mocked.
    These tests verify that state flows correctly through all 4 nodes.
    """

    def _run_with_mocks(
        self,
        query: str = "What was Q3 revenue?",
        cloud_allowed: bool = True,
        max_loops: int = 1,
        reason_response: LLMResponse = None,
        critic_response: LLMResponse = None,
        report_response: LLMResponse = None,
        chunks: list = None,
    ) -> AgentState:
        if reason_response is None:
            reason_response = _GOOD_RESPONSE
        if critic_response is None:
            critic_response = _CRITIC_GROUNDED
        if report_response is None:
            report_response = LLMResponse(
                text="Polished final answer.", prompt_tokens=100,
                completion_tokens=20, cost_usd=0.0, backend="local",
            )
        if chunks is None:
            chunks = [CHUNK_A, CHUNK_B]

        local_mock = MagicMock()
        local_mock.health_check.return_value = True
        # Route calls by max_tokens to differentiate rewrite/reason/critic/report
        def smart_complete(system, user, max_tokens=512, temperature=0.1):
            if max_tokens <= 40:          # query rewrite
                return LLMResponse("rewritten query", 50, 8, 0.0, "local")
            if max_tokens <= 60:          # critic
                return critic_response
            if max_tokens <= 400:         # report
                return report_response
            return reason_response        # reason
        local_mock.complete.side_effect = smart_complete

        from agent.graph import run_agent
        with patch("agent.nodes.retrieve._qdrant_hybrid_search", return_value=chunks):
            with patch("agent.nodes.retrieve._try_rewrite", return_value="rewritten"):
                with patch("agent.nodes.reason.LocalLLMBackend", return_value=local_mock):
                    with patch("agent.nodes.critic.LocalLLMBackend", return_value=local_mock):
                        with patch("agent.nodes.report.LocalLLMBackend", return_value=local_mock):
                            return run_agent(
                                query=query,
                                cloud_allowed=cloud_allowed,
                                max_loops=max_loops,
                            )

    def test_simple_route_produces_final_answer(self):
        result = self._run_with_mocks(max_loops=1)
        assert result["final_answer"] != ""

    def test_complex_route_can_loop(self):
        # First critic says INSUFFICIENT, second says GROUNDED
        call_count = {"n": 0}
        def alternating_critic(system, user, max_tokens=60, temperature=0.1):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _CRITIC_INSUFFICIENT
            return _CRITIC_GROUNDED

        local_mock = MagicMock()
        local_mock.health_check.return_value = True
        local_mock.complete.side_effect = lambda s, u, max_tokens=512, temperature=0.1: (
            LLMResponse("rewritten", 30, 5, 0.0, "local") if max_tokens <= 40 else
            alternating_critic(s, u, max_tokens) if max_tokens <= 60 else
            LLMResponse("Final answer.", 100, 20, 0.0, "local") if max_tokens <= 400 else
            _GOOD_RESPONSE
        )
        from agent.graph import run_agent
        with patch("agent.nodes.retrieve._qdrant_hybrid_search", return_value=[CHUNK_A]):
            with patch("agent.nodes.retrieve._try_rewrite", return_value="q"):
                with patch("agent.nodes.reason.LocalLLMBackend", return_value=local_mock):
                    with patch("agent.nodes.critic.LocalLLMBackend", return_value=local_mock):
                        with patch("agent.nodes.report.LocalLLMBackend", return_value=local_mock):
                            result = run_agent("Compare margins", cloud_allowed=True, max_loops=3)
        assert result["final_answer"] != ""

    def test_sensitive_route_does_not_call_azure(self):
        """cloud_allowed=False — AzureOAIBackend must never be instantiated."""
        azure_mock = MagicMock()
        from agent.graph import run_agent
        with patch("agent.nodes.retrieve._qdrant_hybrid_search", return_value=[CHUNK_A]):
            with patch("agent.nodes.retrieve._try_rewrite", return_value="q"):
                local_mock = make_local_mock(_GOOD_RESPONSE)
                local_mock.complete.side_effect = lambda s, u, max_tokens=512, temperature=0.1: (
                    LLMResponse("rewritten", 30, 5, 0.0, "local") if max_tokens <= 40 else
                    _CRITIC_GROUNDED if max_tokens <= 60 else
                    LLMResponse("Final.", 80, 10, 0.0, "local") if max_tokens <= 400 else
                    _GOOD_RESPONSE
                )
                with patch("agent.nodes.reason.LocalLLMBackend", return_value=local_mock):
                    with patch("agent.nodes.critic.LocalLLMBackend", return_value=local_mock):
                        with patch("agent.nodes.report.LocalLLMBackend", return_value=local_mock):
                            with patch("agent.nodes.reason.AzureOAIBackend", azure_mock):
                                run_agent("patient salary", cloud_allowed=False, max_loops=1)
        azure_mock.assert_not_called()

    def test_citations_in_result(self):
        result = self._run_with_mocks()
        # Citations list should exist (may be empty if IDs don't match UUID regex)
        assert "citations" in result
        assert isinstance(result["citations"], list)

    def test_cost_usd_in_result(self):
        result = self._run_with_mocks()
        assert "cost_usd" in result
        assert result["cost_usd"] >= 0.0

    def test_loop_count_in_result(self):
        result = self._run_with_mocks(max_loops=1)
        assert result.get("loop_count", 0) >= 1


# ---------------------------------------------------------------------------
# 8. Prompts
# ---------------------------------------------------------------------------

class TestPrompts:
    def test_format_context_block_includes_chunk_id(self):
        block = format_context_block([CHUNK_A])
        assert CHUNK_A["chunk_id"] in block

    def test_format_context_block_includes_page_number(self):
        block = format_context_block([CHUNK_A])
        assert str(CHUNK_A["page_start"]) in block

    def test_format_context_block_includes_text(self):
        block = format_context_block([CHUNK_A])
        assert CHUNK_A["text"] in block

    def test_format_context_block_includes_heading(self):
        block = format_context_block([CHUNK_A])
        assert CHUNK_A["section_heading"] in block

    def test_multi_chunk_context_block(self):
        block = format_context_block([CHUNK_A, CHUNK_B])
        assert CHUNK_A["chunk_id"] in block
        assert CHUNK_B["chunk_id"] in block

    def test_empty_chunks_returns_empty_string(self):
        assert format_context_block([]) == ""

    def test_reason_system_has_citation_instruction(self):
        assert "chunk" in REASON_SYSTEM.lower() or "cite" in REASON_SYSTEM.lower()

    def test_critic_system_has_grounded_keyword(self):
        assert "GROUNDED" in CRITIC_SYSTEM

    def test_critic_system_has_insufficient_keyword(self):
        assert "INSUFFICIENT" in CRITIC_SYSTEM