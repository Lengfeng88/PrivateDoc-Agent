"""
agent/graph.py
--------------
Builds and compiles the PrivateDoc LangGraph agent.

Graph topology
--------------

                    ┌─────────────┐
         invoke()──▶│   retrieve  │◀──────────────┐
                    └──────┬──────┘               │ (needs_more_retrieval=True
                           │                      │  AND loop_count < MAX_LOOP)
                           ▼                      │
                    ┌─────────────┐               │
                    │    reason   │               │
                    └──────┬──────┘               │
                           │                      │
                           ▼                      │
                    ┌─────────────┐               │
                    │    critic   │───────────────▶┘
                    └──────┬──────┘
                           │ (needs_more_retrieval=False
                           │  OR loop_count >= MAX_LOOP)
                           ▼
                    ┌─────────────┐
                    │    report   │
                    └──────┬──────┘
                           │
                           ▼
                          END

Key design decisions
--------------------
1. Single graph for all three routes (SIMPLE / COMPLEX / SENSITIVE).
   The route only affects:
     - cloud_allowed flag threaded through state (backends check this)
     - MAX_LOOP: SIMPLE routes get max_loops=1 injected at invoke() time

2. Conditional edge on 'critic':
     needs_more_retrieval=True  AND loop_count < MAX_LOOP → retrieve
     otherwise                                            → report

3. No parallel branches — deliberate choice. The IBM Consulting use case
   is latency-tolerant (analysts, not real-time APIs) so sequential
   retrieve→reason→critic is simpler to debug and audit than concurrent.

4. The compiled graph is module-level singleton (build_graph() is called
   once at import time). Thread-safe: LangGraph compiled graphs are
   stateless between invocations.

Usage
-----
    from agent.graph import agent_graph

    # Synchronous invoke (blocking)
    result = agent_graph.invoke({
        "query": "Compare gross margin Q3 2024 vs Q3 2023",
        "cloud_allowed": True,
        "route_decision": decision.to_dict(),
    })
    print(result["final_answer"])
    print(result["citations"])

    # Streaming (yields node-level dicts)
    for chunk in agent_graph.stream({...}):
        print(chunk)   # {"retrieve": {...}} then {"reason": {...}} etc.
"""

from __future__ import annotations

import os
from typing import Literal

from langgraph.graph import StateGraph, END
from loguru import logger

from agent.state import AgentState
from agent.nodes.retrieve import retrieve_node
from agent.nodes.reason   import reason_node
from agent.nodes.critic   import critic_node
from agent.nodes.report   import report_node

MAX_LOOP = int(os.getenv("AGENT_MAX_LOOPS", "3"))


# ---------------------------------------------------------------------------
# Conditional edge function
# ---------------------------------------------------------------------------

def _route_after_critic(
    state: AgentState,
) -> Literal["retrieve", "report"]:
    """
    Called by LangGraph after the critic node completes.

    Returns "retrieve" to loop back, "report" to terminate.

    Two conditions must BOTH be true to loop:
      1. Critic said needs_more_retrieval = True
      2. We haven't exceeded MAX_LOOP (loop_count is post-increment from retrieve)
    """
    needs_more: bool = state.get("needs_more_retrieval", False)
    loop_count: int  = state.get("loop_count", 0)

    if needs_more and loop_count < MAX_LOOP:
        logger.debug(
            f"[graph] critic→retrieve (loop={loop_count}, needs_more={needs_more})"
        )
        return "retrieve"

    logger.debug(
        f"[graph] critic→report (loop={loop_count}, needs_more={needs_more})"
    )
    return "report"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Construct and compile the agent StateGraph.

    Returns a compiled graph ready for .invoke() and .stream().
    Called once at module load — result cached as `agent_graph`.
    """
    g = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────
    g.add_node("retrieve", retrieve_node)
    g.add_node("reason",   reason_node)
    g.add_node("critic",   critic_node)
    g.add_node("report",   report_node)

    # ── Entry point ───────────────────────────────────────────────────
    g.set_entry_point("retrieve")

    # ── Linear edges ─────────────────────────────────────────────────
    g.add_edge("retrieve", "reason")
    g.add_edge("reason",   "critic")

    # ── Conditional edge: critic → retrieve OR report ─────────────────
    g.add_conditional_edges(
        "critic",
        _route_after_critic,
        {
            "retrieve": "retrieve",
            "report":   "report",
        },
    )

    # ── Terminal edge ─────────────────────────────────────────────────
    g.add_edge("report", END)

    return g.compile()


# ---------------------------------------------------------------------------
# Module-level compiled graph — import and use directly
# ---------------------------------------------------------------------------

agent_graph = build_graph()


# ---------------------------------------------------------------------------
# Convenience invoke wrapper
# ---------------------------------------------------------------------------

def run_agent(
    query: str,
    cloud_allowed: bool = True,
    route_decision: dict | None = None,
    max_loops: int | None = None,
) -> AgentState:
    """
    High-level entry point used by the FastAPI layer.

    Parameters
    ----------
    query          : User's question string.
    cloud_allowed  : From RouteDecision.cloud_allowed — False for SENSITIVE.
    route_decision : RouteDecision.to_dict() for logging/observability.
    max_loops      : Override MAX_LOOP env var (e.g. pass 1 for SIMPLE route).

    Returns the final AgentState dict after graph completion.
    """
    # SIMPLE route: cap loops at 1 (single retrieve→reason→critic→report)
    effective_max = max_loops if max_loops is not None else MAX_LOOP
    # Inject into env so all nodes see it (cheap, process-scoped)
    os.environ["AGENT_MAX_LOOPS"] = str(effective_max)

    initial_state: AgentState = {
        "query":           query,
        "cloud_allowed":   cloud_allowed,
        "route_decision":  route_decision or {},
        "loop_count":      0,
        "cost_usd":        0.0,
        "retrieved_chunks": [],
        "answer_draft":    "",
        "critique":        "",
        "needs_more_retrieval": False,
        "final_answer":    "",
        "citations":       [],
        "confidence":      0.0,
        "used_backend":    "",
        "retrieval_query": "",
        "error":           "",
    }

    logger.info(
        f"[graph] invoke | query='{query[:60]}' "
        f"cloud={cloud_allowed} max_loops={effective_max}"
    )

    result: AgentState = agent_graph.invoke(initial_state)

    logger.success(
        f"[graph] done | backend={result.get('used_backend')} "
        f"cost=${result.get('cost_usd', 0):.6f} "
        f"citations={len(result.get('citations', []))}"
    )
    return result