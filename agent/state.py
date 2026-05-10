"""
agent/state.py
--------------
Single shared data contract that flows through every LangGraph node.

Design notes
------------
LangGraph passes the entire state dict into each node and merges the
returned dict back in.  Every field here is therefore:

  - readable by any node
  - writable by at most ONE node (to avoid merge conflicts)
  - typed explicitly so the graph is self-documenting

Field ownership
---------------
  query, route_decision       set by the API layer before graph.invoke()
  retrieved_chunks            written by: retrieve node
  answer_draft                written by: reason node
  critique                    written by: critic node
  final_answer, citations     written by: report node
  loop_count                  incremented by: retrieve node (loop guard)
  cost_usd                    accumulated by: reason node (token accounting)
  error                       written by any node on unrecoverable failure
"""

from __future__ import annotations

from typing import Any, TypedDict


class RetrievedChunk(TypedDict):
    """One chunk returned from Qdrant hybrid search."""
    chunk_id: str
    text: str
    source_file: str
    page_start: int
    page_end: int
    section_heading: str
    score: float           # RRF-combined relevance score


class Citation(TypedDict):
    """Source reference attached to the final answer."""
    chunk_id: str
    source_file: str
    page_start: int
    page_end: int
    section_heading: str
    excerpt: str           # first 120 chars of the chunk — shown in UI card


class AgentState(TypedDict, total=False):
    """
    Mutable state bag threaded through the LangGraph pipeline.

    All fields are Optional (total=False) so nodes only need to return
    the keys they actually modified.  The graph will merge partial
    updates into the running state automatically.
    """

    # ── Input (set by API, never mutated by nodes) ───────────────────
    query: str                          # raw user query string
    route_decision: dict[str, Any]      # RouteDecision.to_dict() output
    cloud_allowed: bool                 # hard flag from router

    # ── Retrieval ────────────────────────────────────────────────────
    retrieved_chunks: list[RetrievedChunk]
    retrieval_query: str                # possibly rewritten query used for search
    loop_count: int                     # number of retrieve→reason cycles so far

    # ── Reasoning ────────────────────────────────────────────────────
    answer_draft: str                   # candidate answer from reason node
    used_backend: str                   # "local" | "azure" — for cost logging

    # ── Critic ───────────────────────────────────────────────────────
    critique: str                       # "GROUNDED" | "INSUFFICIENT: <why>"
    needs_more_retrieval: bool          # parsed from critique

    # ── Output ───────────────────────────────────────────────────────
    final_answer: str
    citations: list[Citation]
    confidence: float                   # 0–1 self-assessed by reason node
    cost_usd: float                     # accumulated token cost

    # ── Error handling ───────────────────────────────────────────────
    error: str                          # set on unrecoverable failure