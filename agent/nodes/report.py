"""
agent/nodes/report.py
---------------------
LangGraph node: produces the final polished answer and structured citations.

Two responsibilities
--------------------
1. Citation extraction: scan the draft for [chunk_id] references and
   build Citation objects with source file, page range, and excerpt.
   This populates the UI's "source cards" shown below the answer.

2. Answer formatting: call the report LLM prompt to rewrite the draft
   in clean professional prose (or a polite refusal for INSUFFICIENT).

This node never loops — it always terminates the graph.
Cost is minimal: the report prompt is short and temperature=0.
"""

from __future__ import annotations

import re
from loguru import logger

from agent.state import AgentState, Citation, RetrievedChunk
from agent.prompts import REPORT_SYSTEM, REPORT_USER
from agent.backends.local_llm import LocalLLMBackend
from agent.backends.azure_oai import AzureOAIBackend

_CITATION_RE = re.compile(r"\[([a-f0-9\-]{8,})\]")   # UUID-style chunk IDs


def report_node(state: AgentState) -> dict:
    """
    LangGraph node — writes final_answer and citations to state.
    """
    answer_draft: str = state.get("answer_draft", "")
    chunks: list[RetrievedChunk] = state.get("retrieved_chunks", [])
    cloud_allowed: bool = state.get("cloud_allowed", True)
    prev_cost: float = state.get("cost_usd", 0.0)

    # ── Extract citations before rewriting ────────────────────────────
    cited_ids = _extract_cited_ids(answer_draft)
    citations = _build_citations(cited_ids, chunks)

    # ── Rewrite with report prompt ────────────────────────────────────
    user_prompt = REPORT_USER.format(answer_draft=answer_draft)
    final_text, cost = _call_report(
        system=REPORT_SYSTEM,
        user=user_prompt,
        cloud_allowed=cloud_allowed,
    )

    logger.info(
        f"[report] {len(citations)} citations, "
        f"answer_len={len(final_text)} chars, "
        f"total_cost=${prev_cost + cost:.6f}"
    )

    return {
        "final_answer": final_text,
        "citations": citations,
        "cost_usd": prev_cost + cost,
    }


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------

def _extract_cited_ids(text: str) -> list[str]:
    """Return unique chunk IDs referenced in the answer, in order of appearance."""
    seen: set[str] = set()
    ordered: list[str] = []
    for chunk_id in _CITATION_RE.findall(text):
        if chunk_id not in seen:
            seen.add(chunk_id)
            ordered.append(chunk_id)
    return ordered


def _build_citations(
    cited_ids: list[str],
    chunks: list[RetrievedChunk],
) -> list[Citation]:
    """
    Build Citation objects for each chunk_id referenced in the answer.
    Chunks not cited are silently ignored.
    """
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    citations: list[Citation] = []
    for chunk_id in cited_ids:
        chunk = chunk_by_id.get(chunk_id)
        if not chunk:
            continue
        excerpt = chunk["text"][:120].replace("\n", " ").strip()
        if len(chunk["text"]) > 120:
            excerpt += "…"
        citations.append(Citation(
            chunk_id=chunk_id,
            source_file=chunk["source_file"],
            page_start=chunk["page_start"],
            page_end=chunk["page_end"],
            section_heading=chunk["section_heading"],
            excerpt=excerpt,
        ))
    return citations


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_report(system: str, user: str, cloud_allowed: bool) -> tuple[str, float]:
    """
    Returns (final_answer_text, cost_usd).
    Falls back gracefully: local → azure → raw draft passthrough.
    """
    # Try local
    try:
        local = LocalLLMBackend()
        resp = local.complete(system=system, user=user, max_tokens=400, temperature=0.1)
        return resp.text, resp.cost_usd
    except Exception as exc:
        logger.warning(f"[report] local unavailable: {exc}")

    # Cloud fallback
    if cloud_allowed:
        try:
            azure = AzureOAIBackend()
            resp = azure.complete(system=system, user=user, max_tokens=400, temperature=0.1)
            return resp.text, resp.cost_usd
        except Exception as exc:
            logger.warning(f"[report] azure unavailable: {exc}")

    # Passthrough — return raw draft if no LLM available
    logger.warning("[report] all backends unavailable — returning draft as-is")
    return user.split("Draft answer:")[-1].strip(), 0.0