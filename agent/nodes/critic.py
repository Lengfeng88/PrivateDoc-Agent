"""
agent/nodes/critic.py
---------------------
LangGraph node: verifies the draft answer is grounded in retrieved chunks.

This node is the "brain" of the loop decision.
It outputs a critique string that the graph's conditional edge reads
to decide whether to loop back to retrieve or proceed to report.

Two fast-path exits (no LLM call needed)
-----------------------------------------
1. loop_count >= MAX_LOOP  → force GROUNDED to avoid infinite loops.
2. INSUFFICIENT_CONTEXT in draft → force INSUFFICIENT so we retry.

This keeps the critic cheap: LLM call only happens when genuinely needed.

Critique format (strict)
-------------------------
The critic prompt produces exactly one of:
    GROUNDED
    INSUFFICIENT: <one sentence>

The conditional edge in graph.py parses this to route the graph.
"""

from __future__ import annotations

import os
from loguru import logger

from agent.state import AgentState
from agent.prompts import CRITIC_SYSTEM, CRITIC_USER, format_context_block
from agent.backends.local_llm import LocalLLMBackend
from agent.backends.azure_oai import AzureOAIBackend

MAX_LOOP = int(os.getenv("AGENT_MAX_LOOPS", "3"))
_INSUFFICIENT_MARKER = "INSUFFICIENT_CONTEXT"


def critic_node(state: AgentState) -> dict:
    """
    LangGraph node — writes 'critique' and 'needs_more_retrieval' to state.
    """
    answer_draft: str = state.get("answer_draft", "")
    chunks: list = state.get("retrieved_chunks", [])
    query: str = state["query"]
    loop_count: int = state.get("loop_count", 0)
    cloud_allowed: bool = state.get("cloud_allowed", True)

    # ── Fast-path 1: loop guard ───────────────────────────────────────
    if loop_count >= MAX_LOOP:
        logger.info(f"[critic] loop cap hit ({loop_count}) — forcing GROUNDED")
        return {
            "critique": "GROUNDED",
            "needs_more_retrieval": False,
        }

    # ── Fast-path 2: model signalled insufficient context ─────────────
    if _INSUFFICIENT_MARKER in answer_draft:
        logger.info("[critic] draft contains INSUFFICIENT_CONTEXT — will retry")
        return {
            "critique": "INSUFFICIENT: model reported insufficient context",
            "needs_more_retrieval": True,
        }

    # ── Fast-path 3: no chunks at all ────────────────────────────────
    if not chunks:
        return {
            "critique": "INSUFFICIENT: no chunks were retrieved",
            "needs_more_retrieval": True,
        }

    # ── Full critic LLM call ──────────────────────────────────────────
    context_block = format_context_block(chunks)
    user_prompt = CRITIC_USER.format(
        query=query,
        answer_draft=answer_draft,
        context_block=context_block,
    )

    verdict = _call_critic(
        system=CRITIC_SYSTEM,
        user=user_prompt,
        cloud_allowed=cloud_allowed,
    )

    needs_more = verdict.upper().startswith("INSUFFICIENT")
    logger.info(f"[critic] verdict='{verdict[:60]}' needs_more={needs_more}")

    return {
        "critique": verdict,
        "needs_more_retrieval": needs_more,
    }


# ---------------------------------------------------------------------------
# LLM call — local first, cloud fallback, then heuristic fallback
# ---------------------------------------------------------------------------

def _call_critic(system: str, user: str, cloud_allowed: bool) -> str:
    """
    Returns the raw verdict string from the critic LLM.
    Falls back through: local → azure → heuristic.
    """
    # Try local
    try:
        local = LocalLLMBackend()
        resp = local.complete(system=system, user=user, max_tokens=60, temperature=0.0)
        return _parse_verdict(resp.text)
    except Exception as exc:
        logger.warning(f"[critic] local unavailable: {exc}")

    # Cloud fallback
    if cloud_allowed:
        try:
            azure = AzureOAIBackend()
            resp = azure.complete(system=system, user=user, max_tokens=60, temperature=0.0)
            return _parse_verdict(resp.text)
        except Exception as exc:
            logger.warning(f"[critic] azure unavailable: {exc}")

    # Heuristic fallback — if we have chunks, assume grounded
    logger.warning("[critic] all backends unavailable — using heuristic GROUNDED")
    return "GROUNDED"


def _parse_verdict(raw: str) -> str:
    """
    Extract the first line starting with GROUNDED or INSUFFICIENT.
    Handles models that add preamble before the verdict.
    """
    for line in raw.strip().splitlines():
        line = line.strip()
        upper = line.upper()
        if upper.startswith("GROUNDED"):
            return "GROUNDED"
        if upper.startswith("INSUFFICIENT"):
            # Preserve the full "INSUFFICIENT: <reason>" format
            return line
    # If no clean verdict found, treat as grounded to avoid infinite loop
    logger.warning(f"[critic] unparseable verdict: '{raw[:80]}' — defaulting GROUNDED")
    return "GROUNDED"