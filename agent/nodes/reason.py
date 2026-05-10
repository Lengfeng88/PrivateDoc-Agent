"""
agent/nodes/reason.py
---------------------
LangGraph node: generates a draft answer from retrieved chunks.

Backend selection logic
-----------------------
1. Always try local first (LocalLLMBackend → llama.cpp on RTX 4080).
2. If local is unreachable AND cloud_allowed is True → fall back to Azure.
3. If local is unreachable AND cloud_allowed is False (SENSITIVE route)
   → return a hard refusal rather than leaking to cloud.

This is the only node that touches an LLM, so it's also where token
costs are accumulated and the 'used_backend' field is set.

Confidence parsing
------------------
The reason prompt instructs the model to end its response with:
    CONFIDENCE: 0.XX
This node parses that line and stores it in state['confidence'].
The critic uses the draft text; the report node uses the clean answer.
"""

from __future__ import annotations

import re
from loguru import logger

from agent.state import AgentState
from agent.prompts import REASON_SYSTEM, REASON_USER, format_context_block
from agent.backends.local_llm import LocalLLMBackend
from agent.backends.azure_oai import AzureOAIBackend

# Regex to extract the CONFIDENCE line from the model's output.
_CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*(-?[0-9.]+)", re.IGNORECASE)

# If local LLM returns this, we trigger another retrieval loop.
_INSUFFICIENT_MARKER = "INSUFFICIENT_CONTEXT"


def reason_node(state: AgentState) -> dict:
    """
    LangGraph node — generates answer_draft, confidence, cost_usd, used_backend.
    """
    query: str = state["query"]
    chunks: list = state.get("retrieved_chunks", [])
    cloud_allowed: bool = state.get("cloud_allowed", True)
    prev_cost: float = state.get("cost_usd", 0.0)

    if not chunks:
        logger.warning("[reason] no chunks available — returning insufficient marker")
        return {
            "answer_draft": _INSUFFICIENT_MARKER + ": no documents were retrieved.",
            "confidence": 0.0,
            "used_backend": "none",
            "cost_usd": prev_cost,
        }

    context_block = format_context_block(chunks)
    user_prompt = REASON_USER.format(query=query, context_block=context_block)

    # ── Backend selection ─────────────────────────────────────────────
    response, backend_name = _call_with_fallback(
        system=REASON_SYSTEM,
        user=user_prompt,
        cloud_allowed=cloud_allowed,
    )

    if response is None:
        # Both backends failed — return a safe refusal
        return {
            "answer_draft": _INSUFFICIENT_CONTEXT_REFUSAL,
            "confidence": 0.0,
            "used_backend": "none",
            "cost_usd": prev_cost,
        }

    raw_text = response.text
    confidence = _parse_confidence(raw_text)
    clean_draft = _strip_confidence_line(raw_text)

    logger.info(
        f"[reason] backend={backend_name} "
        f"confidence={confidence:.2f} "
        f"cost=${response.cost_usd:.6f} "
        f"tokens={response.prompt_tokens}+{response.completion_tokens}"
    )

    return {
        "answer_draft": clean_draft,
        "confidence": confidence,
        "used_backend": backend_name,
        "cost_usd": prev_cost + response.cost_usd,
    }


# ---------------------------------------------------------------------------
# Backend orchestration
# ---------------------------------------------------------------------------

def _call_with_fallback(system: str, user: str, cloud_allowed: bool):
    """
    Try local → Azure fallback.
    Returns (LLMResponse | None, backend_name_str).
    """
    # 1. Try local
    try:
        local = LocalLLMBackend()
        resp = local.complete(system=system, user=user, max_tokens=600)
        return resp, "local"
    except ConnectionError as exc:
        logger.warning(f"[reason] local LLM unavailable: {exc}")
    except Exception as exc:
        logger.warning(f"[reason] local LLM error: {exc}")

    # 2. Cloud fallback — only if router permits
    if not cloud_allowed:
        logger.warning(
            "[reason] cloud fallback blocked by router (SENSITIVE route) "
            "— returning refusal"
        )
        return None, "blocked"

    try:
        azure = AzureOAIBackend()
        resp = azure.complete(system=system, user=user, max_tokens=600)
        return resp, "azure"
    except RuntimeError as exc:
        logger.error(f"[reason] Azure OAI credentials missing: {exc}")
    except Exception as exc:
        logger.error(f"[reason] Azure OAI error: {exc}")

    return None, "none"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_confidence(text: str) -> float:
    m = _CONFIDENCE_RE.search(text)
    if m:
        try:
            val = float(m.group(1))
            return max(0.0, min(1.0, val))
        except ValueError:
            pass
    # If the model said INSUFFICIENT, confidence is implicitly low.
    if _INSUFFICIENT_MARKER in text:
        return 0.1
    return 0.5   # default when model omits the line


def _strip_confidence_line(text: str) -> str:
    """Remove the CONFIDENCE: line from the draft so the critic sees clean prose."""
    return _CONFIDENCE_RE.sub("", text).strip()


_INSUFFICIENT_CONTEXT_REFUSAL = (
    "INSUFFICIENT_CONTEXT: The system was unable to generate an answer "
    "because both the local and cloud language model backends are "
    "currently unavailable. Please try again later."
)