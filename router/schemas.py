"""
router/schemas.py
-----------------
Data contracts for the routing layer.

RouteDecision is the single object that flows from the classifier into:
  - the LangGraph agent (selects execution path)
  - the FastAPI response (surfaced to the UI as a badge)
  - the cost_logger (tracks which route was used per query)
  - the audit log (SENSITIVE decisions written to append-only file)

Everything is a plain dataclass — no Pydantic dependency here so this
module imports in < 5 ms even in cold-start Lambda / ACA environments.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

Route = Literal["SIMPLE", "COMPLEX", "SENSITIVE"]


@dataclass
class RouteDecision:
    """
    Output of SensitivityClassifier.classify().

    Fields
    ------
    route           : Which execution path to take.
    confidence      : Float in [0, 1]. Not a calibrated probability —
                      treat as a relative signal, not a percentage.
    cloud_allowed   : Hard flag. False means the agent MUST NOT call
                      any external API for this query, regardless of
                      confidence or loop count.
    reason          : Human-readable explanation for the routing decision.
                      Written to audit log and surfaced in the UI tooltip.
    pii_triggers    : Which specific PII signals fired (empty if none).
    complexity_score: Raw score before thresholding (useful for debugging).
    query_word_count: Cached so downstream nodes don't recompute.
    doc_count       : Number of distinct source documents in scope.
    timestamp_utc   : Unix timestamp at classification time.
    """

    route: Route
    confidence: float
    cloud_allowed: bool
    reason: str
    pii_triggers: list[str] = field(default_factory=list)
    complexity_score: int = 0
    query_word_count: int = 0
    doc_count: int = 1
    timestamp_utc: float = field(default_factory=time.time)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_sensitive(self) -> bool:
        return self.route == "SENSITIVE"

    @property
    def needs_agent(self) -> bool:
        """True when the 3-node LangGraph loop should run."""
        return self.route in ("COMPLEX", "SENSITIVE")

    @property
    def allow_cloud_fallback(self) -> bool:
        """Alias with an intent-revealing name for use in agent nodes."""
        return self.cloud_allowed

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-serialisable dict for logging and API responses."""
        return {
            "route": self.route,
            "confidence": round(self.confidence, 4),
            "cloud_allowed": self.cloud_allowed,
            "reason": self.reason,
            "pii_triggers": self.pii_triggers,
            "complexity_score": self.complexity_score,
            "query_word_count": self.query_word_count,
            "doc_count": self.doc_count,
            "timestamp_utc": self.timestamp_utc,
        }

    def to_ui_badge(self) -> dict:
        """Minimal payload for the React route badge component."""
        _colors = {
            "SIMPLE": "green",
            "COMPLEX": "blue",
            "SENSITIVE": "red",
        }
        return {
            "label": self.route,
            "color": _colors[self.route],
            "cloud_allowed": self.cloud_allowed,
            "tooltip": self.reason,
        }