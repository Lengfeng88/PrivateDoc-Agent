"""
router/rules.py
---------------
All tunable classification constants live here — never buried in logic.

Keeping rules in a separate module means:
  - A non-engineer can audit/update the PII keyword list without touching code.
  - Unit tests can import and mutate these lists directly without monkey-patching.
  - The classifier's logic stays readable regardless of how long the lists grow.

Sections
--------
1. PII_KEYWORDS          — hard triggers for SENSITIVE route, cloud disabled.
2. PII_REGEX_LABELS      — structural patterns (SSN, email…) detected by loader.
3. COMPLEXITY_SIGNALS    — vocabulary that implies multi-step reasoning.
4. THRESHOLDS            — numeric knobs for route decisions.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. PII keyword triggers (query-level check, case-insensitive substring match)
# ---------------------------------------------------------------------------
# These are words that, when present in the *query*, indicate the user is
# asking about personally-identifiable or legally sensitive content.
# Even if the underlying document was not flagged, the intent of the question
# may expose PII — so we gate on both.

PII_QUERY_KEYWORDS: frozenset[str] = frozenset({
    # identity
    "ssn", "social security", "passport", "driver's license", "driver license",
    "national id", "sin number", "tax id", "ein",
    # health
    "diagnosis", "patient", "medical record", "prescription", "mrn",
    "health card", "insurance claim", "icd code",
    # financial-personal
    "salary", "compensation", "payroll", "bank account", "routing number",
    "credit card", "sin", "date of birth", "dob",
    # legal
    "confidential", "privileged", "attorney-client", "sealed",
})

# Structural PII types surfaced by ingest/loader.py detect_pii()
# If a chunk carries any of these, the whole query is SENSITIVE.
PII_STRUCTURAL_TYPES: frozenset[str] = frozenset({
    "ssn", "credit_card", "passport", "dob", "medical",
})

# ---------------------------------------------------------------------------
# 2. Complexity signals (query-level, multi-signal scoring)
# ---------------------------------------------------------------------------
# Each match increments the complexity score by 1.
# Reaching COMPLEXITY_THRESHOLD triggers the COMPLEX route.

COMPLEXITY_SIGNALS: frozenset[str] = frozenset({
    # comparative / temporal reasoning
    "compare", "versus", "vs", "difference between", "contrast",
    "year over year", "yoy", "quarter over quarter", "qoq",
    "trend", "over time", "historically", "since", "changed",
    # causal / analytical
    "why", "what drove", "what caused", "explain", "reason for",
    "breakdown", "contributing factor", "impact of",
    # multi-document / cross-reference
    "across", "multiple", "both", "all segments", "each",
    "segment", "by region", "by product",
    # synthesis
    "summarize", "overview", "highlight", "key takeaway",
    "what are the main", "what does management say",
})

# ---------------------------------------------------------------------------
# 3. Numeric thresholds
# ---------------------------------------------------------------------------

THRESHOLDS: dict[str, int | float] = {
    # Complexity score >= this → COMPLEX route
    "complexity_min_score": 2,

    # Query word count >= this adds +1 to complexity score
    # (long questions imply multi-part intent even without trigger words)
    "long_query_words": 20,

    # Number of distinct source documents >= this → forces COMPLEX
    # (cross-document reasoning always needs the agent loop)
    "multi_doc_count": 2,

    # If local LLM confidence signal < this threshold → allow cloud escalation
    # (placeholder: used by the agent node, not the classifier)
    "local_confidence_floor": 0.55,
}