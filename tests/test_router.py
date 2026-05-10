"""
tests/test_router.py
--------------------
Unit tests for router/classifier.py, router/schemas.py, router/rules.py.

Coverage goals
--------------
  - Every classification gate has at least 3 tests:
      one that fires it, one that narrowly misses it, one at the exact boundary.
  - Audit log is tested for both write behaviour and privacy (no raw text stored).
  - RouteDecision serialisation is tested for UI and API consumers.
  - classify_batch is tested for the CI eval harness.
  - Rules module constants are sanity-checked so a bad edit fails loudly.

Run with:  pytest tests/test_router.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from router.classifier import SensitivityClassifier, classify_batch
from router.schemas import RouteDecision
from router import rules


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clf(tmp_path):
    """Classifier with audit log pointing at a temp file."""
    return SensitivityClassifier(audit_log_path=tmp_path / "audit.jsonl")


@pytest.fixture
def clf_no_audit():
    """Classifier with audit logging disabled — for tests that don't need it."""
    return SensitivityClassifier(audit_log_path=None)


def meta(doc_count=1, pii_flagged=False, pii_types=None) -> dict:
    """Shorthand for building doc_meta dicts."""
    return {
        "doc_count": doc_count,
        "pii_flagged": pii_flagged,
        "pii_types": pii_types or [],
    }


# ---------------------------------------------------------------------------
# Gate 1: SENSITIVE — query keyword path
# ---------------------------------------------------------------------------

class TestSensitiveQueryKeywords:
    def test_ssn_keyword_fires(self, clf):
        d = clf.classify("What is the employee SSN on file?", meta())
        assert d.route == "SENSITIVE"
        assert d.cloud_allowed is False

    def test_salary_keyword_fires(self, clf):
        d = clf.classify("Show me the salary breakdown for Q3", meta())
        assert d.route == "SENSITIVE"

    def test_medical_keyword_fires(self, clf):
        d = clf.classify("Retrieve the patient diagnosis from the report", meta())
        assert d.route == "SENSITIVE"

    def test_passport_keyword_fires(self, clf):
        d = clf.classify("Find passport number for the applicant", meta())
        assert d.route == "SENSITIVE"

    def test_confidential_keyword_fires(self, clf):
        d = clf.classify("Summarise the confidential memo", meta())
        assert d.route == "SENSITIVE"

    def test_case_insensitive_match(self, clf):
        d = clf.classify("SALARY information for last quarter", meta())
        assert d.route == "SENSITIVE"

    def test_keyword_in_middle_of_sentence(self, clf):
        d = clf.classify("Can you find the diagnosis date from the attachment?", meta())
        assert d.route == "SENSITIVE"

    def test_non_pii_financial_query_not_sensitive(self, clf):
        d = clf.classify("What was total revenue in Q3 2024?", meta())
        assert d.route != "SENSITIVE"

    def test_gross_margin_not_sensitive(self, clf):
        d = clf.classify("Compare gross margin year over year", meta())
        assert d.route != "SENSITIVE"

    def test_pii_triggers_populated(self, clf):
        d = clf.classify("What is the SSN and dob for this employee?", meta())
        assert len(d.pii_triggers) >= 2
        trigger_strings = " ".join(d.pii_triggers)
        assert "ssn" in trigger_strings
        assert "dob" in trigger_strings


# ---------------------------------------------------------------------------
# Gate 1: SENSITIVE — document-level PII path
# ---------------------------------------------------------------------------

class TestSensitiveDocPii:
    def test_doc_pii_flagged_true_fires(self, clf):
        d = clf.classify(
            "Summarise this document",
            meta(pii_flagged=True, pii_types=["ssn"]),
        )
        assert d.route == "SENSITIVE"
        assert d.cloud_allowed is False

    def test_structural_pii_type_fires(self, clf):
        d = clf.classify(
            "What does this file contain?",
            meta(pii_flagged=True, pii_types=["credit_card"]),
        )
        assert d.route == "SENSITIVE"

    def test_loader_flagged_without_type_still_sensitive(self, clf):
        # Loader said pii_flagged=True but didn't identify type — still gate.
        d = clf.classify(
            "Tell me about this document",
            meta(pii_flagged=True, pii_types=[]),
        )
        assert d.route == "SENSITIVE"

    def test_non_structural_pii_type_does_not_fire(self, clf):
        # "email" is detected by loader but not in PII_STRUCTURAL_TYPES —
        # doc is not gated unless pii_flagged is also True.
        d = clf.classify(
            "What are the contact details?",
            meta(pii_flagged=False, pii_types=["email"]),
        )
        assert d.route != "SENSITIVE"

    def test_combined_query_and_doc_pii(self, clf):
        d = clf.classify(
            "Find the SSN in this file",
            meta(pii_flagged=True, pii_types=["ssn"]),
        )
        assert d.route == "SENSITIVE"
        # Both query and doc triggers should appear
        trigger_types = {t.split(":")[0] for t in d.pii_triggers}
        assert "query" in trigger_types
        assert "doc" in trigger_types

    def test_sensitive_confidence_is_high(self, clf):
        d = clf.classify("Show salary data", meta())
        assert d.confidence >= 0.95

    def test_sensitive_route_cloud_always_false(self, clf):
        # Verify cloud_allowed is False regardless of doc_meta content.
        for doc_count in (1, 5, 10):
            d = clf.classify(
                "Retrieve the patient diagnosis",
                meta(doc_count=doc_count),
            )
            assert d.cloud_allowed is False, f"cloud_allowed should be False for SENSITIVE (doc_count={doc_count})"


# ---------------------------------------------------------------------------
# Gate 2: COMPLEX
# ---------------------------------------------------------------------------

class TestComplex:
    def test_compare_keyword_contributes(self, clf_no_audit):
        d = clf_no_audit.classify(
            "Compare revenue versus last year and explain the trend",
            meta(),
        )
        assert d.route == "COMPLEX"

    def test_multi_doc_alone_does_not_trigger(self, clf_no_audit):
        # multi_doc adds +1 to score; threshold is 2; needs one more signal.
        d = clf_no_audit.classify(
            "What is the revenue?",
            meta(doc_count=2),
        )
        # score = 1 (multi_doc bonus) → below threshold of 2 → SIMPLE
        assert d.route == "SIMPLE"

    def test_multi_doc_plus_signal_triggers(self, clf_no_audit):
        # score = 1 (multi_doc) + 1 (signal) = 2 → COMPLEX
        d = clf_no_audit.classify(
            "Compare the two reports",
            meta(doc_count=2),
        )
        assert d.route == "COMPLEX"

    def test_long_query_plus_signal_triggers(self, clf_no_audit):
        # long_query_words threshold=20; "explain" signal=+1; long query=+1 → score=2
        # Must be > 20 words AND contain a signal word
        long_q = "explain " + "the financial performance results " * 6  # 25 words
        assert len(long_q.split()) >= 20
        d = clf_no_audit.classify(long_q, meta())
        assert d.route == "COMPLEX"

    def test_single_signal_below_threshold(self, clf_no_audit):
        d = clf_no_audit.classify("What was the trend in Q3?", meta())
        # score = 1 → below threshold
        assert d.route == "SIMPLE"

    def test_two_signals_at_threshold(self, clf_no_audit):
        d = clf_no_audit.classify(
            "Compare gross margin and explain what drove the change", meta()
        )
        assert d.route == "COMPLEX"

    def test_complex_cloud_allowed(self, clf_no_audit):
        d = clf_no_audit.classify(
            "Compare Q3 2024 versus Q3 2023 revenue breakdown", meta()
        )
        assert d.route == "COMPLEX"
        assert d.cloud_allowed is True

    def test_complexity_score_populated(self, clf_no_audit):
        d = clf_no_audit.classify(
            "Compare and contrast year over year margin trends", meta()
        )
        assert d.complexity_score >= 2

    def test_confidence_scales_with_score(self, clf_no_audit):
        low_q = "compare and summarize"   # score ≈ 2
        high_q = "compare versus contrast year over year breakdown across segments explain why"
        d_low = clf_no_audit.classify(low_q, meta())
        d_high = clf_no_audit.classify(high_q, meta())
        if d_low.route == "COMPLEX" and d_high.route == "COMPLEX":
            assert d_high.confidence >= d_low.confidence

    def test_needs_agent_true_for_complex(self, clf_no_audit):
        d = clf_no_audit.classify(
            "Compare revenue trends across all segments year over year", meta()
        )
        assert d.needs_agent is True


# ---------------------------------------------------------------------------
# Gate 3: SIMPLE
# ---------------------------------------------------------------------------

class TestSimple:
    def test_basic_factual_query(self, clf_no_audit):
        d = clf_no_audit.classify("What was total revenue in Q3?", meta())
        assert d.route == "SIMPLE"

    def test_single_doc_short_query(self, clf_no_audit):
        d = clf_no_audit.classify("What is the gross profit?", meta(doc_count=1))
        assert d.route == "SIMPLE"

    def test_simple_cloud_allowed(self, clf_no_audit):
        d = clf_no_audit.classify("What is the net income?", meta())
        assert d.cloud_allowed is True

    def test_simple_needs_agent_false(self, clf_no_audit):
        d = clf_no_audit.classify("List the subsidiaries.", meta())
        assert d.needs_agent is False

    def test_simple_confidence_reasonable(self, clf_no_audit):
        d = clf_no_audit.classify("What page is the balance sheet on?", meta())
        assert d.route == "SIMPLE"
        assert 0.7 <= d.confidence <= 1.0

    def test_empty_query_defaults_simple(self, clf_no_audit):
        d = clf_no_audit.classify("", meta())
        assert d.route == "SIMPLE"


# ---------------------------------------------------------------------------
# Gate ordering — SENSITIVE beats COMPLEX
# ---------------------------------------------------------------------------

class TestGateOrdering:
    def test_sensitive_beats_complex(self, clf):
        # Query has both complexity signals AND PII keyword.
        # SENSITIVE gate must fire first.
        d = clf.classify(
            "Compare the salary data year over year across all employees",
            meta(doc_count=2),
        )
        assert d.route == "SENSITIVE"
        assert d.cloud_allowed is False

    def test_doc_pii_beats_complexity(self, clf):
        d = clf.classify(
            "Compare results across all segments",
            meta(doc_count=3, pii_flagged=True, pii_types=["ssn"]),
        )
        assert d.route == "SENSITIVE"


# ---------------------------------------------------------------------------
# RouteDecision properties and serialisation
# ---------------------------------------------------------------------------

class TestRouteDecisionSchema:
    def _make(self, route="SIMPLE", cloud=True, pii=None, score=0) -> RouteDecision:
        return RouteDecision(
            route=route,
            confidence=0.85,
            cloud_allowed=cloud,
            reason="test",
            pii_triggers=pii or [],
            complexity_score=score,
            query_word_count=5,
            doc_count=1,
        )

    def test_is_sensitive_property(self):
        assert self._make("SENSITIVE").is_sensitive is True
        assert self._make("SIMPLE").is_sensitive is False

    def test_needs_agent_complex(self):
        assert self._make("COMPLEX").needs_agent is True

    def test_needs_agent_sensitive(self):
        assert self._make("SENSITIVE").needs_agent is True

    def test_needs_agent_simple(self):
        assert self._make("SIMPLE").needs_agent is False

    def test_allow_cloud_fallback(self):
        assert self._make("COMPLEX", cloud=True).allow_cloud_fallback is True
        assert self._make("SENSITIVE", cloud=False).allow_cloud_fallback is False

    def test_to_dict_has_all_keys(self):
        d = self._make("COMPLEX", score=3).to_dict()
        required = {
            "route", "confidence", "cloud_allowed", "reason",
            "pii_triggers", "complexity_score", "query_word_count",
            "doc_count", "timestamp_utc",
        }
        assert required.issubset(d.keys())

    def test_to_dict_confidence_rounded(self):
        d = RouteDecision(
            route="SIMPLE", confidence=0.123456789, cloud_allowed=True, reason="x"
        ).to_dict()
        # Should be rounded to 4 decimal places
        assert len(str(d["confidence"]).split(".")[-1]) <= 4

    def test_to_ui_badge_simple(self):
        badge = self._make("SIMPLE").to_ui_badge()
        assert badge["label"] == "SIMPLE"
        assert badge["color"] == "green"
        assert badge["cloud_allowed"] is True

    def test_to_ui_badge_sensitive(self):
        badge = self._make("SENSITIVE", cloud=False).to_ui_badge()
        assert badge["color"] == "red"
        assert badge["cloud_allowed"] is False

    def test_to_ui_badge_complex(self):
        badge = self._make("COMPLEX").to_ui_badge()
        assert badge["color"] == "blue"


# ---------------------------------------------------------------------------
# Audit log behaviour
# ---------------------------------------------------------------------------

class TestAuditLog:
    def test_sensitive_decision_written_to_log(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        clf = SensitivityClassifier(audit_log_path=log_path)
        clf.classify("Show me the patient diagnosis", meta())
        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["route"] == "SENSITIVE"

    def test_non_sensitive_not_written(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        clf = SensitivityClassifier(audit_log_path=log_path)
        clf.classify("What is total revenue?", meta())
        # Log file should not be created for non-sensitive queries
        assert not log_path.exists()

    def test_audit_log_has_no_raw_query_text(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        clf = SensitivityClassifier(audit_log_path=log_path)
        secret_query = "Show me the SSN for John Smith employee 1234"
        clf.classify(secret_query, meta())
        log_content = log_path.read_text()
        # The raw query must not appear in the log
        assert "John Smith" not in log_content
        assert "1234" not in log_content
        # But metadata should be there
        record = json.loads(log_content.strip())
        assert "query_word_count" in record
        assert record["cloud_allowed"] is False

    def test_multiple_sensitive_queries_append(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        clf = SensitivityClassifier(audit_log_path=log_path)
        clf.classify("Show SSN", meta())
        clf.classify("Patient diagnosis", meta())
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_audit_disabled_no_file_created(self, tmp_path):
        clf = SensitivityClassifier(audit_log_path=None)
        clf.classify("Show me the salary", meta())
        # No file should be created anywhere in tmp_path
        assert not any(tmp_path.iterdir())

    def test_audit_log_is_valid_jsonl(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        clf = SensitivityClassifier(audit_log_path=log_path)
        for i in range(5):
            clf.classify(f"Patient diagnosis query {i}", meta())
        for line in log_path.read_text().strip().split("\n"):
            obj = json.loads(line)   # raises if invalid JSON
            assert "timestamp_utc" in obj


# ---------------------------------------------------------------------------
# explain() helper
# ---------------------------------------------------------------------------

class TestExplain:
    def test_explain_returns_string(self, clf_no_audit):
        result = clf_no_audit.explain("What was revenue?", meta())
        assert isinstance(result, str)
        assert "Route" in result

    def test_explain_shows_pii_triggers(self, clf_no_audit):
        result = clf_no_audit.explain("Show me the SSN", meta())
        assert "ssn" in result.lower()

    def test_explain_shows_complexity_score(self, clf_no_audit):
        result = clf_no_audit.explain(
            "Compare and contrast year over year trends", meta()
        )
        if "COMPLEX" in result:
            assert "Complexity" in result


# ---------------------------------------------------------------------------
# classify_batch (CI eval harness)
# ---------------------------------------------------------------------------

class TestClassifyBatch:
    def test_returns_correct_count(self):
        items = [
            {"query": "What is revenue?", "doc_meta": meta()},
            {"query": "Compare margins year over year", "doc_meta": meta()},
            {"query": "Show patient SSN", "doc_meta": meta()},
        ]
        results = classify_batch(items)
        assert len(results) == 3

    def test_routes_match_expected(self):
        items = [
            {"query": "What is revenue?",                       "doc_meta": meta()},
            {"query": "Compare margins year over year",         "doc_meta": meta()},
            {"query": "Show patient SSN",                       "doc_meta": meta()},
        ]
        results = classify_batch(items)
        assert results[0].route == "SIMPLE"
        assert results[1].route == "COMPLEX"
        assert results[2].route == "SENSITIVE"

    def test_missing_doc_meta_uses_defaults(self):
        items = [{"query": "What is net income?"}]
        results = classify_batch(items)
        assert results[0].route == "SIMPLE"


# ---------------------------------------------------------------------------
# Rules module sanity checks
# ---------------------------------------------------------------------------

class TestRulesModule:
    def test_pii_keywords_not_empty(self):
        assert len(rules.PII_QUERY_KEYWORDS) > 0

    def test_complexity_signals_not_empty(self):
        assert len(rules.COMPLEXITY_SIGNALS) > 0

    def test_thresholds_have_required_keys(self):
        required = {"complexity_min_score", "long_query_words", "multi_doc_count"}
        assert required.issubset(rules.THRESHOLDS.keys())

    def test_complexity_threshold_is_positive_int(self):
        val = rules.THRESHOLDS["complexity_min_score"]
        assert isinstance(val, int)
        assert val >= 1

    def test_no_overlap_pii_and_complexity(self):
        # PII keywords and complexity signals should be disjoint —
        # a word in both would cause ambiguous routing.
        overlap = rules.PII_QUERY_KEYWORDS & rules.COMPLEXITY_SIGNALS
        assert not overlap, f"Overlapping terms: {overlap}"

    def test_all_pii_structural_types_are_strings(self):
        for t in rules.PII_STRUCTURAL_TYPES:
            assert isinstance(t, str)


# ---------------------------------------------------------------------------
# Metadata edge cases
# ---------------------------------------------------------------------------

class TestMetadataEdgeCases:
    def test_doc_count_zero_treated_as_one(self, clf_no_audit):
        d = clf_no_audit.classify("What is revenue?", meta(doc_count=0))
        assert d.route == "SIMPLE"

    def test_very_large_doc_count(self, clf_no_audit):
        d = clf_no_audit.classify("Compare across docs", meta(doc_count=100))
        # multi_doc=True → score += 1; "compare" → score += 1 → COMPLEX
        assert d.route == "COMPLEX"

    def test_empty_pii_types_list(self, clf_no_audit):
        d = clf_no_audit.classify("What is the balance?", meta(pii_types=[]))
        assert d.route == "SIMPLE"

    def test_unknown_pii_type_ignored(self, clf_no_audit):
        # Type not in PII_STRUCTURAL_TYPES should not trigger SENSITIVE
        # unless pii_flagged is True.
        d = clf_no_audit.classify(
            "What are the contacts?",
            meta(pii_flagged=False, pii_types=["unknown_type"]),
        )
        assert d.route != "SENSITIVE"