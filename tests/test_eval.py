"""
tests/test_eval.py
------------------
Tests for eval/ragas_eval.py and eval/benchmark.py.

All network calls (httpx to the agent API) and RAGAS/OpenAI calls are
mocked so the suite runs anywhere with zero infrastructure.

Coverage
--------
  ground_truth.json  : schema validation, route distribution, PII coverage
  load_ground_truth  : filtering by route, exclude_ids
  build_ragas_dataset: happy path, failed query handling, mock mode
  build_report       : score aggregation, CI gate logic, cost summary
  CI_THRESHOLDS      : values are within sensible ranges
  run_benchmark      : mock mode, route filtering, CSV/JSON output
  aggregate_results  : per-route stats, overall stats, edge cases
  _mock_measure      : deterministic per route
  _score_bar         : visual correctness
  _print_summary     : smoke test (no crash)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.ragas_eval import (
    CI_THRESHOLDS,
    GROUND_TRUTH_PATH,
    _mock_query_agent,
    _print_summary,
    _score_bar,
    build_ragas_dataset,
    build_report,
    load_ground_truth,
)
from eval.benchmark import (
    BENCHMARK_QUERIES,
    _mock_measure,
    _print_benchmark_summary,
    aggregate_results,
    run_benchmark,
    write_csv,
)


# ---------------------------------------------------------------------------
# ground_truth.json schema tests
# ---------------------------------------------------------------------------

class TestGroundTruthJSON:
    @pytest.fixture(scope="class")
    def data(self):
        return json.loads(GROUND_TRUTH_PATH.read_text())

    def test_file_exists(self):
        assert GROUND_TRUTH_PATH.exists()

    def test_has_25_items(self, data):
        assert len(data["items"]) == 25

    def test_all_items_have_required_keys(self, data):
        required = {"id", "route", "query", "ground_truth"}
        for item in data["items"]:
            missing = required - item.keys()
            assert not missing, f"{item['id']} missing keys: {missing}"

    def test_all_routes_valid(self, data):
        valid = {"SIMPLE", "COMPLEX", "SENSITIVE"}
        for item in data["items"]:
            assert item["route"] in valid, f"{item['id']} has invalid route"

    def test_ids_are_unique(self, data):
        ids = [i["id"] for i in data["items"]]
        assert len(ids) == len(set(ids))

    def test_route_distribution(self, data):
        from collections import Counter
        counts = Counter(i["route"] for i in data["items"])
        assert counts["SIMPLE"] >= 8
        assert counts["COMPLEX"] >= 8
        assert counts["SENSITIVE"] >= 2

    def test_sensitive_items_cover_pii_keywords(self, data):
        sensitive = [i for i in data["items"] if i["route"] == "SENSITIVE"]
        all_queries = " ".join(i["query"].lower() for i in sensitive)
        pii_keywords = ["salary", "ssn", "social security", "contact", "address"]
        found = [kw for kw in pii_keywords if kw in all_queries]
        assert len(found) >= 2, f"Expected PII keywords in SENSITIVE queries, found: {found}"

    def test_no_empty_queries(self, data):
        for item in data["items"]:
            assert item["query"].strip() != "", f"{item['id']} has empty query"

    def test_no_empty_ground_truths(self, data):
        for item in data["items"]:
            assert item["ground_truth"].strip() != "", f"{item['id']} has empty ground_truth"

    def test_complex_items_are_harder(self, data):
        """COMPLEX items should be longer questions on average."""
        simple_avg = sum(
            len(i["query"]) for i in data["items"] if i["route"] == "SIMPLE"
        ) / max(1, sum(1 for i in data["items"] if i["route"] == "SIMPLE"))
        complex_avg = sum(
            len(i["query"]) for i in data["items"] if i["route"] == "COMPLEX"
        ) / max(1, sum(1 for i in data["items"] if i["route"] == "COMPLEX"))
        assert complex_avg > simple_avg, "COMPLEX queries should be longer than SIMPLE"


# ---------------------------------------------------------------------------
# load_ground_truth
# ---------------------------------------------------------------------------

class TestLoadGroundTruth:
    def test_loads_all_by_default(self):
        items = load_ground_truth()
        assert len(items) == 25

    def test_filter_by_single_route(self):
        items = load_ground_truth(routes=["SIMPLE"])
        assert all(i["route"] == "SIMPLE" for i in items)
        assert len(items) >= 8

    def test_filter_by_multiple_routes(self):
        items = load_ground_truth(routes=["SIMPLE", "COMPLEX"])
        routes = {i["route"] for i in items}
        assert "SENSITIVE" not in routes
        assert "SIMPLE" in routes and "COMPLEX" in routes

    def test_exclude_specific_ids(self):
        all_items = load_ground_truth()
        first_id  = all_items[0]["id"]
        filtered  = load_ground_truth(exclude_ids=[first_id])
        assert len(filtered) == 24
        assert all(i["id"] != first_id for i in filtered)

    def test_filter_and_exclude_combined(self):
        sensitive = load_ground_truth(routes=["SENSITIVE"])
        sid = sensitive[0]["id"]
        result = load_ground_truth(routes=["SENSITIVE"], exclude_ids=[sid])
        assert len(result) == len(sensitive) - 1


# ---------------------------------------------------------------------------
# _mock_query_agent
# ---------------------------------------------------------------------------

class TestMockQueryAgent:
    def test_returns_dict_with_required_keys(self):
        resp = _mock_query_agent("What is revenue?", "Revenue was $2.16B.")
        required = {"query", "final_answer", "citations", "route_badge",
                    "confidence", "used_backend", "loop_count", "cost_usd",
                    "duration_seconds", "error"}
        assert required.issubset(resp.keys())

    def test_answer_matches_ground_truth(self):
        gt = "Revenue was $2.16B in Q3 2024."
        resp = _mock_query_agent("What is revenue?", gt)
        assert resp["final_answer"] == gt

    def test_citation_excerpt_is_truncated_gt(self):
        gt = "A" * 200
        resp = _mock_query_agent("q", gt)
        assert resp["citations"][0]["excerpt"] == gt[:120]


# ---------------------------------------------------------------------------
# build_ragas_dataset
# ---------------------------------------------------------------------------

class TestBuildRagasDataset:
    def _items(self, n: int = 3) -> list[dict]:
        return [
            {"id": f"t{i}", "route": "SIMPLE", "difficulty": "easy",
             "query": f"query {i}", "ground_truth": f"answer {i}"}
            for i in range(n)
        ]

    def test_mock_mode_returns_all_items(self):
        items  = self._items(5)
        rows   = build_ragas_dataset(items, api_url="http://unused", mock=True)
        assert len(rows) == 5

    def test_mock_mode_populates_ragas_fields(self):
        rows = build_ragas_dataset(self._items(2), api_url="http://x", mock=True)
        for row in rows:
            assert "question"     in row
            assert "answer"       in row
            assert "contexts"     in row
            assert "ground_truth" in row

    def test_mock_contexts_not_empty(self):
        rows = build_ragas_dataset(self._items(2), api_url="http://x", mock=True)
        for row in rows:
            assert len(row["contexts"]) >= 1

    def test_failed_query_captured_as_error_row(self):
        """If httpx raises, the row should have an error key and empty answer."""
        items = self._items(1)
        with patch("eval.ragas_eval.query_agent", side_effect=RuntimeError("conn refused")):
            rows = build_ragas_dataset(items, api_url="http://dead", mock=False, delay_seconds=0)
        assert len(rows) == 1
        assert rows[0]["error"] != ""
        assert rows[0]["answer"] == ""

    def test_metadata_preserved(self):
        items = [{"id": "x1", "route": "COMPLEX", "difficulty": "hard",
                  "query": "q", "ground_truth": "a"}]
        rows = build_ragas_dataset(items, api_url="http://x", mock=True)
        assert rows[0]["id"] == "x1"
        assert rows[0]["route"] == "COMPLEX"
        assert rows[0]["difficulty"] == "hard"

    def test_empty_input_returns_empty(self):
        rows = build_ragas_dataset([], api_url="http://x", mock=True)
        assert rows == []

    def test_live_mode_calls_query_agent(self):
        mock_resp = {
            "final_answer": "Revenue was $2.16B.",
            "citations": [{"chunk_id": "c1", "source_file": "s.pdf",
                           "page_start": 1, "page_end": 1,
                           "section_heading": "Rev", "excerpt": "Rev…"}],
            "route_badge": {"label": "SIMPLE", "color": "green",
                            "cloud_allowed": True, "tooltip": "ok"},
            "confidence": 0.9, "used_backend": "local", "loop_count": 1,
            "cost_usd": 0.0, "duration_seconds": 1.2, "error": "",
        }
        items = self._items(2)
        with patch("eval.ragas_eval.query_agent", return_value=mock_resp):
            rows = build_ragas_dataset(items, api_url="http://x", mock=False, delay_seconds=0)
        assert len(rows) == 2
        assert rows[0]["used_backend"] == "local"


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------

def _sample_rows(n: int = 10) -> list[dict]:
    rows = []
    for i in range(n):
        route = ["SIMPLE", "COMPLEX"][i % 2]
        rows.append({
            "id": f"r{i}", "route": route, "difficulty": "medium",
            "question": f"q{i}", "answer": f"a{i}",
            "contexts": ["ctx"], "ground_truth": f"gt{i}",
            "used_backend": "local" if i % 3 != 0 else "azure",
            "cost_usd": 0.001 if i % 3 == 0 else 0.0,
            "loop_count": i % 3 + 1, "duration_s": 1.5 + i * 0.1, "error": "",
        })
    return rows


_SCORES = {"faithfulness": 0.85, "answer_relevancy": 0.80, "context_recall": 0.75}
_THRESHOLDS = {"faithfulness": 0.80, "answer_relevancy": 0.75, "context_recall": 0.70}


class TestBuildReport:
    @pytest.fixture
    def report(self):
        return build_report(
            rows=_sample_rows(10),
            scores=_SCORES,
            thresholds=_THRESHOLDS,
            duration_total=45.2,
            meta={"api_url": "http://localhost:8000", "mock": False,
                  "judge_model": "gpt-4o", "routes_filter": None},
        )

    def test_report_has_top_level_keys(self, report):
        required = {"metadata", "scores", "ci_gate", "cost_summary",
                    "route_breakdown", "per_item_results"}
        assert required.issubset(report.keys())

    def test_scores_rounded(self, report):
        for v in report["scores"].values():
            assert isinstance(v, float)
            assert 0.0 <= v <= 1.0

    def test_ci_gate_passes_when_all_above_threshold(self, report):
        assert report["ci_gate"]["passed"] is True

    def test_ci_gate_fails_when_below_threshold(self):
        low_scores = {"faithfulness": 0.60, "answer_relevancy": 0.50,
                      "context_recall": 0.40}
        report = build_report(_sample_rows(), low_scores, _THRESHOLDS, 1.0, {})
        assert report["ci_gate"]["passed"] is False

    def test_ci_gate_partial_fail(self):
        mixed = {"faithfulness": 0.85, "answer_relevancy": 0.70,  # below 0.75
                 "context_recall": 0.80}
        report = build_report(_sample_rows(), mixed, _THRESHOLDS, 1.0, {})
        assert report["ci_gate"]["passed"] is False
        assert report["ci_gate"]["results"]["answer_relevancy"]["passed"] is False
        assert report["ci_gate"]["results"]["faithfulness"]["passed"] is True

    def test_per_item_results_count(self, report):
        assert len(report["per_item_results"]) == 10

    def test_per_item_no_raw_answer_text(self, report):
        for item in report["per_item_results"]:
            assert "answer" not in item, "Raw answer text must not appear in per_item_results"

    def test_cost_summary_present(self, report):
        cost = report["cost_summary"]
        assert "total_cost_usd" in cost
        assert "pct_local" in cost
        assert 0 <= cost["pct_local"] <= 100

    def test_route_breakdown_keys(self, report):
        breakdown = report["route_breakdown"]
        assert "SIMPLE" in breakdown
        assert "COMPLEX" in breakdown

    def test_route_breakdown_avg_duration_positive(self, report):
        for route, info in report["route_breakdown"].items():
            assert info["avg_duration_s"] >= 0

    def test_timestamp_in_metadata(self, report):
        assert "timestamp_utc" in report["metadata"]

    def test_error_rows_counted(self):
        rows = _sample_rows(5)
        rows[0]["error"] = "timeout"
        report = build_report(rows, _SCORES, _THRESHOLDS, 10.0, {})
        assert report["metadata"]["error_count"] == 1


# ---------------------------------------------------------------------------
# CI_THRESHOLDS
# ---------------------------------------------------------------------------

class TestCIThresholds:
    def test_all_three_metrics_present(self):
        assert "faithfulness" in CI_THRESHOLDS
        assert "answer_relevancy" in CI_THRESHOLDS
        assert "context_recall" in CI_THRESHOLDS

    def test_thresholds_are_floats(self):
        for v in CI_THRESHOLDS.values():
            assert isinstance(v, float)

    def test_thresholds_in_sensible_range(self):
        for metric, thresh in CI_THRESHOLDS.items():
            assert 0.5 <= thresh <= 0.95, (
                f"{metric} threshold {thresh} is outside [0.5, 0.95]"
            )

    def test_faithfulness_is_highest_bar(self):
        """Faithfulness is the most critical metric — should have highest threshold."""
        assert CI_THRESHOLDS["faithfulness"] >= CI_THRESHOLDS["context_recall"]


# ---------------------------------------------------------------------------
# _score_bar
# ---------------------------------------------------------------------------

class TestScoreBar:
    def test_perfect_score_all_filled(self):
        bar = _score_bar(1.0, width=10)
        assert bar == "[" + "█" * 10 + "]"

    def test_zero_score_all_empty(self):
        bar = _score_bar(0.0, width=10)
        assert bar == "[" + "░" * 10 + "]"

    def test_half_score(self):
        bar = _score_bar(0.5, width=10)
        assert bar.count("█") == 5
        assert bar.count("░") == 5

    def test_bar_total_length_fixed(self):
        for score in [0.0, 0.33, 0.67, 1.0]:
            bar = _score_bar(score, width=10)
            # brackets + 10 chars
            assert len(bar) == 12


# ---------------------------------------------------------------------------
# _print_summary (smoke test)
# ---------------------------------------------------------------------------

class TestPrintSummary:
    def test_does_not_crash(self, capsys):
        report = build_report(
            rows=_sample_rows(6),
            scores=_SCORES,
            thresholds=_THRESHOLDS,
            duration_total=20.0,
            meta={"api_url": "x", "mock": True, "judge_model": "gpt-4o",
                  "routes_filter": None},
        )
        _print_summary(report)
        captured = capsys.readouterr()
        assert "faithfulness" in captured.out
        assert "PASS" in captured.out or "FAIL" in captured.out


# ---------------------------------------------------------------------------
# benchmark — _mock_measure
# ---------------------------------------------------------------------------

class TestMockMeasure:
    def test_simple_faster_than_complex(self):
        # Run multiple times to account for jitter
        simple_times  = [_mock_measure("q", "SIMPLE")["total_ms"] for _ in range(20)]
        complex_times = [_mock_measure("q", "COMPLEX")["total_ms"] for _ in range(20)]
        assert sum(simple_times) / len(simple_times) < sum(complex_times) / len(complex_times)

    def test_sensitive_cloud_disabled(self):
        m = _mock_measure("q", "SENSITIVE")
        assert m["cloud_allowed"] is False

    def test_sensitive_always_local(self):
        for _ in range(10):
            m = _mock_measure("q", "SENSITIVE")
            assert m["used_backend"] == "local"

    def test_simple_zero_cost(self):
        for _ in range(10):
            m = _mock_measure("q", "SIMPLE")
            assert m["cost_usd"] == 0.0

    def test_result_has_required_keys(self):
        m = _mock_measure("q", "COMPLEX")
        required = {"total_ms", "ttft_ms", "used_backend", "loop_count",
                    "cost_usd", "confidence", "answer_len", "num_citations",
                    "route_label", "cloud_allowed", "error"}
        assert required.issubset(m.keys())


# ---------------------------------------------------------------------------
# benchmark — run_benchmark (mock mode)
# ---------------------------------------------------------------------------

class TestRunBenchmark:
    def test_mock_returns_correct_count(self):
        queries = [
            {"route": "SIMPLE",   "difficulty": "easy",   "query": "q1"},
            {"route": "COMPLEX",  "difficulty": "medium", "query": "q2"},
            {"route": "SENSITIVE","difficulty": "easy",   "query": "q3"},
        ]
        results = run_benchmark(
            api_url="http://unused", queries=queries, num_runs=2,
            mock=True, measure_ttft=False, warmup_run=False,
        )
        assert len(results) == 6   # 3 queries × 2 runs

    def test_each_result_has_run_number(self):
        queries = [{"route": "SIMPLE", "difficulty": "easy", "query": "q"}]
        results = run_benchmark(
            api_url="http://x", queries=queries, num_runs=3,
            mock=True, measure_ttft=False, warmup_run=False,
        )
        run_nums = sorted(r["run"] for r in results)
        assert run_nums == [1, 2, 3]

    def test_result_includes_route_metadata(self):
        queries = [{"route": "COMPLEX", "difficulty": "hard", "query": "compare margins"}]
        results = run_benchmark(
            api_url="http://x", queries=queries, num_runs=1,
            mock=True, measure_ttft=False, warmup_run=False,
        )
        assert results[0]["route"] == "COMPLEX"
        assert results[0]["difficulty"] == "hard"

    def test_empty_queries_returns_empty(self):
        results = run_benchmark(
            api_url="http://x", queries=[], num_runs=3,
            mock=True, measure_ttft=False, warmup_run=False,
        )
        assert results == []


# ---------------------------------------------------------------------------
# aggregate_results
# ---------------------------------------------------------------------------

class TestAggregateResults:
    def _results(self) -> list[dict]:
        return [
            {"route": "SIMPLE",  "total_ms": 800.0, "ttft_ms": 320.0,
             "used_backend": "local", "loop_count": 1, "cost_usd": 0.0,
             "confidence": 0.92, "answer_len": 200, "num_citations": 2,
             "route_label": "SIMPLE", "cloud_allowed": True, "error": "",
             "run": 1, "query_id": "s1", "difficulty": "easy", "query": "q"},
            {"route": "SIMPLE",  "total_ms": 850.0, "ttft_ms": 340.0,
             "used_backend": "local", "loop_count": 1, "cost_usd": 0.0,
             "confidence": 0.91, "answer_len": 210, "num_citations": 2,
             "route_label": "SIMPLE", "cloud_allowed": True, "error": "",
             "run": 2, "query_id": "s1", "difficulty": "easy", "query": "q"},
            {"route": "COMPLEX", "total_ms": 3200.0, "ttft_ms": 1100.0,
             "used_backend": "azure", "loop_count": 2, "cost_usd": 0.003,
             "confidence": 0.88, "answer_len": 450, "num_citations": 4,
             "route_label": "COMPLEX", "cloud_allowed": True, "error": "",
             "run": 1, "query_id": "c1", "difficulty": "hard", "query": "q2"},
            {"route": "COMPLEX", "total_ms": -1.0, "ttft_ms": -1.0,
             "used_backend": "error", "loop_count": 0, "cost_usd": 0.0,
             "confidence": 0.0, "answer_len": 0, "num_citations": 0,
             "route_label": "COMPLEX", "cloud_allowed": True,
             "error": "timeout", "run": 2, "query_id": "c1",
             "difficulty": "hard", "query": "q2"},
        ]

    def test_route_keys_present(self):
        agg = aggregate_results(self._results())
        assert "SIMPLE" in agg
        assert "COMPLEX" in agg
        assert "_overall" in agg

    def test_simple_all_local(self):
        agg = aggregate_results(self._results())
        assert agg["SIMPLE"]["pct_local"] == 100.0

    def test_complex_has_error(self):
        agg = aggregate_results(self._results())
        assert agg["COMPLEX"]["n_error"] == 1
        assert agg["COMPLEX"]["n_success"] == 1

    def test_overall_total_count(self):
        agg = aggregate_results(self._results())
        assert agg["_overall"]["n_total"] == 4
        assert agg["_overall"]["n_success"] == 3   # one error row excluded

    def test_latency_mean_reasonable(self):
        agg = aggregate_results(self._results())
        # SIMPLE mean should be between 800 and 850
        assert 800 <= agg["SIMPLE"]["latency_ms"]["mean"] <= 860

    def test_empty_results_returns_only_overall(self):
        agg = aggregate_results([])
        assert agg["_overall"]["n_total"] == 0

    def test_cost_aggregation(self):
        agg = aggregate_results(self._results())
        # Only one COMPLEX success with cost 0.003
        assert agg["COMPLEX"]["total_cost_usd"] == pytest.approx(0.003, abs=1e-6)


# ---------------------------------------------------------------------------
# write_csv
# ---------------------------------------------------------------------------

class TestWriteCSV:
    def test_csv_created(self, tmp_path):
        results = [
            {"route": "SIMPLE", "total_ms": 800.0, "cost_usd": 0.0,
             "used_backend": "local", "run": 1, "error": ""}
        ]
        path = tmp_path / "bench.csv"
        write_csv(results, path)
        assert path.exists()

    def test_csv_has_header(self, tmp_path):
        results = [{"route": "SIMPLE", "total_ms": 900.0, "run": 1}]
        path = tmp_path / "bench.csv"
        write_csv(results, path)
        lines = path.read_text().splitlines()
        assert lines[0] == "route,total_ms,run"

    def test_csv_empty_input_no_crash(self, tmp_path):
        path = tmp_path / "bench.csv"
        write_csv([], path)
        assert not path.exists()   # nothing written for empty input


# ---------------------------------------------------------------------------
# BENCHMARK_QUERIES sanity
# ---------------------------------------------------------------------------

class TestBenchmarkQueries:
    def test_all_three_routes_present(self):
        routes = {q["route"] for q in BENCHMARK_QUERIES}
        assert routes == {"SIMPLE", "COMPLEX", "SENSITIVE"}

    def test_no_empty_queries(self):
        for q in BENCHMARK_QUERIES:
            assert q["query"].strip() != ""

    def test_sensitive_queries_contain_pii_signals(self):
        sensitive = [q for q in BENCHMARK_QUERIES if q["route"] == "SENSITIVE"]
        keywords  = ["salary", "salary", "contact", "address", "ssn", "personal"]
        all_text  = " ".join(q["query"].lower() for q in sensitive)
        found = [kw for kw in keywords if kw in all_text]
        assert len(found) >= 1