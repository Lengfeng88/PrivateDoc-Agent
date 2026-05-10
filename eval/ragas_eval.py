"""
eval/ragas_eval.py
------------------
RAGAS evaluation harness for the PrivateDoc Intelligence Agent.

What this does
--------------
1. Loads ground_truth.json (25 Q&A pairs).
2. Runs each question through the live agent (POST /query).
3. Collects: question, answer, contexts (retrieved chunks), ground_truth.
4. Feeds the dataset into RAGAS and computes:
     - faithfulness        : is the answer supported by the retrieved context?
     - answer_relevancy    : does the answer actually address the question?
     - context_recall      : did retrieval surface the relevant information?
5. Writes a timestamped JSON report to eval/results/.
6. Optionally enforces CI thresholds (--ci flag) — exits non-zero if any
   metric falls below the configured gate.

Usage
-----
# Full eval against live API (needs Qdrant + LLM running):
    python eval/ragas_eval.py \\
        --api-url http://localhost:8000 \\
        --openai-key $OPENAI_API_KEY \\
        --output eval/results/report.json

# CI gate (fails if faithfulness < 0.80):
    python eval/ragas_eval.py --ci --api-url http://localhost:8000

# Dry-run with mock API (no LLM needed — for testing the harness itself):
    python eval/ragas_eval.py --mock

Architecture note
-----------------
RAGAS uses an LLM judge (OpenAI GPT-4o by default) to score faithfulness
and answer_relevancy.  context_recall is computed by comparing retrieved
chunk content against the ground_truth string.

The harness is designed to be LLM-judge-agnostic: the --judge-model flag
accepts any OpenAI-compatible endpoint (including a local vLLM server),
so you can run offline evals on the 4080 without an API key.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

# ---------------------------------------------------------------------------
# CI thresholds — configurable, enforced by --ci flag
# ---------------------------------------------------------------------------

CI_THRESHOLDS: dict[str, float] = {
    "faithfulness":      0.80,
    "answer_relevancy":  0.75,
    "context_recall":    0.70,
}

# ---------------------------------------------------------------------------
# Ground truth loader
# ---------------------------------------------------------------------------

GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.json"


def load_ground_truth(
    routes: Optional[list[str]] = None,
    exclude_ids: Optional[list[str]] = None,
) -> list[dict]:
    """
    Load Q&A pairs from ground_truth.json.

    Parameters
    ----------
    routes : list[str] | None
        Filter by route labels, e.g. ["SIMPLE", "COMPLEX"].
        None means include all.
    exclude_ids : list[str] | None
        Skip specific item IDs (e.g. hard-refusal tests that return no answer).
    """
    data = json.loads(GROUND_TRUTH_PATH.read_text())
    items = data["items"]

    if routes:
        items = [i for i in items if i["route"] in routes]
    if exclude_ids:
        items = [i for i in items if i["id"] not in exclude_ids]

    logger.info(f"Loaded {len(items)} ground truth items")
    return items


# ---------------------------------------------------------------------------
# Agent client
# ---------------------------------------------------------------------------

def query_agent(
    api_url: str,
    query: str,
    pii_flagged: bool = False,
    pii_types: Optional[list[str]] = None,
    timeout: float = 60.0,
) -> dict:
    """
    Call POST /query on the live API and return the QueryResponse dict.
    Raises on HTTP error or timeout.
    """
    import httpx

    payload = {
        "query": query,
        "pii_flagged": pii_flagged,
        "pii_types": pii_types or [],
    }
    response = httpx.post(
        f"{api_url.rstrip('/')}/query",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _mock_query_agent(query: str, gt_answer: str) -> dict:
    """
    Synthetic agent response for --mock mode (no API needed).
    Returns a plausible but deterministic response for harness testing.
    """
    return {
        "query": query,
        "final_answer": gt_answer,      # perfect answer — gives upper-bound scores
        "citations": [{
            "chunk_id": "mock-chunk-001",
            "source_file": "shopify_q3_2024.pdf",
            "page_start": 1,
            "page_end": 2,
            "section_heading": "Revenue Overview",
            "excerpt": gt_answer[:120],
        }],
        "route_badge": {"label": "SIMPLE", "color": "green",
                        "cloud_allowed": True, "tooltip": "mock"},
        "confidence": 0.92,
        "used_backend": "local",
        "loop_count": 1,
        "cost_usd": 0.0,
        "duration_seconds": 0.01,
        "error": "",
    }


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_ragas_dataset(
    ground_truth_items: list[dict],
    api_url: str,
    mock: bool = False,
    delay_seconds: float = 1.0,
) -> list[dict]:
    """
    Runs each ground truth item through the agent and assembles the
    RAGAS-compatible dataset rows.

    RAGAS expects four fields per row:
      - question    : str
      - answer      : str  (agent's output)
      - contexts    : list[str]  (retrieved chunk texts)
      - ground_truth: str  (reference answer)

    Returns a list of dicts with these fields plus PrivateDoc metadata.
    """
    rows = []
    total = len(ground_truth_items)

    for idx, item in enumerate(ground_truth_items, start=1):
        qid   = item["id"]
        query = item["query"]
        gt    = item["ground_truth"]
        route = item["route"]

        logger.info(f"[{idx}/{total}] {qid} ({route}): {query[:60]}…")
        t0 = time.perf_counter()

        try:
            if mock:
                resp = _mock_query_agent(query, gt)
            else:
                pii_flagged = route == "SENSITIVE"
                resp = query_agent(api_url, query, pii_flagged=pii_flagged)

            answer   = resp.get("final_answer", "")
            contexts = [c["excerpt"] for c in resp.get("citations", [])]
            # Fallback: if no citations, use empty context so RAGAS still runs
            if not contexts:
                contexts = ["[no context retrieved]"]

            duration = time.perf_counter() - t0
            rows.append({
                # RAGAS core fields
                "question":     query,
                "answer":       answer,
                "contexts":     contexts,
                "ground_truth": gt,
                # PrivateDoc metadata (not used by RAGAS, kept for reporting)
                "id":           qid,
                "route":        route,
                "difficulty":   item.get("difficulty", "unknown"),
                "used_backend": resp.get("used_backend", "unknown"),
                "cost_usd":     resp.get("cost_usd", 0.0),
                "loop_count":   resp.get("loop_count", 0),
                "duration_s":   round(duration, 3),
                "error":        resp.get("error", ""),
            })

        except Exception as exc:
            logger.error(f"[{idx}/{total}] {qid} FAILED: {exc}")
            rows.append({
                "question": query, "answer": "", "contexts": [],
                "ground_truth": gt, "id": qid, "route": route,
                "difficulty": item.get("difficulty", "unknown"),
                "used_backend": "error", "cost_usd": 0.0,
                "loop_count": 0, "duration_s": 0.0, "error": str(exc),
            })

        # Polite delay to avoid hammering the local LLM
        if not mock and idx < total:
            time.sleep(delay_seconds)

    return rows


# ---------------------------------------------------------------------------
# RAGAS scorer
# ---------------------------------------------------------------------------

def run_ragas(
    rows: list[dict],
    openai_api_key: str,
    judge_model: str = "gpt-4o",
    judge_base_url: Optional[str] = None,
) -> dict[str, float]:
    """
    Feed the dataset into RAGAS and return per-metric float scores.

    Parameters
    ----------
    openai_api_key : str
        API key for the LLM judge.  If judge_base_url is set, this key
        is used against a local OpenAI-compatible endpoint (e.g. vLLM).
    judge_model : str
        Model name for the RAGAS judge.
    judge_base_url : str | None
        Override the OpenAI base URL.  Set to http://localhost:8080/v1
        to use a local llama.cpp server as the judge (fully offline eval).

    Returns
    -------
    dict mapping metric_name → float score in [0, 1].
    """
    import os
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    os.environ["OPENAI_API_KEY"] = openai_api_key

    # Configure the judge LLM
    judge_kwargs: dict = {"model": judge_model, "temperature": 0}
    if judge_base_url:
        judge_kwargs["base_url"] = judge_base_url
        logger.info(f"Using local judge at {judge_base_url}")
    else:
        logger.info(f"Using OpenAI judge: {judge_model}")

    llm = ChatOpenAI(**judge_kwargs)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Build HuggingFace Dataset (RAGAS input format)
    ragas_rows = [
        {
            "question":     r["question"],
            "answer":       r["answer"],
            "contexts":     r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in rows
        if not r.get("error")  # skip failed queries
    ]

    if not ragas_rows:
        raise ValueError("No successful query results to evaluate.")

    dataset = Dataset.from_list(ragas_rows)
    logger.info(f"Running RAGAS on {len(ragas_rows)} rows…")

    metrics = [faithfulness, answer_relevancy, context_recall]
    for m in metrics:
        m.llm = llm
        if hasattr(m, "embeddings"):
            m.embeddings = embeddings

    result = evaluate(dataset, metrics=metrics)

    scores = {
        "faithfulness":     float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "context_recall":   float(result["context_recall"]),
    }
    return scores


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(
    rows: list[dict],
    scores: dict[str, float],
    thresholds: dict[str, float],
    duration_total: float,
    meta: dict,
) -> dict:
    """
    Assemble the full JSON report saved to eval/results/.
    Includes: aggregate scores, per-route breakdown, cost summary,
    per-item results, CI gate pass/fail.
    """
    # Per-route score breakdown
    route_groups: dict[str, list[dict]] = {}
    for row in rows:
        route_groups.setdefault(row["route"], []).append(row)

    route_breakdown: dict[str, dict] = {}
    for route, group_rows in route_groups.items():
        answered = [r for r in group_rows if not r.get("error")]
        route_breakdown[route] = {
            "count":          len(group_rows),
            "answered":       len(answered),
            "avg_duration_s": round(
                sum(r["duration_s"] for r in answered) / len(answered), 3
            ) if answered else 0.0,
            "avg_cost_usd":   round(
                sum(r["cost_usd"] for r in answered) / len(answered), 6
            ) if answered else 0.0,
            "avg_loops":      round(
                sum(r["loop_count"] for r in answered) / len(answered), 2
            ) if answered else 0.0,
        }

    # Cost summary
    total_cost  = sum(r["cost_usd"] for r in rows)
    local_count = sum(1 for r in rows if r["used_backend"] == "local")
    azure_count = sum(1 for r in rows if r["used_backend"] == "azure")
    error_count = sum(1 for r in rows if r.get("error"))

    # CI gate
    gate_results = {
        metric: {
            "score":     scores.get(metric, 0.0),
            "threshold": thresholds[metric],
            "passed":    scores.get(metric, 0.0) >= thresholds[metric],
        }
        for metric in thresholds
    }
    all_passed = all(g["passed"] for g in gate_results.values())

    return {
        "metadata": {
            **meta,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "total_items":   len(rows),
            "error_count":   error_count,
            "duration_total_s": round(duration_total, 1),
        },
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "ci_gate": {
            "passed":  all_passed,
            "results": gate_results,
        },
        "cost_summary": {
            "total_cost_usd":  round(total_cost, 6),
            "local_queries":   local_count,
            "azure_queries":   azure_count,
            "pct_local":       round(local_count / max(len(rows), 1) * 100, 1),
            "avg_cost_per_query_usd": round(
                total_cost / max(len(rows) - error_count, 1), 6
            ),
        },
        "route_breakdown":  route_breakdown,
        "per_item_results": [
            {
                "id":          r["id"],
                "route":       r["route"],
                "difficulty":  r["difficulty"],
                "used_backend":r["used_backend"],
                "duration_s":  r["duration_s"],
                "cost_usd":    r["cost_usd"],
                "loop_count":  r["loop_count"],
                "answer_len":  len(r["answer"]),
                "error":       r.get("error", ""),
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RAGAS evaluation harness for PrivateDoc Intelligence Agent"
    )
    p.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the running FastAPI server (default: http://localhost:8000)",
    )
    p.add_argument(
        "--openai-key",
        default=None,
        help="OpenAI API key for the RAGAS judge LLM. "
             "Falls back to OPENAI_API_KEY env var.",
    )
    p.add_argument(
        "--judge-model",
        default="gpt-4o",
        help="OpenAI-compatible model name for RAGAS judge (default: gpt-4o)",
    )
    p.add_argument(
        "--judge-base-url",
        default=None,
        help="Base URL for a local OpenAI-compatible judge (e.g. http://localhost:8080/v1)",
    )
    p.add_argument(
        "--routes",
        nargs="*",
        default=None,
        choices=["SIMPLE", "COMPLEX", "SENSITIVE"],
        help="Evaluate only these route types (default: all)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Path to write the JSON report (default: eval/results/report_<timestamp>.json)",
    )
    p.add_argument(
        "--ci",
        action="store_true",
        help="Exit non-zero if any metric falls below CI thresholds",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="Use mock agent responses (harness testing — no API or LLM needed)",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to wait between agent calls (default: 1.0)",
    )
    p.add_argument(
        "--skip-sensitive",
        action="store_true",
        default=True,
        help="Exclude SENSITIVE items from RAGAS scoring (they test refusal, not quality)",
    )
    return p.parse_args()


def main() -> int:
    import os
    args = _parse_args()

    # Resolve OpenAI key
    openai_key = args.openai_key or os.getenv("OPENAI_API_KEY", "")
    if not openai_key and not args.mock:
        logger.error(
            "No OpenAI API key found. "
            "Set OPENAI_API_KEY or pass --openai-key, "
            "or use --judge-base-url with a local server, "
            "or use --mock for a dry run."
        )
        return 1

    # Load ground truth
    routes_filter = args.routes
    gt_items = load_ground_truth(routes=routes_filter)

    # SENSITIVE items test refusal behaviour, not answer quality.
    # Exclude them from RAGAS by default (they'd score 0 faithfulness by design).
    sensitive_ids = [i["id"] for i in gt_items if i["route"] == "SENSITIVE"]
    ragas_items   = [i for i in gt_items if i["route"] != "SENSITIVE"]

    logger.info(
        f"Running eval on {len(ragas_items)} items "
        f"(excluded {len(sensitive_ids)} SENSITIVE items from RAGAS)"
    )

    # Query the agent
    t0 = time.perf_counter()
    rows = build_ragas_dataset(
        ground_truth_items=ragas_items,
        api_url=args.api_url,
        mock=args.mock,
        delay_seconds=args.delay,
    )

    # Score with RAGAS
    if args.mock:
        # In mock mode, return deterministic near-perfect scores
        scores = {
            "faithfulness":     0.97,
            "answer_relevancy": 0.95,
            "context_recall":   0.93,
        }
        logger.info(f"Mock scores: {scores}")
    else:
        scores = run_ragas(
            rows=rows,
            openai_api_key=openai_key,
            judge_model=args.judge_model,
            judge_base_url=args.judge_base_url,
        )

    duration_total = time.perf_counter() - t0

    # Build report
    report = build_report(
        rows=rows,
        scores=scores,
        thresholds=CI_THRESHOLDS,
        duration_total=duration_total,
        meta={
            "api_url":      args.api_url,
            "judge_model":  args.judge_model,
            "mock":         args.mock,
            "routes_filter":routes_filter,
        },
    )

    # Write report
    output_path = Path(args.output) if args.output else (
        Path(__file__).parent / "results" /
        f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    logger.success(f"Report written to {output_path}")

    # Print summary
    _print_summary(report)

    # CI gate
    if args.ci and not report["ci_gate"]["passed"]:
        logger.error("CI GATE FAILED — one or more metrics below threshold")
        return 1

    return 0


def _print_summary(report: dict) -> None:
    """Pretty-print the evaluation summary to stdout."""
    scores    = report["scores"]
    gate      = report["ci_gate"]
    cost      = report["cost_summary"]
    meta      = report["metadata"]

    print("\n" + "=" * 60)
    print("  PrivateDoc RAGAS Evaluation Report")
    print("=" * 60)
    print(f"  Items evaluated : {meta['total_items']} "
          f"({meta['error_count']} errors)")
    print(f"  Total duration  : {meta['duration_total_s']:.1f}s")
    print(f"  Local queries   : {cost['local_queries']} "
          f"({cost['pct_local']}%)")
    print(f"  Azure queries   : {cost['azure_queries']}")
    print(f"  Total cost      : ${cost['total_cost_usd']:.4f}")
    print(f"  Avg cost/query  : ${cost['avg_cost_per_query_usd']:.6f}")
    print()
    print("  Metric Scores vs Thresholds:")
    for metric, gate_info in gate["results"].items():
        status  = "✅ PASS" if gate_info["passed"] else "❌ FAIL"
        score   = gate_info["score"]
        thresh  = gate_info["threshold"]
        bar     = _score_bar(score)
        print(f"    {metric:<22} {score:.4f}  (≥{thresh})  {bar}  {status}")
    print()
    print(f"  CI Gate: {'✅ PASSED' if gate['passed'] else '❌ FAILED'}")
    print()
    print("  Per-Route Breakdown:")
    for route, info in report["route_breakdown"].items():
        print(f"    {route:<12} {info['answered']}/{info['count']} answered  "
              f"avg {info['avg_duration_s']:.2f}s  "
              f"avg ${info['avg_cost_usd']:.6f}")
    print("=" * 60 + "\n")


def _score_bar(score: float, width: int = 10) -> str:
    """ASCII progress bar for score display."""
    filled = round(score * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


if __name__ == "__main__":
    sys.exit(main())