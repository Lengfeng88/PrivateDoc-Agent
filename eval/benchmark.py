"""
eval/benchmark.py
-----------------
Latency and cost profiling for the PrivateDoc Intelligence Agent.

What this does
--------------
Runs a configurable set of queries through each route type (SIMPLE,
COMPLEX, SENSITIVE) and measures:
  - Time-to-first-token (TTFT) — via /query/stream
  - Total latency — wall-clock time for a complete response
  - Loop count — how many retrieve→reason→critic cycles the agent used
  - Backend used — local or azure
  - Token cost — from the QueryResponse.cost_usd field

Outputs
-------
  - eval/results/benchmark_<timestamp>.csv   : per-run raw data
  - eval/results/benchmark_<timestamp>.json  : aggregate summary

Usage
-----
    # Full benchmark (needs live API):
    python eval/benchmark.py --api-url http://localhost:8000 --runs 3

    # Mock benchmark (no API needed):
    python eval/benchmark.py --mock

    # Compare local vs azure backends explicitly:
    python eval/benchmark.py --api-url http://localhost:8000 --runs 5 \\
        --routes SIMPLE COMPLEX
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Representative benchmark queries (one per difficulty level, per route)
# ---------------------------------------------------------------------------

BENCHMARK_QUERIES: list[dict] = [
    # SIMPLE — fast path, should be < 3s on local LLM
    {"route": "SIMPLE", "difficulty": "easy",
     "query": "What was total revenue in Q3 2024?"},
    {"route": "SIMPLE", "difficulty": "easy",
     "query": "What was the gross profit margin in Q3 2024?"},
    {"route": "SIMPLE", "difficulty": "easy",
     "query": "What was monthly recurring revenue at end of Q3 2024?"},

    # COMPLEX — multi-hop, should use LangGraph loops
    {"route": "COMPLEX", "difficulty": "medium",
     "query": "Compare gross margin year over year and explain what drove the change."},
    {"route": "COMPLEX", "difficulty": "medium",
     "query": "How did operating expenses change as a percentage of revenue in Q3 2024 vs Q3 2023?"},
    {"route": "COMPLEX", "difficulty": "hard",
     "query": "Calculate the revenue take rate for Q3 2024 and compare it to Q3 2023. What trend does this show?"},
    {"route": "COMPLEX", "difficulty": "hard",
     "query": "What does management say about international growth and how does GMV distribution across regions support this?"},

    # SENSITIVE — cloud should be disabled, should be fast (local only)
    {"route": "SENSITIVE", "difficulty": "easy",
     "query": "What are the salaries of Shopify executives listed in this filing?"},
    {"route": "SENSITIVE", "difficulty": "easy",
     "query": "Show me personal contact information for board members."},
]


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _now_ms() -> float:
    return time.perf_counter() * 1000


def _measure_blocking(api_url: str, query: str, pii_flagged: bool = False) -> dict:
    """Measure total latency for a blocking /query call."""
    import httpx

    t0 = _now_ms()
    resp = httpx.post(
        f"{api_url.rstrip('/')}/query",
        json={"query": query, "pii_flagged": pii_flagged},
        timeout=120.0,
    )
    total_ms = _now_ms() - t0
    resp.raise_for_status()
    data = resp.json()

    return {
        "total_ms":     round(total_ms, 1),
        "used_backend": data.get("used_backend", "unknown"),
        "loop_count":   data.get("loop_count", 0),
        "cost_usd":     data.get("cost_usd", 0.0),
        "confidence":   data.get("confidence", 0.0),
        "answer_len":   len(data.get("final_answer", "")),
        "num_citations":len(data.get("citations", [])),
        "error":        data.get("error", ""),
        "route_label":  data.get("route_badge", {}).get("label", "unknown"),
        "cloud_allowed":data.get("route_badge", {}).get("cloud_allowed", True),
    }


def _measure_ttft(api_url: str, query: str, pii_flagged: bool = False) -> float:
    """
    Measure time-to-first-token (ms) via the SSE streaming endpoint.
    Returns the time from request send to receipt of the first 'chunks' event.
    """
    import httpx

    t0 = _now_ms()
    ttft = -1.0

    with httpx.stream(
        "POST",
        f"{api_url.rstrip('/')}/query/stream",
        json={"query": query, "pii_flagged": pii_flagged},
        timeout=120.0,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line.startswith("data:"):
                raw = line[5:].strip()
                try:
                    event = json.loads(raw)
                    # "chunks" event means retrieval is done — first meaningful output
                    if event.get("type") == "chunks" and ttft < 0:
                        ttft = _now_ms() - t0
                    if event.get("type") == "done":
                        break
                except json.JSONDecodeError:
                    pass

    return round(ttft, 1) if ttft >= 0 else -1.0


def _mock_measure(query: str, route: str) -> dict:
    """Synthetic timing data for --mock mode."""
    base_ms = {"SIMPLE": 800, "COMPLEX": 3500, "SENSITIVE": 600}.get(route, 1000)
    jitter  = random.uniform(-100, 200)
    return {
        "total_ms":      round(base_ms + jitter, 1),
        "ttft_ms":       round((base_ms + jitter) * 0.4, 1),
        "used_backend":  "local" if route != "COMPLEX" else random.choice(["local", "azure"]),
        "loop_count":    1 if route == "SIMPLE" else random.randint(1, 3),
        "cost_usd":      0.0 if route in ("SIMPLE", "SENSITIVE") else round(random.uniform(0.001, 0.005), 6),
        "confidence":    round(random.uniform(0.80, 0.96), 3),
        "answer_len":    random.randint(150, 600),
        "num_citations": random.randint(1, 4),
        "route_label":   route,
        "cloud_allowed": route != "SENSITIVE",
        "error":         "",
    }


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    api_url: str,
    queries: list[dict],
    num_runs: int = 3,
    mock: bool = False,
    measure_ttft: bool = True,
    warmup_run: bool = True,
) -> list[dict]:
    """
    Run each query `num_runs` times and collect timing measurements.
    Returns a list of raw result dicts (one per query × run).
    """
    results: list[dict] = []
    total_calls = len(queries) * (num_runs + (1 if warmup_run else 0))
    call_num = 0

    for item in queries:
        query   = item["query"]
        route   = item["route"]
        diff    = item["difficulty"]
        pii     = route == "SENSITIVE"

        # Warmup run (discarded — warms up model KV cache)
        if warmup_run and not mock:
            logger.debug(f"Warmup: {query[:50]}…")
            try:
                _measure_blocking(api_url, query, pii_flagged=pii)
            except Exception as exc:
                logger.warning(f"Warmup failed: {exc}")

        for run_idx in range(num_runs):
            call_num += 1
            logger.info(
                f"[{call_num}/{total_calls - (len(queries) if warmup_run else 0)}] "
                f"Run {run_idx + 1}/{num_runs} | {route} | {query[:50]}…"
            )

            if mock:
                m = _mock_measure(query, route)
            else:
                try:
                    m = _measure_blocking(api_url, query, pii_flagged=pii)
                    if measure_ttft and not m.get("error"):
                        m["ttft_ms"] = _measure_ttft(api_url, query, pii_flagged=pii)
                    else:
                        m["ttft_ms"] = -1.0
                except Exception as exc:
                    logger.error(f"Measurement failed: {exc}")
                    m = {
                        "total_ms": -1.0, "ttft_ms": -1.0,
                        "used_backend": "error", "loop_count": 0,
                        "cost_usd": 0.0, "confidence": 0.0,
                        "answer_len": 0, "num_citations": 0,
                        "route_label": route, "cloud_allowed": True,
                        "error": str(exc),
                    }

            results.append({
                "query_id":   f"{route.lower()}_{diff}_{run_idx:02d}",
                "route":      route,
                "difficulty": diff,
                "run":        run_idx + 1,
                "query":      query[:80],   # truncated for readability
                **m,
            })

            # Brief pause between runs
            if not mock:
                time.sleep(0.5)

    return results


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------

def aggregate_results(results: list[dict]) -> dict:
    """
    Compute per-route aggregate statistics from raw benchmark results.
    """
    def _stats(values: list[float]) -> dict:
        if not values:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0}
        sorted_v = sorted(values)
        p95_idx  = max(0, int(len(sorted_v) * 0.95) - 1)
        return {
            "mean": round(sum(sorted_v) / len(sorted_v), 1),
            "min":  round(sorted_v[0], 1),
            "max":  round(sorted_v[-1], 1),
            "p95":  round(sorted_v[p95_idx], 1),
        }

    routes = sorted({r["route"] for r in results})
    agg: dict = {}

    for route in routes:
        group   = [r for r in results if r["route"] == route and not r.get("error")]
        total   = [r for r in results if r["route"] == route]
        latencies = [r["total_ms"] for r in group if r["total_ms"] > 0]
        ttfts     = [r["ttft_ms"]  for r in group if r.get("ttft_ms", -1) > 0]
        costs     = [r["cost_usd"] for r in group]
        loops     = [r["loop_count"] for r in group]

        local_count = sum(1 for r in group if r["used_backend"] == "local")
        azure_count = sum(1 for r in group if r["used_backend"] == "azure")

        agg[route] = {
            "n_total":         len(total),
            "n_success":       len(group),
            "n_error":         len(total) - len(group),
            "latency_ms":      _stats(latencies),
            "ttft_ms":         _stats(ttfts),
            "avg_loops":       round(sum(loops) / max(len(loops), 1), 2),
            "avg_cost_usd":    round(sum(costs) / max(len(costs), 1), 6),
            "total_cost_usd":  round(sum(costs), 6),
            "pct_local":       round(local_count / max(len(group), 1) * 100, 1),
            "pct_azure":       round(azure_count / max(len(group), 1) * 100, 1),
        }

    # Overall
    all_group = [r for r in results if not r.get("error")]
    agg["_overall"] = {
        "n_total":        len(results),
        "n_success":      len(all_group),
        "total_cost_usd": round(sum(r["cost_usd"] for r in all_group), 6),
        "pct_local":      round(
            sum(1 for r in all_group if r["used_backend"] == "local")
            / max(len(all_group), 1) * 100, 1
        ),
        "avg_latency_ms": round(
            sum(r["total_ms"] for r in all_group if r["total_ms"] > 0)
            / max(len([r for r in all_group if r["total_ms"] > 0]), 1), 1
        ),
    }

    return agg


def write_csv(results: list[dict], path: Path) -> None:
    if not results:
        return
    fieldnames = list(results[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def _print_benchmark_summary(agg: dict) -> None:
    print("\n" + "=" * 70)
    print("  PrivateDoc Benchmark Summary")
    print("=" * 70)
    for route, stats in agg.items():
        if route == "_overall":
            continue
        lat = stats["latency_ms"]
        print(f"\n  {route}")
        print(f"    Latency  mean={lat['mean']}ms  p95={lat['p95']}ms  "
              f"min={lat['min']}ms  max={lat['max']}ms")
        if stats["ttft_ms"]["mean"] > 0:
            ttft = stats["ttft_ms"]
            print(f"    TTFT     mean={ttft['mean']}ms  p95={ttft['p95']}ms")
        print(f"    Loops    avg={stats['avg_loops']}")
        print(f"    Cost     avg=${stats['avg_cost_usd']:.6f}  "
              f"total=${stats['total_cost_usd']:.4f}")
        print(f"    Backend  {stats['pct_local']}% local  "
              f"{stats['pct_azure']}% azure")

    ov = agg["_overall"]
    print(f"\n  Overall")
    print(f"    {ov['n_success']}/{ov['n_total']} successful")
    print(f"    Avg latency: {ov['avg_latency_ms']}ms")
    print(f"    Total cost:  ${ov['total_cost_usd']:.4f}")
    print(f"    Local path:  {ov['pct_local']}% of queries")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Latency and cost benchmark for PrivateDoc Intelligence Agent"
    )
    p.add_argument("--api-url", default="http://localhost:8000")
    p.add_argument("--runs", type=int, default=3,
                   help="Number of timed runs per query (default: 3)")
    p.add_argument("--routes", nargs="*",
                   choices=["SIMPLE", "COMPLEX", "SENSITIVE"],
                   default=None,
                   help="Restrict to specific route types")
    p.add_argument("--no-warmup", action="store_true",
                   help="Skip warmup run")
    p.add_argument("--no-ttft", action="store_true",
                   help="Skip TTFT measurement (saves one extra request per query)")
    p.add_argument("--output-dir", default=None,
                   help="Directory for output files (default: eval/results/)")
    p.add_argument("--mock", action="store_true",
                   help="Use synthetic data (no API needed)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    queries = BENCHMARK_QUERIES
    if args.routes:
        queries = [q for q in queries if q["route"] in args.routes]

    if not queries:
        logger.error("No queries match the specified route filters.")
        return 1

    logger.info(
        f"Benchmarking {len(queries)} queries × {args.runs} runs "
        f"({'mock' if args.mock else args.api_url})"
    )

    results = run_benchmark(
        api_url=args.api_url,
        queries=queries,
        num_runs=args.runs,
        mock=args.mock,
        measure_ttft=not args.no_ttft,
        warmup_run=not args.no_warmup,
    )

    agg = aggregate_results(results)
    _print_benchmark_summary(agg)

    # Write outputs
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir  = Path(args.output_dir) if args.output_dir else (
        Path(__file__).parent / "results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path  = output_dir / f"benchmark_{ts}.csv"
    json_path = output_dir / f"benchmark_{ts}.json"

    write_csv(results, csv_path)
    json_path.write_text(json.dumps({
        "timestamp": ts,
        "config": {
            "api_url": args.api_url,
            "runs": args.runs,
            "mock": args.mock,
        },
        "aggregate": agg,
    }, indent=2))

    logger.success(f"CSV  → {csv_path}")
    logger.success(f"JSON → {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())