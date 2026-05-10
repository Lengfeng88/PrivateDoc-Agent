"""
api/cost_logger.py
------------------
Thread-safe, in-memory metrics store for the /metrics endpoint.

Design decisions
----------------
- No external dependency (no Redis, no Prometheus): this keeps the
  local dev loop zero-config.  Azure Monitor / LangSmith are the
  production observability layer; this is a lightweight dashboard source.
- Uses threading.Lock so concurrent FastAPI requests don't corrupt counts.
- All data resets on process restart — intentional. Persistent metrics
  belong in Azure Monitor, not in the API process.
- The logger also writes a JSONL append file (logs/queries.jsonl) for
  offline analysis and the eval harness.

Typical usage
-------------
    from api.cost_logger import metrics

    metrics.record(
        route="COMPLEX",
        backend="local",
        cost_usd=0.0,
        duration_seconds=2.31,
        loop_count=2,
    )
    snapshot = metrics.snapshot()   # MetricsResponse-compatible dict
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Per-request record (written to JSONL)
# ---------------------------------------------------------------------------

@dataclass
class QueryRecord:
    timestamp_utc: float
    route: str
    backend: str
    cost_usd: float
    duration_seconds: float
    loop_count: int
    num_citations: int
    query_len: int          # char count — no raw query text stored

    def to_dict(self) -> dict:
        return {
            "timestamp_utc":    self.timestamp_utc,
            "route":            self.route,
            "backend":          self.backend,
            "cost_usd":         round(self.cost_usd, 8),
            "duration_seconds": round(self.duration_seconds, 3),
            "loop_count":       self.loop_count,
            "num_citations":    self.num_citations,
            "query_len":        self.query_len,
        }


# ---------------------------------------------------------------------------
# Metrics accumulator
# ---------------------------------------------------------------------------

class MetricsStore:
    """
    Thread-safe in-memory metrics accumulator.

    All public methods acquire the lock before mutating state.
    The `snapshot()` method returns a plain dict so the /metrics endpoint
    can serialise it without holding the lock.
    """

    def __init__(self, log_path: Optional[str | Path] = "logs/queries.jsonl") -> None:
        self._lock = threading.Lock()
        self._log_path = Path(log_path) if log_path else None

        # Counters
        self._total_queries: int = 0
        self._total_cost_usd: float = 0.0
        self._total_duration: float = 0.0
        self._total_loops: int = 0

        # Distribution counts
        self._route_counts:   dict[str, int] = {"SIMPLE": 0, "COMPLEX": 0, "SENSITIVE": 0}
        self._backend_counts: dict[str, int] = {"local": 0, "azure": 0, "none": 0, "blocked": 0}

        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def record(
        self,
        route: str,
        backend: str,
        cost_usd: float,
        duration_seconds: float,
        loop_count: int,
        num_citations: int = 0,
        query_len: int = 0,
    ) -> None:
        """Thread-safe: increment all counters and append to JSONL."""
        record = QueryRecord(
            timestamp_utc=time.time(),
            route=route,
            backend=backend,
            cost_usd=cost_usd,
            duration_seconds=duration_seconds,
            loop_count=loop_count,
            num_citations=num_citations,
            query_len=query_len,
        )
        with self._lock:
            self._total_queries += 1
            self._total_cost_usd += cost_usd
            self._total_duration += duration_seconds
            self._total_loops += loop_count

            # Route counter — default to "UNKNOWN" bucket if unexpected value
            self._route_counts[route] = self._route_counts.get(route, 0) + 1

            # Backend counter
            self._backend_counts[backend] = self._backend_counts.get(backend, 0) + 1

        self._append_jsonl(record)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Return a consistent snapshot safe for serialisation."""
        with self._lock:
            n = self._total_queries
            return {
                "total_queries":        n,
                "total_cost_usd":       round(self._total_cost_usd, 6),
                "route_counts":         dict(self._route_counts),
                "backend_counts":       dict(self._backend_counts),
                "avg_duration_seconds": round(self._total_duration / n, 3) if n else 0.0,
                "avg_loops":            round(self._total_loops / n, 2) if n else 0.0,
            }

    def reset(self) -> None:
        """Reset all counters — used in tests only."""
        with self._lock:
            self._total_queries = 0
            self._total_cost_usd = 0.0
            self._total_duration = 0.0
            self._total_loops = 0
            self._route_counts   = {"SIMPLE": 0, "COMPLEX": 0, "SENSITIVE": 0}
            self._backend_counts = {"local": 0, "azure": 0, "none": 0, "blocked": 0}

    # ------------------------------------------------------------------
    # JSONL logging
    # ------------------------------------------------------------------

    def _append_jsonl(self, record: QueryRecord) -> None:
        if not self._log_path:
            return
        try:
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record.to_dict()) + "\n")
        except OSError as exc:
            logger.warning(f"cost_logger: JSONL write failed: {exc}")


# ---------------------------------------------------------------------------
# Module-level singleton — imported by main.py and tests
# ---------------------------------------------------------------------------

metrics = MetricsStore(log_path=None)   # log_path set at startup via init_metrics()


def init_metrics(log_path: str | Path | None = "logs/queries.jsonl") -> None:
    """
    Called once at FastAPI startup to configure the log path.
    Replaces the module-level singleton so all importers see the update.
    """
    global metrics
    metrics = MetricsStore(log_path=log_path)
    logger.info(f"cost_logger: initialised (log={log_path})")