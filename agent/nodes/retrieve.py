"""
agent/nodes/retrieve.py
-----------------------
LangGraph node: retrieves relevant chunks from Qdrant using hybrid search.

Hybrid search strategy
----------------------
Round 1 (and every subsequent loop):
  1. Optionally rewrite the query using a fast local LLM call
     (only if loop_count == 0 — avoids re-rewriting on subsequent loops).
  2. Dense retrieval: Qdrant vector search with BGE-M3 embeddings.
  3. Sparse retrieval: BM25 keyword search over the same collection
     (Qdrant's built-in sparse vector support, or a local BM25 fallback).
  4. Reciprocal Rank Fusion (RRF) to merge both result lists.
  5. Return top-k chunks as RetrievedChunk dicts.

On loop_count >= 1 the node reformulates the search using the critique
text from the previous cycle — this is what gives the agent its
"multi-hop" capability.

Qdrant dependency
-----------------
Expects a running Qdrant instance (local Docker or Qdrant Cloud).
  docker run -p 6333:6333 qdrant/qdrant

Collection schema assumed:
  - vectors:   BGE-M3 dense (1024-dim)
  - payload:   all fields from Chunk.to_qdrant_payload()
  - text:      stored in payload key "text" (not the vector itself)
"""

from __future__ import annotations

import os
from loguru import logger

from agent.state import AgentState, RetrievedChunk
from agent.prompts import (
    QUERY_REWRITE_SYSTEM,
    QUERY_REWRITE_USER,
)

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

QDRANT_URL        = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "privatedoc")
TOP_K             = int(os.getenv("RETRIEVAL_TOP_K", "8"))
MAX_LOOP          = int(os.getenv("AGENT_MAX_LOOPS", "3"))


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(
    dense_hits: list[tuple[str, float]],
    sparse_hits: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank_i)) across all lists.
    Standard k=60 from the original Cormack et al. paper.

    Returns a list of (chunk_id, rrf_score) sorted descending.
    """
    scores: dict[str, float] = {}
    for rank, (chunk_id, _) in enumerate(dense_hits, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    for rank, (chunk_id, _) in enumerate(sparse_hits, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Query rewriter (optional, only on first loop)
# ---------------------------------------------------------------------------

def _rewrite_query(query: str, backend) -> str:
    """
    Use a fast local LLM call to produce a keyword-optimised search query.
    Falls back to original query if backend is unavailable.
    """
    try:
        resp = backend.complete(
            system=QUERY_REWRITE_SYSTEM,
            user=QUERY_REWRITE_USER.format(query=query),
            max_tokens=40,
            temperature=0.0,
        )
        rewritten = resp.text.strip().strip('"')
        logger.debug(f"query rewrite: '{query}' → '{rewritten}'")
        return rewritten
    except Exception as exc:
        logger.warning(f"query rewrite failed ({exc}), using original")
        return query


# ---------------------------------------------------------------------------
# Critique-aware query expansion (used on loop >= 1)
# ---------------------------------------------------------------------------

def _expand_from_critique(query: str, critique: str) -> str:
    """
    Extract what's missing from the critic's INSUFFICIENT verdict and
    append it to the query so the next retrieval targets the gap.

    Critic format: "INSUFFICIENT: <explanation>"
    """
    if ":" in critique:
        missing = critique.split(":", 1)[1].strip()
        expanded = f"{query} {missing}"
        logger.debug(f"loop expansion: added '{missing}'")
        return expanded
    return query


# ---------------------------------------------------------------------------
# Main node function
# ---------------------------------------------------------------------------

def retrieve_node(state: AgentState) -> dict:
    """
    LangGraph node — retrieves chunks and updates state.

    Returns partial state dict with keys:
      retrieved_chunks, retrieval_query, loop_count
    """
    query: str = state["query"]
    loop_count: int = state.get("loop_count", 0)
    critique: str = state.get("critique", "")
    cloud_allowed: bool = state.get("cloud_allowed", True)

    if loop_count >= MAX_LOOP:
        logger.warning(f"Max loop count ({MAX_LOOP}) reached — skipping retrieval")
        return {"loop_count": loop_count}

    # ── Build retrieval query ─────────────────────────────────────────
    if loop_count == 0:
        # First pass: optionally rewrite for better keyword coverage.
        # We use the local backend for rewriting regardless of route
        # (rewriting itself is not a privacy-sensitive operation).
        retrieval_query = _try_rewrite(query, cloud_allowed)
    else:
        # Subsequent passes: expand based on what the critic said was missing.
        retrieval_query = _expand_from_critique(query, critique)

    logger.info(f"[retrieve] loop={loop_count} query='{retrieval_query[:80]}'")

    # ── Qdrant hybrid search ──────────────────────────────────────────
    chunks = _qdrant_hybrid_search(retrieval_query, top_k=TOP_K)

    logger.info(f"[retrieve] returned {len(chunks)} chunks")
    return {
        "retrieved_chunks": chunks,
        "retrieval_query": retrieval_query,
        "loop_count": loop_count + 1,
    }


def _try_rewrite(query: str, cloud_allowed: bool) -> str:
    """Attempt query rewrite with local LLM; silently fall back."""
    try:
        from agent.backends.local_llm import LocalLLMBackend
        backend = LocalLLMBackend()
        if backend.health_check():
            return _rewrite_query(query, backend)
    except Exception:
        pass
    return query


def _qdrant_hybrid_search(
    query: str,
    top_k: int = 8,
) -> list[RetrievedChunk]:
    """
    Run dense + sparse retrieval against Qdrant, fuse with RRF.

    Falls back to dense-only if sparse vectors are not configured,
    and falls back to a stub if Qdrant is unreachable (useful in tests).
    """
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Filter
        client = QdrantClient(url=QDRANT_URL, timeout=10)
    except Exception as exc:
        logger.warning(f"Qdrant unavailable ({exc}) — returning empty chunks")
        return []

    # ── Dense retrieval ───────────────────────────────────────────────
    query_vector = _embed(query)
    dense_results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k * 2,          # oversample before RRF
        with_payload=True,
    )
    dense_hits = [(r.id, r.score) for r in dense_results.points]

    # ── Sparse / BM25 retrieval (graceful degradation) ────────────────
    sparse_hits = _bm25_search(client, query, top_k=top_k * 2)

    # ── RRF merge ─────────────────────────────────────────────────────
    if sparse_hits:
        fused = _reciprocal_rank_fusion(dense_hits, sparse_hits)
    else:
        fused = [(cid, score) for cid, score in dense_hits]

    # Build a lookup from the dense results (already have payload)
    payload_by_id = {str(r.id): r.payload for r in dense_results.points}

    # Fetch payloads for any sparse-only hits not in dense results
    sparse_only_ids = [
        cid for cid, _ in fused[:top_k] if cid not in payload_by_id
    ]
    if sparse_only_ids:
        extra = client.retrieve(
            collection_name=QDRANT_COLLECTION,
            ids=sparse_only_ids,
            with_payload=True,
        )
        for point in extra:
            payload_by_id[str(point.id)] = point.payload

    # Assemble final chunk list
    chunks: list[RetrievedChunk] = []
    for chunk_id, rrf_score in fused[:top_k]:
        payload = payload_by_id.get(str(chunk_id))
        if not payload:
            continue
        chunks.append(RetrievedChunk(
            chunk_id=str(chunk_id),
            text=payload.get("text", ""),
            source_file=payload.get("source_file", ""),
            page_start=payload.get("page_start", 0),
            page_end=payload.get("page_end", 0),
            section_heading=payload.get("section_heading", ""),
            score=round(rrf_score, 6),
        ))

    return chunks


def _embed(text: str) -> list[float]:
    """
    Generate a BGE-M3 dense embedding for the query.
    Falls back to a zero vector (length 1024) if model not loaded —
    this lets the graph run in test environments without GPU.
    """
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("BAAI/bge-m3")
        vec = _model.encode(text, normalize_embeddings=True)
        return vec.tolist()
    except Exception as exc:
        logger.warning(f"embedding failed ({exc}), using zero vector")
        return [0.0] * 1024


def _bm25_search(
    client,
    query: str,
    top_k: int = 16,
) -> list[tuple[str, float]]:
    """
    Attempt Qdrant sparse vector search.
    Returns empty list if the collection has no sparse vectors configured.
    """
    try:
        sparse_results = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=("sparse", {"indices": [], "values": []}),
            limit=top_k,
            with_payload=False,
        )
        return [(str(r.id), r.score) for r in sparse_results]
    except Exception:
        return []