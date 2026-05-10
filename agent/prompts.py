from __future__ import annotations
from agent.state import RetrievedChunk

# ── Query rewrite ─────────────────────────────────────────────────────────────
QUERY_REWRITE_SYSTEM = """You are a search query optimizer.
Given an original question and a critic's feedback, rewrite the query to
retrieve more relevant context. Output only the rewritten query, nothing else."""

# ── Reason ────────────────────────────────────────────────────────────────────
REASON_SYSTEM = """You are a precise financial analyst assistant.
Answer the user's question based ONLY on the provided context chunks.
If the context is insufficient, begin your answer with INSUFFICIENT_CONTEXT:
Always cite chunk IDs in square brackets e.g. [chunk-id] after each claim.
End your answer with a confidence score on its own line: CONFIDENCE: 0.85"""

REASON_USER = """Context:
{context_block}

Question: {query}

Answer:"""

# ── Critic ────────────────────────────────────────────────────────────────────
CRITIC_SYSTEM = """You are a grounding critic.
Given a question, context chunks, and a draft answer, output exactly one of:
  GROUNDED
  INSUFFICIENT: <reason>
Output nothing else."""

CRITIC_USER = """Context:
{context_block}

Question: {query}

Draft answer: {answer_draft}

Verdict:"""

# ── Report ────────────────────────────────────────────────────────────────────
REPORT_SYSTEM = """You are a report formatter.
Given a verified answer and its source citations, produce a clean final answer.
Keep all chunk ID references in square brackets."""

REPORT_USER = """Verified answer: {answer_draft}

Produce the final answer:"""


def format_context_block(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a numbered context block for the LLM prompt."""
    if not chunks:
        return ""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = (
            f"[{i}] {chunk['source_file']} "
            f"p.{chunk['page_start']}–{chunk['page_end']} "
            f"| {chunk['section_heading']} "
            f"(score {chunk['score']:.3f})"
            f" chunk_id={chunk['chunk_id']}"
        )
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(parts)

QUERY_REWRITE_USER = """Original question: {query}

Critic feedback: {critique}

Rewritten search query:"""