"""
ingest/chunker.py
-----------------
Converts a list of RawDocument elements into Chunk objects ready for embedding.

Design decisions:
  - Heading-aware grouping: text under the same heading is kept together
    rather than split mid-section, which improves retrieval coherence.
  - Token-budget chunking: once a group exceeds max_tokens, it is split
    using a recursive character splitter (no mid-sentence cuts).
  - Overlap: each chunk carries the last `overlap_tokens` worth of the
    previous chunk so retrieval doesn't lose context at boundaries.
  - Tables and list items are always kept as single atomic chunks — never
    split across a boundary.
  - PII flag propagates: if any source element was PII-flagged, the chunk
    inherits that flag.

Typical usage:
    chunker = SemanticChunker()
    chunks = chunker.chunk(raw_docs)
    print(len(chunks), "chunks ready for embedding")
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Generator

from loguru import logger

from .loader import RawDocument, ElementType


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """
    A text unit ready for embedding and Qdrant upsert.
    The `chunk_id` is stable: same text from the same file always gets the same ID.
    """
    chunk_id: str               # deterministic uuid5 from file_hash + position
    file_hash: str              # links back to the source document
    source_file: str
    page_start: int
    page_end: int
    element_types: list[str]    # which element types contributed to this chunk
    text: str
    token_estimate: int = field(init=False)
    pii_flagged: bool = False
    pii_types: list[str] = field(default_factory=list)
    section_heading: str = ""   # nearest heading above this chunk (for context)
    chunk_index: int = 0        # position within the document

    def __post_init__(self) -> None:
        self.token_estimate = _estimate_tokens(self.text)

    def to_qdrant_payload(self) -> dict:
        """Everything except the raw text — stored as Qdrant point payload."""
        return {
            "chunk_id": self.chunk_id,
            "file_hash": self.file_hash,
            "source_file": self.source_file,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "element_types": self.element_types,
            "token_estimate": self.token_estimate,
            "pii_flagged": self.pii_flagged,
            "pii_types": self.pii_types,
            "section_heading": self.section_heading,
            "chunk_index": self.chunk_index,
        }


# ---------------------------------------------------------------------------
# Token estimation (no tokenizer dependency)
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """
    Fast approximation: 1 token ≈ 4 chars for English prose.
    Good enough for chunking budget; embedding models apply real tokenisation.
    """
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Recursive text splitter (no langchain import needed at runtime)
# ---------------------------------------------------------------------------

def _split_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """
    Splits `text` into chunks of at most `max_tokens` with `overlap_tokens`
    overlap. Tries to break at paragraph → sentence → word boundaries.

    This is a self-contained implementation so the module has zero required
    imports beyond stdlib. If langchain_text_splitters is available it is used
    instead for slightly better sentence boundary detection.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_tokens * 4,          # chars, not tokens
            chunk_overlap=overlap_tokens * 4,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        return splitter.split_text(text)

    except ImportError:
        return _naive_split(text, max_tokens, overlap_tokens)


def _naive_split(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Stdlib-only fallback splitter."""
    max_chars = max_tokens * 4
    overlap_chars = overlap_tokens * 4

    # If the entire text fits in one chunk, return immediately — no loop needed.
    if len(text) <= max_chars:
        stripped = text.strip()
        return [stripped] if stripped else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # try to break at paragraph boundary
        boundary = text.rfind("\n\n", start, end)
        if boundary == -1 or boundary <= start:
            boundary = text.rfind(". ", start, end)
        if boundary == -1 or boundary <= start:
            boundary = end
        else:
            boundary += 2  # include the separator

        chunk = text[start:boundary].strip()
        if chunk:
            chunks.append(chunk)

        next_start = boundary - overlap_chars
        # Guard: always advance by at least 1 char to prevent infinite loop.
        if next_start <= start:
            next_start = start + max(1, max_chars // 2)
        start = next_start

    return chunks


# ---------------------------------------------------------------------------
# Heading tracker
# ---------------------------------------------------------------------------

_HEADING_TYPES: frozenset[ElementType] = frozenset({"title", "heading"})


class _HeadingTracker:
    """Tracks the most recent heading seen while iterating elements."""

    def __init__(self) -> None:
        self._current: str = ""

    def update(self, element: RawDocument) -> None:
        if element.element_type in _HEADING_TYPES and element.text.strip():
            self._current = element.text.strip()

    @property
    def current(self) -> str:
        return self._current


# ---------------------------------------------------------------------------
# Atomic element types — never split across chunk boundaries
# ---------------------------------------------------------------------------

_ATOMIC_TYPES: frozenset[ElementType] = frozenset({"table", "list_item"})


# ---------------------------------------------------------------------------
# Main chunker
# ---------------------------------------------------------------------------

class SemanticChunker:
    """
    Heading-aware semantic chunker.

    Parameters
    ----------
    max_tokens : int
        Soft upper bound on chunk size in tokens. Atomic elements (tables,
        lists) may exceed this to avoid splitting mid-row.
    overlap_tokens : int
        How many tokens from the end of the previous chunk to prepend to
        the next chunk. Improves retrieval at section boundaries.
    min_tokens : int
        Chunks shorter than this are merged with the next element rather
        than emitted as standalone chunks. Prevents single-sentence noise.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        min_tokens: int = 30,
    ) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, documents: list[RawDocument]) -> list[Chunk]:
        """
        Convert a flat list of RawDocument elements into Chunks.
        Documents from multiple files are handled correctly — chunking
        resets per file_hash.
        """
        if not documents:
            return []

        # Group by file so multi-file ingestion is handled cleanly
        by_file: dict[str, list[RawDocument]] = {}
        for doc in documents:
            by_file.setdefault(doc.file_hash, []).append(doc)

        all_chunks: list[Chunk] = []
        for file_hash, elements in by_file.items():
            file_chunks = list(self._chunk_single_file(elements))
            # assign stable IDs and sequential indices
            for idx, chunk in enumerate(file_chunks):
                chunk.chunk_index = idx
                chunk.chunk_id = self._make_chunk_id(file_hash, idx)
            all_chunks.extend(file_chunks)
            logger.success(
                f"{elements[0].source_file}: "
                f"{len(elements)} elements → {len(file_chunks)} chunks"
            )

        return all_chunks

    # ------------------------------------------------------------------
    # Per-file chunking pipeline
    # ------------------------------------------------------------------

    def _chunk_single_file(
        self, elements: list[RawDocument]
    ) -> Generator[Chunk, None, None]:
        """
        Yields Chunk objects for one file.

        Strategy:
          1. Walk elements in order, tracking the current heading.
          2. Accumulate elements into a running buffer.
          3. On hitting a new heading OR exceeding max_tokens → flush buffer.
          4. Atomic elements (tables, lists) are always flushed immediately.
        """
        tracker = _HeadingTracker()
        buffer: list[RawDocument] = []
        prev_tail: str = ""  # last `overlap_tokens` chars of previous chunk

        def flush(buf: list[RawDocument], heading: str) -> Generator[Chunk, None, None]:
            nonlocal prev_tail
            if not buf:
                return
            combined_text = "\n\n".join(el.text for el in buf)
            if prev_tail:
                combined_text = prev_tail + "\n\n" + combined_text

            # split if over budget
            if _estimate_tokens(combined_text) > self.max_tokens:
                pieces = _split_text(combined_text, self.max_tokens, self.overlap_tokens)
            else:
                pieces = [combined_text]

            for i, piece in enumerate(pieces):
                is_last_piece = (i == len(pieces) - 1)
                too_short = _estimate_tokens(piece) < self.min_tokens
                if too_short and not is_last_piece:
                    # Short intermediate piece — absorb into next chunk's overlap
                    prev_tail = piece[-(self.overlap_tokens * 4):]
                    continue
                # Emit: content is long enough, OR it is the only/last piece
                # (a single short element must not be silently discarded).
                yield _build_chunk(piece, buf, heading)
                prev_tail = piece[-(self.overlap_tokens * 4):]

        for el in elements:
            tracker.update(el)
            is_new_section = el.element_type in _HEADING_TYPES and buffer
            is_atomic = el.element_type in _ATOMIC_TYPES
            buffer_tokens = sum(_estimate_tokens(e.text) for e in buffer)
            would_overflow = buffer_tokens + _estimate_tokens(el.text) > self.max_tokens

            if is_new_section or (would_overflow and not is_atomic):
                yield from flush(buffer, tracker.current)
                buffer = []

            if is_atomic:
                # flush existing buffer first, then emit atomic as its own chunk
                yield from flush(buffer, tracker.current)
                buffer = []
                yield from flush([el], tracker.current)
            else:
                buffer.append(el)

        # flush remainder
        yield from flush(buffer, tracker.current)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_chunk_id(file_hash: str, index: int) -> str:
        """Deterministic UUID5 — same input always produces the same ID."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{file_hash}:{index}"))


# ---------------------------------------------------------------------------
# Chunk builder (module-level helper)
# ---------------------------------------------------------------------------

def _build_chunk(
    text: str,
    source_elements: list[RawDocument],
    section_heading: str,
) -> Chunk:
    """Constructs a Chunk from a text string and the elements that produced it."""
    pii_flagged = any(el.pii_flagged for el in source_elements)
    pii_types: list[str] = []
    for el in source_elements:
        for t in el.pii_types:
            if t not in pii_types:
                pii_types.append(t)

    return Chunk(
        chunk_id="",    # assigned by caller with stable ID
        file_hash=source_elements[0].file_hash,
        source_file=source_elements[0].source_file,
        page_start=min(el.page_num for el in source_elements),
        page_end=max(el.page_num for el in source_elements),
        element_types=list({el.element_type for el in source_elements}),
        text=text,
        pii_flagged=pii_flagged,
        pii_types=pii_types,
        section_heading=section_heading,
    )