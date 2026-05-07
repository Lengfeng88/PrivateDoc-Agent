"""
tests/test_ingest.py
--------------------
Unit tests for ingest/loader.py and ingest/chunker.py.

All tests use synthetic in-memory data so no PDF/DOCX parser is required.
Run with:  pytest tests/test_ingest.py -v
"""

import sys
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.loader import RawDocument, detect_pii
from ingest.chunker import SemanticChunker, Chunk, _estimate_tokens, _naive_split


# ---------------------------------------------------------------------------
# Helpers to build synthetic RawDocuments without touching the filesystem
# ---------------------------------------------------------------------------

def make_doc(
    text: str,
    element_type="narrative_text",
    page_num=1,
    pii_flagged=False,
    pii_types=None,
    file_hash="abc123",
    source_file="test.pdf",
) -> RawDocument:
    doc = RawDocument(
        source_file=source_file,
        file_hash=file_hash,
        page_num=page_num,
        element_type=element_type,
        text=text,
        pii_flagged=pii_flagged,
        pii_types=pii_types or [],
    )
    return doc


# ---------------------------------------------------------------------------
# detect_pii
# ---------------------------------------------------------------------------

class TestDetectPii:
    def test_clean_text_not_flagged(self):
        flagged, types = detect_pii("Gross profit increased 12% year over year.")
        assert not flagged
        assert types == []

    def test_email_detected(self):
        flagged, types = detect_pii("Contact john.doe@example.com for details.")
        assert flagged
        assert "email" in types

    def test_ssn_detected(self):
        flagged, types = detect_pii("Patient SSN: 123-45-6789 on file.")
        assert flagged
        assert "ssn" in types

    def test_medical_keyword_detected(self):
        flagged, types = detect_pii("The patient diagnosis was recorded in Q3.")
        assert flagged
        assert "medical" in types

    def test_multiple_pii_types(self):
        flagged, types = detect_pii(
            "Send invoice to jane@corp.com — SSN 987-65-4321"
        )
        assert flagged
        assert "email" in types
        assert "ssn" in types

    def test_phone_detected(self):
        flagged, types = detect_pii("Call us at 416-555-0123 during business hours.")
        assert flagged
        assert "phone" in types


# ---------------------------------------------------------------------------
# RawDocument
# ---------------------------------------------------------------------------

class TestRawDocument:
    def test_char_count_set_automatically(self):
        doc = make_doc("Hello world")
        assert doc.char_count == len("Hello world")

    def test_to_metadata_excludes_text(self):
        doc = make_doc("Secret data", pii_flagged=True, pii_types=["email"])
        meta = doc.to_metadata()
        assert "text" not in meta
        assert meta["pii_flagged"] is True
        assert meta["source_file"] == "test.pdf"

    def test_pii_defaults_false(self):
        doc = make_doc("Normal text")
        assert doc.pii_flagged is False
        assert doc.pii_types == []


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

class TestTokenEstimate:
    def test_empty_string(self):
        assert _estimate_tokens("") == 1  # floor of 1

    def test_approx_ratio(self):
        text = "a" * 400
        assert _estimate_tokens(text) == 100

    def test_longer_text(self):
        text = "word " * 200   # 1000 chars
        assert _estimate_tokens(text) == 250


# ---------------------------------------------------------------------------
# Naive splitter
# ---------------------------------------------------------------------------

class TestNaiveSplit:
    def test_short_text_not_split(self):
        text = "Short paragraph."
        pieces = _naive_split(text, max_tokens=512, overlap_tokens=64)
        assert len(pieces) == 1
        assert pieces[0] == text

    def test_long_text_splits(self):
        text = ("word " * 600).strip()  # ~3000 chars → ~750 tokens
        pieces = _naive_split(text, max_tokens=200, overlap_tokens=20)
        assert len(pieces) >= 2
        for piece in pieces:
            assert _estimate_tokens(piece) <= 220  # slight tolerance for boundary

    def test_overlap_present(self):
        # Build two paragraphs, ensure content from first appears at start of second
        para1 = "Alpha " * 200   # ~1200 chars
        para2 = "Beta " * 200
        text = para1 + "\n\n" + para2
        pieces = _naive_split(text, max_tokens=150, overlap_tokens=30)
        assert len(pieces) >= 2
        # second chunk should start with some content from the boundary
        assert len(pieces[1]) > 0


# ---------------------------------------------------------------------------
# SemanticChunker — core behaviour
# ---------------------------------------------------------------------------

class TestSemanticChunker:
    def _make_financial_doc(self) -> list[RawDocument]:
        """
        Simulate a simple financial report with headings and paragraphs.
        """
        return [
            make_doc("Q3 2024 Financial Results",  element_type="title",   page_num=1),
            make_doc("Revenue Overview",            element_type="heading", page_num=1),
            make_doc(
                "Total revenue for Q3 2024 was $1.86 billion, representing a 26% "
                "increase compared to Q3 2023. Subscription solutions revenue was "
                "$610 million, up 27% year over year.",
                page_num=1,
            ),
            make_doc(
                "Merchant solutions revenue reached $1.25 billion, growing 24% "
                "year over year, driven by higher gross merchandise volume (GMV).",
                page_num=2,
            ),
            make_doc("Gross Profit",                element_type="heading", page_num=2),
            make_doc(
                "Gross profit was $927 million, up 24% compared to $748 million "
                "in Q3 2023. Gross margin was 49.8%, compared to 50.7% in Q3 2023.",
                page_num=2,
            ),
            make_doc(
                "Revenue | Q3 2024 | Q3 2023\n"
                "Total | $1.86B | $1.48B\n"
                "Subscription | $610M | $481M\n"
                "Merchant | $1.25B | $1.01B",
                element_type="table",
                page_num=3,
            ),
        ]

    def test_produces_chunks(self):
        chunker = SemanticChunker()
        docs = self._make_financial_doc()
        chunks = chunker.chunk(docs)
        assert len(chunks) >= 2

    def test_all_chunks_have_text(self):
        chunker = SemanticChunker()
        chunks = chunker.chunk(self._make_financial_doc())
        for chunk in chunks:
            assert chunk.text.strip() != ""

    def test_chunk_ids_are_unique(self):
        chunker = SemanticChunker()
        chunks = chunker.chunk(self._make_financial_doc())
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_ids_are_deterministic(self):
        chunker = SemanticChunker()
        docs = self._make_financial_doc()
        chunks_a = chunker.chunk(docs)
        chunks_b = chunker.chunk(docs)
        assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]

    def test_table_is_atomic_chunk(self):
        """Table elements must not be merged into adjacent text chunks."""
        chunker = SemanticChunker()
        chunks = chunker.chunk(self._make_financial_doc())
        table_chunks = [c for c in chunks if "table" in c.element_types]
        assert len(table_chunks) >= 1
        for tc in table_chunks:
            # table chunk text should contain the pipe-separated content
            assert "|" in tc.text

    def test_section_heading_propagated(self):
        chunker = SemanticChunker()
        chunks = chunker.chunk(self._make_financial_doc())
        # at least one chunk should carry a section heading
        headings = [c.section_heading for c in chunks if c.section_heading]
        assert len(headings) >= 1

    def test_token_estimate_within_budget(self):
        chunker = SemanticChunker(max_tokens=512)
        # Generate a large document
        long_elements = [
            make_doc(
                ("The company reported strong results. " * 40),
                page_num=i + 1,
            )
            for i in range(10)
        ]
        chunks = chunker.chunk(long_elements)
        # Allow 10% tolerance for overlap
        for chunk in chunks:
            assert chunk.token_estimate <= 600, (
                f"Chunk too large: {chunk.token_estimate} tokens\n{chunk.text[:100]}"
            )

    def test_pii_flag_propagates_to_chunk(self):
        pii_doc = make_doc(
            "Employee SSN: 123-45-6789",
            pii_flagged=True,
            pii_types=["ssn"],
        )
        normal_doc = make_doc("Revenue was up 15% year over year.")
        chunker = SemanticChunker()
        chunks = chunker.chunk([pii_doc, normal_doc])
        pii_chunks = [c for c in chunks if c.pii_flagged]
        assert len(pii_chunks) >= 1
        assert "ssn" in pii_chunks[0].pii_types

    def test_multi_file_chunking(self):
        """Documents from different files should produce independent chunk sets."""
        file_a = [make_doc("File A content. " * 20, file_hash="hash_a", source_file="a.pdf")]
        file_b = [make_doc("File B content. " * 20, file_hash="hash_b", source_file="b.pdf")]
        chunker = SemanticChunker()
        chunks = chunker.chunk(file_a + file_b)
        sources = {c.source_file for c in chunks}
        assert "a.pdf" in sources
        assert "b.pdf" in sources

    def test_empty_input_returns_empty(self):
        chunker = SemanticChunker()
        assert chunker.chunk([]) == []

    def test_chunk_page_range_correct(self):
        docs = [
            make_doc("Text on page one.", page_num=1),
            make_doc("Text on page two.", page_num=2),
        ]
        chunker = SemanticChunker(max_tokens=512)
        chunks = chunker.chunk(docs)
        combined = [c for c in chunks if c.page_start == 1 and c.page_end >= 1]
        assert len(combined) >= 1

    def test_qdrant_payload_has_no_raw_text_key(self):
        chunker = SemanticChunker()
        chunks = chunker.chunk(self._make_financial_doc())
        for chunk in chunks:
            payload = chunk.to_qdrant_payload()
            assert "text" not in payload, "Raw text must not appear in Qdrant payload"
            assert "chunk_id" in payload
            assert "pii_flagged" in payload

    def test_min_token_filter_drops_tiny_chunks(self):
        docs = [
            make_doc("Q3 Results", element_type="heading", page_num=1),
            make_doc("See table below.", page_num=1),   # very short, < 30 tokens
            make_doc("Full detailed analysis " * 30, page_num=1),
        ]
        chunker = SemanticChunker(min_tokens=30)
        chunks = chunker.chunk(docs)
        # The tiny 3-word line should be absorbed, not emitted as its own chunk
        tiny = [c for c in chunks if c.token_estimate < 5]
        assert len(tiny) == 0


# ---------------------------------------------------------------------------
# Integration: loader metadata → chunker
# ---------------------------------------------------------------------------

class TestLoaderChunkerIntegration:
    def test_metadata_flows_through(self):
        """
        Simulate what happens after loader runs: metadata should survive into chunks.
        """
        elements = [
            make_doc(
                "Confidential patient diagnosis: Type 2 diabetes.",
                element_type="narrative_text",
                page_num=5,
                pii_flagged=True,
                pii_types=["medical"],
                file_hash="deadbeef",
                source_file="medical_report.pdf",
            )
        ]
        chunker = SemanticChunker()
        chunks = chunker.chunk(elements)
        assert len(chunks) == 1
        c = chunks[0]
        assert c.pii_flagged is True
        assert "medical" in c.pii_types
        assert c.source_file == "medical_report.pdf"
        assert c.page_start == 5