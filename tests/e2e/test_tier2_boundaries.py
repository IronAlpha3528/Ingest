"""Tier 2 — Boundary & Corner Cases: Opaque-Box E2E Tests.

Tests edge cases, constraints, and boundary conditions:
- Empty & whitespace-only text inputs
- Oversized attachments (>20 MB)
- Whitespace-only titles in classification
- Long titles (>80 chars) and filename slug capping (<=60 chars)
- Tags exceeding schema maximum (>6 tags)
- Multiple filename collisions (-2, -3 suffixing)
- Missing / null metadata for college documents
- Malformed LLM responses & vault atomicity guarantees
- Missing attachment paths
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
import unittest

from tests.e2e.harness import (
    E2ETestHarness,
    Category,
    Classification,
    IngestError,
    render_slug,
    render_filename,
)


class TestTier2Boundaries(unittest.TestCase):
    """Tier 2 Boundary & Corner Case Verification."""

    def setUp(self):
        self.harness = E2ETestHarness()

    def tearDown(self):
        self.harness.cleanup()

    def test_empty_text_input_rejection(self):
        """Boundary 1: Empty or whitespace-only text must be rejected with IngestError; vault untouched."""
        # 1. Empty string
        with self.assertRaises(IngestError) as ctx1:
            self.harness.send_text("")
        self.assertEqual(ctx1.exception.stage, "input")
        self.harness.assert_vault_clean()

        # 2. Whitespace-only string
        with self.assertRaises(IngestError) as ctx2:
            self.harness.send_text("   \n\t  \r  ")
        self.assertEqual(ctx2.exception.stage, "input")
        self.harness.assert_vault_clean()

    def test_oversized_input_rejection(self):
        """Boundary 2: Attachment > 20 MB must be rejected at intake without calling OCR or writing vault."""
        # Create a file just over 20MB: 20MB + 1KB = 20 * 1024 * 1024 + 1024 bytes
        large_size = 20 * 1024 * 1024 + 1024
        large_filename = "large_scan.pdf"
        local_path = self.harness.media_dir / large_filename
        
        # Write sparse/seek to avoid huge disk allocation
        with open(local_path, "wb") as f:
            f.seek(large_size - 1)
            f.write(b"\0")

        with self.assertRaises(IngestError) as ctx:
            self.harness.send_document(large_filename, local_path.read_bytes())
        
        self.assertEqual(ctx.exception.stage, "download")
        self.assertIn("exceeds 20MB limit", str(ctx.exception))
        # Verify OCR was never invoked for oversized file
        self.assertEqual(len(self.harness.ocr.calls), 0)
        # Verify vault is completely clean
        self.harness.assert_vault_clean()

    def test_whitespace_only_title_validation(self):
        """Boundary 3: LLM classification returning whitespace-only title raises validation error."""
        self.harness.llm.queue_json_response({
            "category": "reference",
            "title": "   \n\t   ",
            "summary": "Valid summary.",
            "tags": ["test"],
        })

        with self.assertRaises(IngestError) as ctx:
            self.harness.send_text("Sample input with whitespace title response")

        self.assertEqual(ctx.exception.stage, "classify")
        self.assertIn("title is empty", str(ctx.exception))
        self.harness.assert_vault_clean()

    def test_long_title_truncation_and_slug_capping(self):
        """Boundary 4: Title > 80 chars sanitized and filename slug strictly capped at 60 chars."""
        long_title = (
            "An Extremely Detailed And Comprehensive Analysis Of Distributed Consensus "
            "Protocols In High Throughput Byzantine Fault Tolerant Systems Across Continents"
        )
        self.assertGreater(len(long_title), 80)

        # Slug generator must cap at 60 chars
        slug = render_slug(long_title)
        self.assertLessEqual(len(slug), 60)
        self.assertFalse(slug.endswith("-"))

        # Verify filename formatting
        filename = render_filename(long_title, date(2026, 11, 3))
        self.assertTrue(filename.startswith("2026-11-03 - "))
        self.assertTrue(filename.endswith(".md"))
        slug_part = filename[len("2026-11-03 - "):-3]
        self.assertLessEqual(len(slug_part), 60)

        # Pipeline test with long title
        self.harness.llm.queue_json_response({
            "category": "reference",
            "title": long_title[:80],  # Valid max_length for model
            "summary": "Summary of long title research.",
            "tags": ["distributed-systems", "byzantine"],
        })
        proposal_rec = self.harness.send_text("Byzantine consensus protocol evaluation")
        inbox_file = proposal_rec.inbox_note_path.name
        self.assertTrue(inbox_file.endswith(".md"))
        base_slug = inbox_file[len("2026-11-03 - "):-3]
        self.assertLessEqual(len(base_slug), 60)

    def test_tags_exceeding_max_limit(self):
        """Boundary 5: Classification tags > 6 violates schema constraint and raises IngestError('classify')."""
        self.harness.llm.queue_json_response({
            "category": "reference",
            "title": "Topic With Excess Tags",
            "summary": "Summary for excess tags.",
            "tags": ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7", "tag8"],
        })
        with self.assertRaises(IngestError) as ctx:
            self.harness.send_text("Content that generates too many tags")
        self.assertEqual(ctx.exception.stage, "classify")
        self.harness.assert_vault_clean()

    def test_filename_collision_handling(self):
        """Boundary 6: Multiple captures with identical titles on same day produce -2, -3 suffixes without overwriting."""
        title = "Hostel Fee Receipt"
        day = date(2026, 11, 3)

        # Inbound capture 1
        self.harness.llm.queue_json_response({
            "category": "reference",
            "title": title,
            "summary": "First receipt capture.",
            "tags": ["receipt"],
        })
        prop1 = self.harness.send_text("Receipt 1")
        self.assertEqual(prop1.inbox_note_path.name, "2026-11-03 - Hostel-Fee-Receipt.md")

        # Inbound capture 2 (same title)
        self.harness.llm.queue_json_response({
            "category": "reference",
            "title": title,
            "summary": "Second receipt capture with different content.",
            "tags": ["receipt"],
        })
        prop2 = self.harness.send_text("Receipt 2")
        self.assertEqual(prop2.inbox_note_path.name, "2026-11-03 - Hostel-Fee-Receipt-2.md")

        # Inbound capture 3 (same title)
        self.harness.llm.queue_json_response({
            "category": "reference",
            "title": title,
            "summary": "Third receipt capture.",
            "tags": ["receipt"],
        })
        prop3 = self.harness.send_text("Receipt 3")
        self.assertEqual(prop3.inbox_note_path.name, "2026-11-03 - Hostel-Fee-Receipt-3.md")

        # Verify all 3 files exist simultaneously in inbox
        inbox_files = sorted(self.harness.list_folder("inbox"))
        self.assertEqual(inbox_files, [
            "2026-11-03 - Hostel-Fee-Receipt-2.md",
            "2026-11-03 - Hostel-Fee-Receipt-3.md",
            "2026-11-03 - Hostel-Fee-Receipt.md",
        ])

        # Verify none overwrote another
        _, body1 = self.harness.read_note("inbox", "2026-11-03 - Hostel-Fee-Receipt.md")
        _, body2 = self.harness.read_note("inbox", "2026-11-03 - Hostel-Fee-Receipt-2.md")
        _, body3 = self.harness.read_note("inbox", "2026-11-03 - Hostel-Fee-Receipt-3.md")
        self.assertIn("First receipt capture", body1)
        self.assertIn("Second receipt capture", body2)
        self.assertIn("Third receipt capture", body3)

    def test_missing_optional_metadata(self):
        """Boundary 7: College document with null/missing optional fields renders without error."""
        self.harness.llm.queue_json_response({
            "category": "college-documents",
            "title": "Minimal College Document",
            "summary": "A certificate with no date or reference number visible.",
            "tags": ["college"],
            "document_type": None,
            "issuer": None,
            "issued_date": None,
            "reference_ids": [],
        })
        proposal_rec = self.harness.send_text("A blurred certificate without dates")
        filing_rec = self.harness.tap_approve(proposal_rec)

        fm, body = self.harness.read_note("college-documents", filing_rec.destination_path.name)
        self.assertEqual(fm["category"], "college-documents")
        self.assertIsNone(fm["document_type"])
        self.assertIsNone(fm["issuer"])
        self.assertIsNone(fm["issued_date"])
        self.assertEqual(fm["reference_ids"], [])
        self.assertIn("General academic documentation", body)

    def test_malformed_llm_json_response_failure(self):
        """Boundary 8: Broken LLM output raises IngestError('classify'); vault atomicity preserved (zero partial notes)."""
        # Queue invalid response missing required category key
        self.harness.llm.queue_json_response({
            "invalid_key": "some value",
        })

        with self.assertRaises(IngestError) as ctx:
            self.harness.send_text("Document that causes LLM to hallucinate invalid JSON")

        self.assertEqual(ctx.exception.stage, "classify")
        # Assert vault has zero notes written (strict atomicity)
        self.harness.assert_vault_clean()

    def test_nonexistent_attachment_path(self):
        """Boundary 9: Local path not found raises IngestError('download') and writes zero notes."""
        non_existent_file = self.harness.media_dir / "does_not_exist.pdf"
        from tests.e2e.harness import Capture, CaptureKind, sync_run
        capture = Capture(
            chat_id=self.harness.chat_id,
            message_id=self.harness._next_message_id(),
            kind=CaptureKind.DOCUMENT,
            filename="does_not_exist.pdf",
            local_path=non_existent_file,
        )
        with self.assertRaises(IngestError) as ctx:
            sync_run(self.harness._execute_pipeline(capture))

        self.assertEqual(ctx.exception.stage, "download")
        self.harness.assert_vault_clean()
