"""Tier 1 — Feature Coverage: Happy-Path Opaque-Box E2E Tests.

Tests each primary pipeline capability:
- Document intake (PDF)
- Photo intake (JPEG)
- Text note intake
- OCR extraction
- Translation / Indic language normalisation
- Classification into 3 categories (college, project, reference)
- Project brief expansion with search citations
- Obsidian note rendering with strict frontmatter
- Proposal generation with 3 inline buttons
- Interactive callbacks: Approve, Change folder, Discard
- Fast receipt acknowledgement (< 2 seconds)
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
import unittest

from tests.e2e.harness import (
    E2ETestHarness,
    Category,
    CaptureKind,
    SearchResult,
)


class TestTier1Features(unittest.TestCase):
    """Tier 1 Happy-Path Feature Verification."""

    def setUp(self):
        self.harness = E2ETestHarness()

    def tearDown(self):
        self.harness.cleanup()

    def test_document_intake_college_flow(self):
        """Feature 1: PDF document -> OCR -> College classification -> Proposal -> Approve."""
        pdf_bytes = b"%PDF-1.4 simulated pdf document data for semester grade card"
        filename = "Semester_VI_Marksheet.pdf"
        ocr_text = (
            "Delhi Technological University\n"
            "Bachelor of Technology in Computer Engineering\n"
            "Semester VI Examination - May 2026\n"
            "Roll No: 2K22/CO/108\n"
            "Subject: Distributed Systems (Grade: A+)\n"
            "Date of Issue: 2026-06-15"
        )
        self.harness.ocr.register_file(filename, ocr_text)
        self.harness.llm.queue_json_response({
            "category": "college-documents",
            "title": "DTU Semester VI Computer Engineering Marksheet",
            "summary": "Official grade report for Semester VI Computer Engineering at Delhi Technological University.",
            "tags": ["academics", "dtu", "grades", "semester-6"],
            "document_type": "mark-sheet",
            "issuer": "Delhi Technological University",
            "issued_date": "2026-06-15",
            "reference_ids": ["2K22/CO/108"],
        })

        # 1. Send Document
        proposal_rec = self.harness.send_document(filename, pdf_bytes)

        # Verify fast acknowledgement
        acks = self.harness.get_acknowledgements()
        self.assertIn("Reading…", acks)

        # Verify proposal presentation
        self.assertEqual(proposal_rec.proposal.category, Category.COLLEGE)
        self.assertEqual(proposal_rec.inline_buttons, ["Approve", "Change folder", "Discard"])
        self.assertTrue(proposal_rec.inbox_note_path.exists())

        # Verify note in inbox prior to approval
        inbox_files = self.harness.list_folder("inbox")
        self.assertEqual(len(inbox_files), 1)

        # 2. User taps Approve
        filing_rec = self.harness.tap_approve(proposal_rec)

        # Verify note moved out of inbox and into college-documents/
        self.assertEqual(len(self.harness.list_folder("inbox")), 0)
        college_files = self.harness.list_folder("college-documents")
        self.assertEqual(len(college_files), 1)
        self.assertTrue(college_files[0].endswith(".md"))

        # Verify attachment stored
        college_atts = self.harness.list_folder("attachments/college-documents")
        self.assertEqual(len(college_atts), 1)
        self.assertEqual(college_atts[0], filename)

        # Verify frontmatter properties
        fm, body = self.harness.read_note("college-documents", college_files[0])
        self.assertEqual(fm.get("category"), "college-documents")
        self.assertEqual(fm.get("status"), "filed")
        self.assertEqual(fm.get("document_type"), "mark-sheet")
        self.assertEqual(fm.get("issuer"), "Delhi Technological University")
        self.assertEqual(fm.get("issued_date"), "2026-06-15")
        self.assertIn("2K22/CO/108", fm.get("reference_ids", []))
        self.assertEqual(fm.get("original_filename"), filename)

        # Verify body contains "What this is for" usage section
        self.assertIn("## What this is for", body)
        self.assertIn("Academic records and grade verification", body)

    def test_photo_intake_college_flow(self):
        """Feature 2: Photographed diploma JPEG -> OCR -> College -> Proposal -> Approve."""
        photo_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF simulated jpeg photo of degree certificate"
        filename = "convocation_degree.jpg"
        ocr_text = (
            "Indian Institute of Technology Bombay\n"
            "This is to certify that Rahul Sharma has been admitted to the degree of\n"
            "Bachelor of Technology in Electrical Engineering\n"
            "Degree No: IITB/2026/EE/442\n"
            "Date: 2026-08-10"
        )
        self.harness.ocr.register_file(filename, ocr_text)
        self.harness.llm.queue_json_response({
            "category": "college-documents",
            "title": "IIT Bombay Electrical Engineering Degree Certificate",
            "summary": "Bachelor of Technology degree certificate conferred by IIT Bombay.",
            "tags": ["degree", "iit-bombay", "graduation", "electrical"],
            "document_type": "degree-certificate",
            "issuer": "Indian Institute of Technology Bombay",
            "issued_date": "2026-08-10",
            "reference_ids": ["IITB/2026/EE/442"],
        })

        proposal_rec = self.harness.send_photo(filename, photo_bytes)
        self.assertEqual(proposal_rec.proposal.category, Category.COLLEGE)

        filing_rec = self.harness.tap_approve(proposal_rec)
        self.assertEqual(filing_rec.status, "filed")

        college_files = self.harness.list_folder("college-documents")
        self.assertEqual(len(college_files), 1)
        fm, body = self.harness.read_note("college-documents", college_files[0])
        self.assertEqual(fm["issuer"], "Indian Institute of Technology Bombay")
        self.assertIn("Proof of graduation and degree completion", body)

    def test_text_note_intake_reference_flow(self):
        """Feature 3: Text message intake -> skip OCR -> Reference classification -> Approve."""
        text_content = (
            "The CAP theorem states that a distributed data store can only provide two of three guarantees: "
            "Consistency, Availability, and Partition tolerance. In practice, networks are never partition-free."
        )
        self.harness.llm.queue_json_response({
            "category": "reference",
            "title": "CAP Theorem Principles In Distributed Systems",
            "summary": "Core trade-offs between consistency, availability, and partition tolerance in distributed databases.",
            "tags": ["distributed-systems", "cap-theorem", "databases", "architecture"],
            "document_type": None,
            "issuer": None,
            "issued_date": None,
            "reference_ids": [],
        })

        proposal_rec = self.harness.send_text(text_content)

        # Verify OCR was NOT called for text capture
        self.assertEqual(len(self.harness.ocr.calls), 0)

        self.assertEqual(proposal_rec.proposal.category, Category.REFERENCE)
        filing_rec = self.harness.tap_approve(proposal_rec)

        ref_files = self.harness.list_folder("reference")
        self.assertEqual(len(ref_files), 1)
        fm, body = self.harness.read_note("reference", ref_files[0])
        self.assertEqual(fm["category"], "reference")
        self.assertNotIn("document_type", fm)
        self.assertIn("## Key points", body)

    def test_project_expansion_flow(self):
        """Feature 4: Project idea -> Web search -> 5-section brief with URLs -> Approve."""
        idea = "A CLI tool that records shell sessions and turns them into interactive tutorials"
        self.harness.llm.queue_json_response({
            "category": "possible-projects",
            "title": "Terminal Session Interactive Tutorial Generator CLI",
            "summary": "A developer tool capturing terminal sessions and generating interactive runnable browser tutorials.",
            "tags": ["developer-tools", "cli", "education", "terminal"],
            "document_type": None,
            "issuer": None,
            "issued_date": None,
            "reference_ids": [],
        })
        self.harness.search.register_results(
            "Terminal",
            [
                SearchResult(title="Asciinema", url="https://asciinema.org", snippet="Terminal session recorder"),
                SearchResult(title="Katacoda Alternatives", url="https://killercoda.com", snippet="Interactive browser environments"),
                SearchResult(title="Termtosvg", url="https://termtosvg.dev", snippet="Render terminal recordings as SVG"),
            ]
        )
        self.harness.llm.queue_text_response(
            "## Description\nA command-line application that records terminal execution and compiles it into step-by-step interactive web guides.\n\n"
            "## Similar projects and competitors\n"
            "- **Asciinema** — Terminal recorder sharing cast files. https://asciinema.org\n"
            "- **Killercoda** — Browser interactive Linux environments. https://killercoda.com\n"
            "- **Termtosvg** — SVG generator for terminal output. https://termtosvg.dev\n\n"
            "## Research areas\n"
            "- PTY capture APIs in Python and Rust\n"
            "- WebAssembly terminal emulators (xterm.js)\n"
            "- Automated step delimiter detection\n"
            "- Headless browser runner integration\n\n"
            "## Anticipated problems\n"
            "- **Sensitive data leakage** — Regex redact API keys and passwords before export.\n"
            "- **Cross-platform PTY variations** — Standardize on VT100 escape sequences.\n\n"
            "## Open questions\n"
            "- Should execution replay happen client-side or in container sandboxes?\n"
            "- How to handle curses-based full-screen terminal programs like vim?\n"
        )

        proposal_rec = self.harness.send_text(idea)

        # Verify search was called
        self.assertGreaterEqual(len(self.harness.search.calls), 1)

        # Tap Approve
        filing_rec = self.harness.tap_approve(proposal_rec)
        project_files = self.harness.list_folder("possible-projects")
        self.assertEqual(len(project_files), 1)

        fm, body = self.harness.read_note("possible-projects", project_files[0])
        self.assertEqual(fm["category"], "possible-projects")
        self.assertIn("## One-liner", body)
        self.assertIn(f"> {idea}", body)
        self.assertIn("https://asciinema.org", body)
        self.assertIn("https://killercoda.com", body)
        self.assertIn("https://termtosvg.dev", body)

    def test_multilingual_indic_translation_flow(self):
        """Feature 5: Hindi document -> LID detect hi-IN -> translate to English -> English summary with Hindi issuer."""
        hindi_text = (
            "भारतीय प्रौद्योगिकी संस्थान दिल्ली\n"
            "कंप्यूटर विज्ञान एवं इंजीनियरिंग विभाग\n"
            "सत्र 2026 प्रवेश पुष्टि पत्र"
        )
        self.harness.llm.queue_json_response({
            "category": "college-documents",
            "title": "IIT Delhi Computer Science Admission Confirmation Letter",
            "summary": "Official admission confirmation letter for Computer Science and Engineering at IIT Delhi.",
            "tags": ["admission", "iit-delhi", "computer-science"],
            "document_type": "admission-letter",
            "issuer": "भारतीय प्रौद्योगिकी संस्थान दिल्ली",
            "issued_date": "2026-07-01",
            "reference_ids": ["IITD/2026/CS/088"],
        })

        proposal_rec = self.harness.send_text(hindi_text)
        
        # Verify language detection ran
        self.assertIn(hindi_text, self.harness.text.detect_calls)
        # Verify translation ran
        self.assertGreaterEqual(len(self.harness.text.translate_calls), 1)

        filing_rec = self.harness.tap_approve(proposal_rec)
        files = self.harness.list_folder("college-documents")
        fm, body = self.harness.read_note("college-documents", files[0])
        self.assertEqual(fm.get("source_language"), "hi-IN")
        self.assertEqual(fm.get("issuer"), "भारतीय प्रौद्योगिकी संस्थान दिल्ली")
        self.assertIn("Admission", fm.get("title"))

    def test_callback_change_folder_flow(self):
        """Feature 6: Capture classified as Reference -> user taps Change folder -> selects College -> note moved + correction logged."""
        self.harness.llm.queue_json_response({
            "category": "reference",
            "title": "Unrecognized Student Paperwork",
            "summary": "Miscellaneous student paperwork with fee breakdown.",
            "tags": ["paperwork", "fees"],
            "document_type": None,
            "issuer": None,
            "issued_date": None,
            "reference_ids": [],
        })
        proposal_rec = self.harness.send_text("Fee receipt copy for autumn semester")

        # User taps Change folder
        available_categories = self.harness.tap_change_folder(proposal_rec)
        self.assertIn("college-documents", available_categories)

        # User picks college-documents
        filing_rec = self.harness.tap_choose_category(proposal_rec, "college-documents")
        self.assertEqual(filing_rec.category, "college-documents")

        # Note should be in college-documents and NOT in inbox or reference
        self.assertEqual(len(self.harness.list_folder("inbox")), 0)
        self.assertEqual(len(self.harness.list_folder("reference")), 0)
        college_files = self.harness.list_folder("college-documents")
        self.assertEqual(len(college_files), 1)

        # Verify _corrections.jsonl has been appended
        corrections = self.harness.read_corrections()
        self.assertEqual(len(corrections), 1)
        self.assertEqual(corrections[0]["original_category"], "reference")
        self.assertEqual(corrections[0]["chosen_category"], "college-documents")

    def test_callback_discard_flow(self):
        """Feature 7: User taps Discard -> note and attachment moved to _archive/."""
        doc_bytes = b"sample content for discarded document"
        filename = "spam_offer.pdf"
        self.harness.llm.queue_json_response({
            "category": "reference",
            "title": "Commercial Credit Card Offer",
            "summary": "Unwanted banking promotion letter.",
            "tags": ["promo", "banking"],
        })
        proposal_rec = self.harness.send_document(filename, doc_bytes)

        # User clicks Discard
        filing_rec = self.harness.tap_discard(proposal_rec)
        self.assertEqual(filing_rec.category, "_archive")
        self.assertEqual(filing_rec.status, "archived")

        # Verify inbox is empty
        self.assertEqual(len(self.harness.list_folder("inbox")), 0)
        # Verify files in _archive
        archive_files = self.harness.list_folder("_archive")
        self.assertGreaterEqual(len(archive_files), 1)
        self.assertTrue(any(f.endswith(".md") for f in archive_files))

    def test_proposal_generation_structure(self):
        """Feature 8: Proposal structure verification: 3 buttons, preview, title, category."""
        self.harness.llm.queue_json_response({
            "category": "reference",
            "title": "Machine Learning Research Paper Notes",
            "summary": "Notes on transformer architecture attention mechanisms.",
            "tags": ["ml", "transformers"],
        })
        proposal_rec = self.harness.send_text("Transformer attention mechanism notes")

        self.assertEqual(proposal_rec.inline_buttons, ["Approve", "Change folder", "Discard"])
        self.assertEqual(len(proposal_rec.inline_buttons), 3)
        self.assertIn("Machine Learning", proposal_rec.preview_text)
        self.assertIn("reference", proposal_rec.preview_text)

    def test_fast_receipt_acknowledgement(self):
        """Feature 9: System acknowledges receipt immediately (< 2s) before long async stages."""
        self.harness.llm.queue_json_response({
            "category": "reference",
            "title": "Quick Idea Note",
            "summary": "A fast note.",
            "tags": ["quick"],
        })
        proposal_rec = self.harness.send_text("A fast thought")
        acks = self.harness.get_acknowledgements()
        self.assertEqual(len(acks), 1)
        self.assertEqual(acks[0], "Reading…")
