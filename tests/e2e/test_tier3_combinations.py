"""Tier 3 — Cross-Feature Combinations: Pairwise Interaction Opaque-Box E2E Tests.

Tests interaction between multiple features and pipeline stages:
- Multilingual Photo Doc (OCR -> LID -> Translation -> College Frontmatter -> Approval)
- Obscure Project Text (Search Query -> Brief Expansion with Citations -> Discard Flow)
- Indic Tamil Doc (OCR -> Tamil LID -> Translation -> Reclassification -> Corrections Log)
- Cross-Folder Collision during Reclassification (-2 Suffix in Destination)
- Search Service External Failure during Project Brief Expansion (Error Tagging & Vault Atomicity)
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
    IngestError,
)


class TestTier3Combinations(unittest.TestCase):
    """Tier 3 Pairwise Cross-Feature Verification."""

    def setUp(self):
        self.harness = E2ETestHarness()

    def tearDown(self):
        self.harness.cleanup()

    def test_hindi_photo_doc_ocr_translation_college_approval(self):
        """Combination 1: Hindi Photo Doc -> OCR -> LID (hi-IN) -> Translation -> College Note -> Approval."""
        filename = "bhu_marksheet.jpg"
        photo_bytes = b"\xff\xd8\xff\xe0 simulated photo of hindi university marksheet"
        ocr_text = (
            "बनारस हिंदू विश्वविद्यालय\n"
            "कला संकाय - वार्षिक परीक्षा 2026\n"
            "छात्र का नाम: अमित कुमार\n"
            "अनुक्रमांक: BHU/2026/A402\n"
            "विषय: प्राचीन भारतीय इतिहास (अंक: 88/100)\n"
            "जारी करने की तिथि: 2026-06-25"
        )
        self.harness.ocr.register_file(filename, ocr_text)
        self.harness.llm.queue_json_response({
            "category": "college-documents",
            "title": "BHU Faculty Of Arts Annual Marksheet",
            "summary": "Annual examination grade card for Ancient Indian History from Banaras Hindu University.",
            "tags": ["academics", "bhu", "history", "grades"],
            "document_type": "mark-sheet",
            "issuer": "बनारस हिंदू विश्वविद्यालय",
            "issued_date": "2026-06-25",
            "reference_ids": ["BHU/2026/A402"],
        })

        # 1. Intake
        proposal_rec = self.harness.send_photo(filename, photo_bytes)

        # Verify OCR called
        self.assertIn(self.harness.media_dir / filename, self.harness.ocr.calls)
        # Verify LID called with Hindi text
        self.assertIn(ocr_text, self.harness.text.detect_calls)
        # Verify translation called with source hi-IN
        self.assertTrue(any(src == "hi-IN" for _, src in self.harness.text.translate_calls))

        # Verify proposal contains category and preview
        self.assertEqual(proposal_rec.proposal.category, Category.COLLEGE)
        self.assertIn("BHU", proposal_rec.preview_text)

        # 2. Tap Approve
        filing_rec = self.harness.tap_approve(proposal_rec)
        self.assertEqual(filing_rec.category, "college-documents")

        # 3. Verify Obsidian note properties & body
        college_notes = self.harness.list_folder("college-documents")
        self.assertEqual(len(college_notes), 1)
        fm, body = self.harness.read_note("college-documents", college_notes[0])
        self.assertEqual(fm["source_language"], "hi-IN")
        self.assertEqual(fm["issuer"], "बनारस हिंदू विश्वविद्यालय")
        self.assertEqual(fm["document_type"], "mark-sheet")
        self.assertIn("BHU/2026/A402", fm["reference_ids"])
        self.assertEqual(fm["status"], "filed")
        self.assertEqual(fm["attachment"], filename)

        # Verify attachment moved to attachments/college-documents/
        atts = self.harness.list_folder("attachments/college-documents")
        self.assertEqual(len(atts), 1)
        self.assertEqual(atts[0], filename)

    def test_obscure_project_text_search_expansion_discard(self):
        """Combination 2: Obscure Project Text -> Web Search -> Brief Expansion -> Discard Flow."""
        idea = "A peer-to-peer mesh communication protocol using near-ultrasound audio between laptop speakers and mics"
        self.harness.llm.queue_json_response({
            "category": "possible-projects",
            "title": "Acoustic Ultrasound P2P Mesh Communication Protocol",
            "summary": "High-frequency acoustic data transmission protocol establishing localized mesh networks without Wi-Fi.",
            "tags": ["audio", "networking", "p2p", "mesh"],
            "document_type": None,
            "issuer": None,
            "issued_date": None,
            "reference_ids": [],
        })
        self.harness.search.register_results(
            "ultrasound",
            [
                SearchResult(title="Quiet Modem Project", url="https://quiet.github.io/quiet-js/", snippet="Acoustic modem library using web audio"),
                SearchResult(title="Amodem Audio Transmission", url="https://github.com/romanz/amodem", snippet="Audio modem communication protocol"),
                SearchResult(title="ggwave Data-over-Sound", url="https://github.com/ggerganov/ggwave", snippet="Tiny data over sound library"),
            ]
        )
        self.harness.llm.queue_text_response(
            "## Description\nA localized mesh networking system leveraging near-ultrasound acoustic modulation between commercial laptop sound hardware.\n\n"
            "## Similar projects and competitors\n"
            "- **Quiet Modem** — Open source Web Audio acoustic modem. https://quiet.github.io/quiet-js/\n"
            "- **Amodem** — Python audio modem implementation. https://github.com/romanz/amodem\n"
            "- **ggwave** — Data-over-sound library in C++. https://github.com/ggerganov/ggwave\n\n"
            "## Research areas\n"
            "- FSK and OFDM modulation in 18kHz-21kHz range\n"
            "- Hardware microphone bandpass filter limitations\n"
            "- Ambient room reverberation and echo cancellation\n"
            "- Battery drain from continuous microphone sampling\n\n"
            "## Anticipated problems\n"
            "- **Audible harmonics** — Low-end speakers leak into audible range (16kHz).\n"
            "- **Throughput limit** — Physical bandwidth caps acoustic speeds under 2 kbps.\n\n"
            "## Open questions\n"
            "- Is 2 kbps sufficient for text messages?\n"
            "- How will domestic pets react to continuous ultrasonic carrier frequencies?\n"
        )

        # 1. Intake
        proposal_rec = self.harness.send_text(idea)
        self.assertEqual(proposal_rec.proposal.category, Category.PROJECT)

        # Verify note temporarily in inbox
        inbox_files = self.harness.list_folder("inbox")
        self.assertEqual(len(inbox_files), 1)

        # 2. User reviews and Discards
        filing_rec = self.harness.tap_discard(proposal_rec)
        self.assertEqual(filing_rec.category, "_archive")
        self.assertEqual(filing_rec.status, "archived")

        # 3. Verify inbox is clean, note is in _archive/ with status archived
        self.assertEqual(len(self.harness.list_folder("inbox")), 0)
        self.assertEqual(len(self.harness.list_folder("possible-projects")), 0)
        archive_notes = self.harness.list_folder("_archive")
        self.assertEqual(len(archive_notes), 1)
        fm, body = self.harness.read_note("_archive", archive_notes[0])
        self.assertEqual(fm["status"], "archived")
        self.assertIn("https://quiet.github.io/quiet-js/", body)

    def test_tamil_document_translation_change_folder(self):
        """Combination 3: Tamil Document -> OCR -> Translation -> Reference -> Change Folder -> College Documents."""
        filename = "anna_univ_bonafide.pdf"
        doc_bytes = b"%PDF-1.4 simulated tamil university bonafide certificate"
        ocr_text = (
            "அண்ணா பல்கலைக்கழகம் சென்னை\n"
            "மாணவர் சேர்க்கை மற்றும் உண்மைச் சான்றிதழ்\n"
            "மாணவர் பெயர்: சுந்தர் ராமன்\n"
            "பதிவு எண்: 2026/AU/CS/771\n"
            "தேதி: 2026-07-14"
        )
        self.harness.ocr.register_file(filename, ocr_text)
        # Initially classified as Reference
        self.harness.llm.queue_json_response({
            "category": "reference",
            "title": "Anna University Chennai Student Document",
            "summary": "Tamil student verification document from Anna University Chennai.",
            "tags": ["tamil", "anna-university", "student"],
            "document_type": None,
            "issuer": None,
            "issued_date": None,
            "reference_ids": [],
        })

        # 1. Intake
        proposal_rec = self.harness.send_document(filename, doc_bytes)
        self.assertEqual(proposal_rec.proposal.category, Category.REFERENCE)

        # 2. User taps Change folder
        available_categories = self.harness.tap_change_folder(proposal_rec)
        self.assertIn("college-documents", available_categories)

        # 3. User chooses college-documents
        filing_rec = self.harness.tap_choose_category(proposal_rec, "college-documents")
        self.assertEqual(filing_rec.category, "college-documents")

        # 4. Verification: Note in college-documents/, attachment in attachments/college-documents/
        self.assertEqual(len(self.harness.list_folder("inbox")), 0)
        self.assertEqual(len(self.harness.list_folder("reference")), 0)
        college_notes = self.harness.list_folder("college-documents")
        self.assertEqual(len(college_notes), 1)
        self.assertEqual(len(self.harness.list_folder("attachments/college-documents")), 1)

        # 5. Verification: _corrections.jsonl has recorded this reclassification
        corrections = self.harness.read_corrections()
        self.assertEqual(len(corrections), 1)
        self.assertEqual(corrections[0]["original_category"], "reference")
        self.assertEqual(corrections[0]["chosen_category"], "college-documents")

    def test_collision_across_reclassification(self):
        """Combination 4: Pre-existing note in destination folder causes collision suffix (-2) during reclassification."""
        title = "Financial Audit Report"
        day = date(2026, 11, 3)

        # Pre-seed Note A in reference folder
        self.harness.llm.queue_json_response({
            "category": "reference",
            "title": title,
            "summary": "Original corporate financial report.",
            "tags": ["finance"],
        })
        prop_a = self.harness.send_text("Corporate audit notes")
        self.harness.tap_approve(prop_a)
        self.assertEqual(self.harness.list_folder("reference"), ["2026-11-03 - Financial-Audit-Report.md"])

        # Capture Note B: initially classified into college-documents with SAME title
        self.harness.llm.queue_json_response({
            "category": "college-documents",
            "title": title,
            "summary": "University departmental financial audit report.",
            "tags": ["college", "finance"],
            "document_type": "fee-receipt",
            "issuer": "University Finance Office",
            "issued_date": "2026-11-03",
            "reference_ids": ["AUD-01"],
        })
        prop_b = self.harness.send_text("Departmental financial audit")
        self.assertEqual(prop_b.inbox_note_path.name, "2026-11-03 - Financial-Audit-Report.md")

        # User reclassifies Note B to reference!
        filing_b = self.harness.tap_choose_category(prop_b, "reference")
        self.assertEqual(filing_b.category, "reference")

        # Destination folder 'reference' now contains BOTH files without overwriting
        ref_files = sorted(self.harness.list_folder("reference"))
        self.assertEqual(ref_files, [
            "2026-11-03 - Financial-Audit-Report-2.md",
            "2026-11-03 - Financial-Audit-Report.md",
        ])

        # Verify contents are preserved separately
        _, body_orig = self.harness.read_note("reference", "2026-11-03 - Financial-Audit-Report.md")
        _, body_new = self.harness.read_note("reference", "2026-11-03 - Financial-Audit-Report-2.md")
        self.assertIn("Original corporate financial report", body_orig)
        self.assertIn("University departmental financial audit report", body_new)

    def test_search_service_failure_during_project_expansion(self):
        """Combination 5: Search API failure during project expansion raises IngestError('search'); vault stays clean."""
        idea = "A wearable health device for continuous blood glucose monitoring"
        self.harness.llm.queue_json_response({
            "category": "possible-projects",
            "title": "Continuous Non-Invasive Glucose Monitor Device",
            "summary": "Wearable sensor transmitting real-time glucose metrics to smartphone.",
            "tags": ["health", "hardware", "biotech"],
        })
        # Simulate search API connection failure
        self.harness.search.fail_search = True

        with self.assertRaises(IngestError) as ctx:
            self.harness.send_text(idea)

        self.assertEqual(ctx.exception.stage, "search")
        self.assertIn("Web search API connection error", str(ctx.exception))
        # Verify vault atomicity: no partial notes in inbox or possible-projects
        self.harness.assert_vault_clean()
        self.assertEqual(len(self.harness.list_folder("possible-projects")), 0)
