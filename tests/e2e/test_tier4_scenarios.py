"""Tier 4 — Real-World Application Scenarios: End-to-End User Workflows.

Tests end-to-end user workflows matching real-world operational scenarios:
- Scenario 1: Academic semester marksheet filing (student workflow with multi-field extraction, usage lookup, attachment archiving).
- Scenario 2: Startup idea capture with research brief (founder workflow with web search grounding, competitor citations, 5-section brief).
- Scenario 3: Multilingual receipt capture with user correction (bilingual receipt, translation normalisation, category correction to reference, audit logging).
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


class TestTier4Scenarios(unittest.TestCase):
    """Tier 4 Real-World Application Scenario Verification."""

    def setUp(self):
        self.harness = E2ETestHarness()

    def tearDown(self):
        self.harness.cleanup()

    def test_academic_semester_marksheet_filing_scenario(self):
        """Scenario 1: Student files official final semester grade card PDF.
        
        Workflow:
        1. Student receives 'Final_Semester_Grade_Card_DTU_2026.pdf' from university portal.
        2. Ingest pipeline runs OCR, language detection, and extracts academic metadata.
        3. Classification tags: college-documents, document_type: mark-sheet, usage resolved.
        4. User reviews proposal on Telegram and approves.
        5. Vault note filed in college-documents/ and attachment preserved in attachments/college-documents/.
        """
        filename = "Final_Semester_Grade_Card_DTU_2026.pdf"
        pdf_bytes = b"%PDF-1.4 official transcript binary content from DTU examination cell"
        scanned_ocr = (
            "Delhi Technological University\n"
            "Formerly Delhi College of Engineering - Established by Govt. of NCT of Delhi\n"
            "Shahbad Daulatpur, Bawana Road, Delhi-110042\n\n"
            "PROVISIONAL GRADE STATEMENT\n"
            "Eighth Semester Bachelor of Technology (Software Engineering) Examination May 2026\n"
            "Candidate Name: Abhay Sharma\n"
            "Roll No: 2K22/SE/042\n"
            "Registration No: DTU/2022/88219\n\n"
            "Course Code | Course Title | Credits | Grade\n"
            "SE-401 | Cloud Computing & Microservices | 4 | A+\n"
            "SE-402 | Deep Learning Applications | 4 | O\n"
            "SE-403 | Automated Software Testing | 3 | A\n"
            "SE-404 | Major Project Dissertation | 6 | O\n\n"
            "Semester SGPA: 9.24 | Cumulative CGPA: 8.92\n"
            "Result: Passed with Distinction\n"
            "Date of Issue: 2026-06-20\n"
            "Controller of Examinations: Prof. S. K. Gupta"
        )
        self.harness.ocr.register_file(filename, scanned_ocr)
        self.harness.llm.queue_json_response({
            "category": "college-documents",
            "title": "DTU Final Semester Grade Card Software Engineering",
            "summary": "Official grade statement for final Semester VIII B.Tech in Software Engineering at Delhi Technological University. Passed with Distinction and CGPA 8.92.",
            "tags": ["dtu", "grades", "software-engineering", "cgpa", "final-semester"],
            "document_type": "mark-sheet",
            "issuer": "Delhi Technological University",
            "issued_date": "2026-06-20",
            "reference_ids": ["2K22/SE/042", "DTU/2022/88219"],
        })

        # 1. User sends document via Telegram
        proposal_rec = self.harness.send_document(filename, pdf_bytes)

        # 2. Fast acknowledgement verified
        self.assertEqual(self.harness.get_acknowledgements(), ["Reading…"])

        # 3. Proposal verification
        self.assertEqual(proposal_rec.proposal.category, Category.COLLEGE)
        self.assertIn("DTU", proposal_rec.preview_text)
        self.assertIn("mark-sheet", proposal_rec.proposal.note.body)
        self.assertEqual(proposal_rec.inline_buttons, ["Approve", "Change folder", "Discard"])

        # 4. User approves filing
        filing_rec = self.harness.tap_approve(proposal_rec)
        self.assertEqual(filing_rec.category, "college-documents")
        self.assertEqual(filing_rec.status, "filed")

        # 5. Opaque Vault Inspection
        self.assertEqual(len(self.harness.list_folder("inbox")), 0)
        college_notes = self.harness.list_folder("college-documents")
        self.assertEqual(len(college_notes), 1)
        expected_note_name = "2026-11-03 - DTU-Final-Semester-Grade-Card-Software-Engineering.md"
        self.assertEqual(college_notes[0], expected_note_name)

        # 6. Frontmatter & Body Validation
        fm, body = self.harness.read_note("college-documents", expected_note_name)
        self.assertEqual(fm["title"], "DTU Final Semester Grade Card Software Engineering")
        self.assertEqual(fm["category"], "college-documents")
        self.assertEqual(fm["status"], "filed")
        self.assertEqual(fm["document_type"], "mark-sheet")
        self.assertEqual(fm["issuer"], "Delhi Technological University")
        self.assertEqual(fm["issued_date"], "2026-06-20")
        self.assertEqual(fm["reference_ids"], ["2K22/SE/042", "DTU/2022/88219"])
        self.assertEqual(fm["original_filename"], filename)
        self.assertEqual(fm["attachment"], filename)

        # 7. Body template table and fixed usage lookup validation
        self.assertIn("## Summary", body)
        self.assertIn("CGPA 8.92", body)
        self.assertIn("## Key details", body)
        self.assertIn("| Document type | mark-sheet |", body)
        self.assertIn("| Issuer | Delhi Technological University |", body)
        self.assertIn("| Issued | 2026-06-20 |", body)
        self.assertIn("| Reference | 2K22/SE/042, DTU/2022/88219 |", body)
        self.assertIn("## What this is for", body)
        self.assertIn("Academic records and grade verification", body)

        # 8. Attachment stored and accessible
        atts = self.harness.list_folder("attachments/college-documents")
        self.assertEqual(atts, [filename])

    def test_startup_idea_capture_with_research_brief_scenario(self):
        """Scenario 2: Founder captures startup idea and receives grounded research brief.
        
        Workflow:
        1. Founder sends one-liner idea: AI patent infringement monitoring agent for hardware startups.
        2. Pipeline classifies as possible-projects.
        3. Web search searches competitive landscape and returns 5 grounded results.
        4. LLM formats research brief with cited competitor URLs, research areas, risks, and open questions.
        5. Founder reviews proposal and approves into possible-projects/.
        """
        idea = "An autonomous AI agent that monitors daily patent filings and GitHub releases to alert hardware startups about infringement risks before tape-out"
        self.harness.llm.queue_json_response({
            "category": "possible-projects",
            "title": "Autonomous Hardware Patent Infringement Monitoring Agent",
            "summary": "AI intelligence agent continuously auditing USPTO and EPO patent filings alongside open source semiconductor repositories to flag infringement vectors before silicon tape-out.",
            "tags": ["hardware", "patents", "semiconductors", "ai-agent", "intellectual-property"],
            "document_type": None,
            "issuer": None,
            "issued_date": None,
            "reference_ids": [],
        })
        self.harness.search.register_results(
            "patent",
            [
                SearchResult(title="Patsnap Connected Innovation Platform", url="https://www.patsnap.com", snippet="AI-powered patent intelligence for R&D teams"),
                SearchResult(title="ClaimMaster Automated Patent Analytics", url="https://www.patentclaimmaster.com", snippet="Patent proofreading and claim analysis automation"),
                SearchResult(title="IPlytics Essentiality Intelligence", url="https://www.iplytics.com", snippet="Market intelligence platform for standard-essential patents"),
                SearchResult(title="Amplified AI Patent Search", url="https://www.amplified.ai", snippet="Instant prior art and freedom-to-operate search engine"),
                SearchResult(title="Clarivate Derwent Patent Index", url="https://clarivate.com/products/ip-intelligence", snippet="Curated global patent research database"),
            ]
        )
        self.harness.llm.queue_text_response(
            "## Description\n"
            "An autonomous monitoring platform designed specifically for semiconductor and hardware startups that indexes newly published patent applications from USPTO, EPO, and WIPO. It cross-references proprietary hardware RTL descriptions and public GitHub hardware projects to flag potential freedom-to-operate (FTO) conflicts before committing to million-dollar tape-outs.\n\n"
            "The system delivers automated risk alerts with claim-level element mapping to hardware engineers, drastically reducing patent attorney billable discovery hours.\n\n"
            "## Similar projects and competitors\n"
            "- **Patsnap** — Broad enterprise IP intelligence and R&D monitoring platform. https://www.patsnap.com\n"
            "- **ClaimMaster** — Desktop automated patent proofreading and claim structure analyzer. https://www.patentclaimmaster.com\n"
            "- **IPlytics** — Analytics platform specializing in telecommunications standard-essential patents. https://www.iplytics.com\n"
            "- **Amplified AI** — AI-assisted prior art and freedom-to-operate searching. https://www.amplified.ai\n"
            "- **Clarivate Derwent** — Global curated database of patent publications and citations. https://clarivate.com/products/ip-intelligence\n\n"
            "## Research areas\n"
            "- Natural language AST parsing of Verilog and VHDL hardware descriptions\n"
            "- Semantic embedding models trained on patent legal claim language\n"
            "- USPTO PAIR bulk data ingestion pipelines and publication APIs\n"
            "- Claim element tree matching against RTL module definitions\n"
            "- Confidentiality protocols for scanning proprietary IP without cloud leakage\n\n"
            "## Anticipated problems\n"
            "- **Patent publication latency** — Patent applications remain secret for 18 months from filing date; mitigation: combine with academic preprint monitoring.\n"
            "- **Willful infringement liability** — Reviewing competitor patents can expose companies to treble damages; mitigation: implement strict legal cleanroom boundaries.\n"
            "- **High false positive rate** — Broad patent claims trigger spurious warnings; mitigation: engineer claim limitation ranking algorithms.\n\n"
            "## Open questions\n"
            "- Will hardware startup founders upload pre-silicon RTL to a SaaS service?\n"
            "- Can local LLMs running on an air-gapped server perform claim-matching accurately?\n"
            "- What pricing model aligns with early-stage hardware seed rounds?\n"
        )

        # 1. Send idea
        proposal_rec = self.harness.send_text(idea)

        # 2. Verify search was called with the idea
        self.assertGreaterEqual(len(self.harness.search.calls), 1)

        # 3. Approve
        filing_rec = self.harness.tap_approve(proposal_rec)
        self.assertEqual(filing_rec.category, "possible-projects")

        # 4. Verify note filed in possible-projects/
        proj_files = self.harness.list_folder("possible-projects")
        self.assertEqual(len(proj_files), 1)
        expected_note = "2026-11-03 - Autonomous-Hardware-Patent-Infringement-Monitoring-Agent.md"
        self.assertEqual(proj_files[0], expected_note)

        # 5. Verify full content and 5 required sections
        fm, body = self.harness.read_note("possible-projects", expected_note)
        self.assertEqual(fm["status"], "filed")
        self.assertEqual(fm["category"], "possible-projects")
        self.assertIn("## One-liner", body)
        self.assertIn(f"> {idea}", body)
        self.assertIn("## Description", body)
        self.assertIn("## Similar projects and competitors", body)
        self.assertIn("## Research areas", body)
        self.assertIn("## Anticipated problems", body)
        self.assertIn("## Open questions", body)

        # 6. Verify competitor URLs are grounded in search results
        self.assertIn("https://www.patsnap.com", body)
        self.assertIn("https://www.patentclaimmaster.com", body)
        self.assertIn("https://www.iplytics.com", body)
        self.assertIn("https://www.amplified.ai", body)
        self.assertIn("https://clarivate.com/products/ip-intelligence", body)

    def test_multilingual_receipt_note_capture_scenario(self):
        """Scenario 3: Multilingual receipt capture with folder correction to reference.
        
        Workflow:
        1. Student photographs a Hindi hostel mess bill.
        2. Pipeline performs OCR, LID (hi-IN), and English translation.
        3. Classifier initially places it in college-documents (fee-receipt).
        4. Student wants financial receipts categorized under reference for personal expense tracking.
        5. Student clicks Change folder -> selects reference.
        6. Note filed in reference/, attachment in attachments/reference/, and _corrections.jsonl logged.
        """
        filename = "hostel_mess_bill_nov2026.jpg"
        photo_bytes = b"\xff\xd8\xff\xe0 photo bytes of hindi hostel mess bill"
        ocr_text = (
            "भारतीय प्रौद्योगिकी संस्थान रुड़की\n"
            "छात्रावास भोजनालय समिति - मेस देय रसीद\n"
            "छात्र का नाम: गौरव वर्मा | कमरा संख्या: बी-214\n"
            "माह: अक्टूबर 2026 | कुल देय राशि: ₹4,850\n"
            "भुगतान संदर्भ संख्या: IITR/MESS/2026/10/9941\n"
            "दिनांक: 2026-11-01"
        )
        self.harness.ocr.register_file(filename, ocr_text)
        self.harness.llm.queue_json_response({
            "category": "college-documents",
            "title": "IIT Roorkee Hostel Mess Fee Receipt",
            "summary": "Monthly mess bill receipt for October 2026 from IIT Roorkee Hostel Dining Committee totaling 4,850 INR.",
            "tags": ["iit-roorkee", "hostel", "mess", "fees", "expenses"],
            "document_type": "fee-receipt",
            "issuer": "भारतीय प्रौद्योगिकी संस्थान रुड़की",
            "issued_date": "2026-11-01",
            "reference_ids": ["IITR/MESS/2026/10/9941"],
        })

        # 1. Inbound Photo
        proposal_rec = self.harness.send_photo(filename, photo_bytes)
        self.assertEqual(proposal_rec.proposal.category, Category.COLLEGE)

        # 2. User taps Change folder -> reference
        available_folders = self.harness.tap_change_folder(proposal_rec)
        self.assertIn("reference", available_folders)

        filing_rec = self.harness.tap_choose_category(proposal_rec, "reference")
        self.assertEqual(filing_rec.category, "reference")
        self.assertEqual(filing_rec.status, "filed")

        # 3. Verify filing locations
        self.assertEqual(len(self.harness.list_folder("inbox")), 0)
        self.assertEqual(len(self.harness.list_folder("college-documents")), 0)
        ref_notes = self.harness.list_folder("reference")
        self.assertEqual(len(ref_notes), 1)

        expected_note = "2026-11-03 - IIT-Roorkee-Hostel-Mess-Fee-Receipt.md"
        self.assertEqual(ref_notes[0], expected_note)

        # 4. Verify Frontmatter retains source language
        fm, body = self.harness.read_note("reference", expected_note)
        self.assertEqual(fm["source_language"], "hi-IN")
        self.assertEqual(fm["status"], "filed")
        self.assertEqual(fm["attachment"], filename)

        # 5. Verify attachment moved to attachments/reference/
        ref_atts = self.harness.list_folder("attachments/reference")
        self.assertEqual(ref_atts, [filename])

        # 6. Verify audit logging in _corrections.jsonl
        corrections = self.harness.read_corrections()
        self.assertEqual(len(corrections), 1)
        self.assertEqual(corrections[0]["original_category"], "college-documents")
        self.assertEqual(corrections[0]["chosen_category"], "reference")
        self.assertIn("IIT Roorkee", corrections[0]["note_title"])
