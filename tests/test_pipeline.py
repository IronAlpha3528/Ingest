"""Unit and pipeline integration tests using zero-network fakes."""

from __future__ import annotations

from datetime import datetime
import inspect
from pathlib import Path
import shutil
import tempfile
from typing import Any
import pytest

from ingest.adapters.vault import FileVault
from ingest.models import (
    Capture,
    CaptureKind,
    Category,
    Classification,
    IngestError,
    Note,
    Proposal,
    SearchResult,
)
from ingest.pipeline import (
    Deps,
    build_note,
    classify,
    expand_project,
    file_proposal,
    ingest,
    normalise,
)
from tests.fakes import FakeLlm, FakeOcr, FakeSearch, FakeText


def sync_run(coro):
    """Executes a coroutine chain synchronously without socketpair overhead."""
    stack = [coro]
    result = None
    while stack:
        current = stack[-1]
        try:
            val = current.send(result)
            if inspect.iscoroutine(val):
                stack.append(val)
                result = None
            else:
                result = val
        except StopIteration as e:
            stack.pop()
            result = e.value
    return result


@pytest.fixture
def test_deps():
    temp_dir = tempfile.mkdtemp()
    vault = FileVault(temp_dir)
    llm = FakeLlm(
        payload={
            "category": "college-documents",
            "title": "Engineering Physics Marksheet",
            "summary": "Official engineering physics semester marksheet.",
            "tags": ["college", "physics", "grades"],
            "document_type": "Marksheet",
            "issuer": "State University",
            "issued_date": "2026-05-10",
            "reference_ids": ["PHY-2026-01"],
        }
    )
    ocr = FakeOcr("Engineering Physics Final Exam Marks 2026 State University")
    text = FakeText(lang="en-IN")
    search = FakeSearch(
        [
            SearchResult(
                title="Expense Tracker Open Source",
                url="https://github.com/example/mess-tracker",
                snippet="Track hostel food and mess expenses easily.",
            )
        ]
    )
    return Deps(llm=llm, ocr=ocr, text=text, search=search, vault=vault)


# ===========================================================================
# Stage 1: normalise tests
# ===========================================================================

def test_normalise_text_capture(test_deps: Deps):
    async def _test():
        capture = Capture(
            chat_id=101,
            message_id=1,
            kind=CaptureKind.TEXT,
            text="Hostel mess bill reminder",
        )
        orig, trans = await normalise(capture, test_deps)
        assert orig == "Hostel mess bill reminder"
        assert trans is None

    sync_run(_test())


def test_normalise_caption_appended(test_deps: Deps):
    async def _test():
        capture = Capture(
            chat_id=101,
            message_id=2,
            kind=CaptureKind.TEXT,
            text="Hostel mess bill",
            caption="Monthly payment reminder",
        )
        orig, trans = await normalise(capture, test_deps)
        assert orig == "Hostel mess bill\n\nMonthly payment reminder"
        assert trans is None

    sync_run(_test())


def test_normalise_document_calls_ocr(test_deps: Deps):
    async def _test():
        with tempfile.NamedTemporaryFile("w", suffix=".pdf", delete=False) as tf:
            tf.write("dummy")
            path = Path(tf.name)

        try:
            capture = Capture(
                chat_id=101,
                message_id=3,
                kind=CaptureKind.DOCUMENT,
                local_path=path,
                filename="marksheet.pdf",
            )
            orig, trans = await normalise(capture, test_deps)
            assert "Engineering Physics" in orig
            assert trans is None
        finally:
            if path.exists():
                path.unlink()

    sync_run(_test())


def test_normalise_photo_with_caption_calls_ocr(test_deps: Deps):
    async def _test():
        with tempfile.NamedTemporaryFile("w", suffix=".jpg", delete=False) as tf:
            tf.write("dummy-photo-bytes")
            path = Path(tf.name)

        try:
            capture = Capture(
                chat_id=101,
                message_id=4,
                kind=CaptureKind.PHOTO,
                local_path=path,
                filename="exam_receipt.jpg",
                caption="Semester 1 exam fee receipt",
            )
            orig, trans = await normalise(capture, test_deps)
            assert "Engineering Physics Final Exam Marks" in orig
            assert "Semester 1 exam fee receipt" in orig
            assert trans is None
        finally:
            if path.exists():
                path.unlink()

    sync_run(_test())


def test_normalise_indic_language_bypasses_translation():
    async def _test():
        fake_text = FakeText(lang="hi-IN", translation="Translated English")
        deps = Deps(
            llm=FakeLlm(),
            ocr=FakeOcr(),
            text=fake_text,
            search=FakeSearch(),
            vault=FileVault(tempfile.mkdtemp()),
        )
        capture = Capture(
            chat_id=101,
            message_id=5,
            kind=CaptureKind.TEXT,
            text="छात्रवृत्ति आवेदन पत्र",
        )
        orig, trans = await normalise(capture, deps)
        assert orig == "छात्रवृत्ति आवेदन पत्र"
        # English-only directive requires bypassing translation
        assert trans is None
        assert len(fake_text.translate_calls) == 0

    sync_run(_test())


def test_normalise_media_missing_local_path_raises_ocr_error(test_deps: Deps):
    async def _test():
        capture = Capture(
            chat_id=101,
            message_id=6,
            kind=CaptureKind.DOCUMENT,
            local_path=None,
        )
        with pytest.raises(IngestError) as exc_info:
            await normalise(capture, test_deps)
        assert exc_info.value.stage == "ocr"

    sync_run(_test())


def test_normalise_ocr_failure_raises_ocr_error():
    async def _test():
        with tempfile.NamedTemporaryFile("w", suffix=".pdf", delete=False) as tf:
            tf.write("dummy")
            path = Path(tf.name)

        try:
            bad_ocr = FakeOcr(text=RuntimeError("OCR vision service unavailable"))
            deps = Deps(
                llm=FakeLlm(),
                ocr=bad_ocr,
                text=FakeText(),
                search=FakeSearch(),
                vault=FileVault(tempfile.mkdtemp()),
            )
            capture = Capture(
                chat_id=101,
                message_id=7,
                kind=CaptureKind.DOCUMENT,
                local_path=path,
            )
            with pytest.raises(IngestError) as exc_info:
                await normalise(capture, deps)
            assert exc_info.value.stage == "ocr"
        finally:
            if path.exists():
                path.unlink()

    sync_run(_test())


def test_normalise_text_detect_failure_raises_download_error():
    async def _test():
        class FailingText:
            async def detect(self, text: str) -> str:
                raise ConnectionError("Language detection service down")

            async def to_english(self, text: str, source: str) -> str:
                return text

        deps = Deps(
            llm=FakeLlm(),
            ocr=FakeOcr(),
            text=FailingText(),  # type: ignore
            search=FakeSearch(),
            vault=FileVault(tempfile.mkdtemp()),
        )
        capture = Capture(chat_id=101, message_id=8, kind=CaptureKind.TEXT, text="Some text")
        with pytest.raises(IngestError) as exc_info:
            await normalise(capture, deps)
        assert exc_info.value.stage == "download"

    sync_run(_test())


# ===========================================================================
# Stage 2: classify tests
# ===========================================================================

def test_classify_success(test_deps: Deps):
    async def _test():
        capture = Capture(chat_id=101, message_id=10, kind=CaptureKind.TEXT, text="Physics marksheet")
        cls = await classify(capture, "Physics marksheet", None, test_deps)
        assert cls.category == Category.COLLEGE
        assert cls.title == "Engineering Physics Marksheet"
        assert cls.document_type == "Marksheet"

    sync_run(_test())


def test_classify_repair_attempt_success(test_deps: Deps):
    async def _test():
        class RepairLlm:
            def __init__(self):
                self.calls = []
                self.responses = [
                    {},  # First call returns invalid schema
                    {    # Second repair call returns valid schema
                        "category": "reference",
                        "title": "Machine Learning Research Paper",
                        "summary": "Notes on neural attention architectures.",
                        "tags": ["machine-learning", "ai"],
                        "document_type": None,
                        "issuer": None,
                        "issued_date": None,
                        "reference_ids": [],
                    },
                ]

            async def complete_json(self, system: str, user: str) -> dict[str, Any]:
                self.calls.append((system, user))
                return self.responses.pop(0)

            async def complete_text(self, system: str, user: str) -> str:
                return ""

        repair_llm = RepairLlm()
        deps = Deps(
            llm=repair_llm,  # type: ignore
            ocr=test_deps.ocr,
            text=test_deps.text,
            search=test_deps.search,
            vault=test_deps.vault,
        )
        capture = Capture(chat_id=101, message_id=11, kind=CaptureKind.TEXT, text="ML attention paper")
        cls = await classify(capture, "ML attention paper", None, deps)
        assert cls.category == Category.REFERENCE
        assert cls.title == "Machine Learning Research Paper"
        assert len(repair_llm.calls) == 2
        assert "Previous response was invalid" in repair_llm.calls[1][1]

    sync_run(_test())


def test_classify_invalid_json_raises_ingest_error(test_deps: Deps):
    async def _test():
        bad_llm = FakeLlm(payload={})  # Missing required fields like category, title, summary
        bad_deps = Deps(
            llm=bad_llm,
            ocr=test_deps.ocr,
            text=test_deps.text,
            search=test_deps.search,
            vault=test_deps.vault,
        )
        capture = Capture(chat_id=101, message_id=12, kind=CaptureKind.TEXT, text="invalid")
        with pytest.raises(IngestError) as exc_info:
            await classify(capture, "invalid", None, bad_deps)
        assert exc_info.value.stage == "classify"

    sync_run(_test())


def test_classify_llm_network_error_raises_classify_error(test_deps: Deps):
    async def _test():
        error_llm = FakeLlm(payload=ConnectionResetError("Remote API disconnected"))
        deps = Deps(
            llm=error_llm,
            ocr=test_deps.ocr,
            text=test_deps.text,
            search=test_deps.search,
            vault=test_deps.vault,
        )
        capture = Capture(chat_id=101, message_id=13, kind=CaptureKind.TEXT, text="test")
        with pytest.raises(IngestError) as exc_info:
            await classify(capture, "test", None, deps)
        assert exc_info.value.stage == "classify"

    sync_run(_test())


# ===========================================================================
# Stage 3: expand_project tests
# ===========================================================================

def test_expand_project_generates_brief(test_deps: Deps):
    async def _test():
        brief = await expand_project("Hostel mess tracking app", test_deps)
        assert "## Description" in brief

    sync_run(_test())


def test_expand_project_with_search_results():
    async def _test():
        custom_search = FakeSearch(
            results=[
                SearchResult(
                    title="Mess Management System",
                    url="https://example.com/mess-app",
                    snippet="Automated hostel mess billing",
                ),
                SearchResult(
                    title="Campus Expense Ledger",
                    url="https://example.com/ledger",
                    snippet="Student daily expense records",
                ),
            ]
        )
        llm = FakeLlm(
            text=(
                "## Description\nA student expense tracker.\n\n"
                "## Similar projects and competitors\n"
                "- **Mess Management System** — Automated hostel mess billing. https://example.com/mess-app\n"
                "- **Campus Expense Ledger** — Student expense records. https://example.com/ledger\n"
            )
        )
        deps = Deps(
            llm=llm,
            ocr=FakeOcr(),
            text=FakeText(),
            search=custom_search,
            vault=FileVault(tempfile.mkdtemp()),
        )
        brief = await expand_project("Mess bill tracker", deps)
        assert "https://example.com/mess-app" in brief
        assert len(custom_search.calls) == 1
        assert custom_search.calls[0] == ("Mess bill tracker", 5)

    sync_run(_test())


def test_expand_project_search_failure_raises_search_error(test_deps: Deps):
    async def _test():
        class FailingSearch:
            async def search(self, query: str, k: int = 5) -> list[SearchResult]:
                raise TimeoutError("Search service gateway timeout")

        deps = Deps(
            llm=test_deps.llm,
            ocr=test_deps.ocr,
            text=test_deps.text,
            search=FailingSearch(),  # type: ignore
            vault=test_deps.vault,
        )
        with pytest.raises(IngestError) as exc_info:
            await expand_project("New idea", deps)
        assert exc_info.value.stage == "search"

    sync_run(_test())


def test_expand_project_llm_failure_raises_search_error(test_deps: Deps):
    async def _test():
        bad_llm = FakeLlm(text=RuntimeError("LLM brief completion quota exceeded"))
        deps = Deps(
            llm=bad_llm,
            ocr=test_deps.ocr,
            text=test_deps.text,
            search=test_deps.search,
            vault=test_deps.vault,
        )
        with pytest.raises(IngestError) as exc_info:
            await expand_project("New idea", deps)
        assert exc_info.value.stage == "search"

    sync_run(_test())


# ===========================================================================
# Stage 4: build_note tests
# ===========================================================================

def test_build_note_college_with_attachment(test_deps: Deps):
    async def _test():
        capture = Capture(
            chat_id=101,
            message_id=20,
            kind=CaptureKind.DOCUMENT,
            local_path=Path("/tmp/marksheet.pdf"),
            filename="marksheet.pdf",
            received_at=datetime(2026, 11, 3, 10, 0, 0),
        )
        cls = Classification(
            category=Category.COLLEGE,
            title="Engineering Physics Marksheet",
            summary="Final semester physics marks.",
            tags=["college", "physics"],
            document_type="Marksheet",
            issuer="State University",
            issued_date="2026-05-10",
            reference_ids=["PHY-01"],
        )
        note = await build_note(capture, cls, test_deps)
        assert note.filename == "2026-11-03 - Engineering-Physics-Marksheet.md"
        assert note.frontmatter["attachment"] == "attachments/college-documents/marksheet.pdf"
        assert note.frontmatter["category"] == "college-documents"
        assert note.frontmatter["status"] == "inbox"
        assert note.frontmatter["document_type"] == "Marksheet"
        assert note.frontmatter["issuer"] == "State University"
        assert "## Key details" in note.body
        assert "| Document type | Marksheet |" in note.body

    sync_run(_test())


def test_build_note_project_with_extra_body(test_deps: Deps):
    async def _test():
        capture = Capture(
            chat_id=101,
            message_id=21,
            kind=CaptureKind.TEXT,
            text="Hostel mess billing assistant",
            received_at=datetime(2026, 11, 3, 10, 0, 0),
        )
        cls = Classification(
            category=Category.PROJECT,
            title="Hostel Mess Billing Assistant",
            summary="An automated tracker for mess bills.",
            tags=["hostel", "finance"],
        )
        extra_brief = "## Description\nDetailed research brief.\n\n## Competitors\n- Competitor A"
        note = await build_note(capture, cls, test_deps, body_extra=extra_brief)
        assert note.filename == "2026-11-03 - Hostel-Mess-Billing-Assistant.md"
        assert note.frontmatter.get("attachment") is None
        assert "## One-liner" in note.body
        assert "> Hostel mess billing assistant" in note.body
        assert "## Competitors" in note.body

    sync_run(_test())


def test_build_note_reference_split_sentences(test_deps: Deps):
    async def _test():
        capture = Capture(
            chat_id=101,
            message_id=22,
            kind=CaptureKind.TEXT,
            text="Attention is all you need paper",
            received_at=datetime(2026, 11, 3, 10, 0, 0),
        )
        cls = Classification(
            category=Category.REFERENCE,
            title="Transformer Architecture Overview",
            summary="The Transformer model relies on self-attention mechanisms. It dispenses with recurrence and convolutions entirely.",
            tags=["ml", "transformers"],
        )
        note = await build_note(capture, cls, test_deps)
        assert "## Summary" in note.body
        assert "## Key points" in note.body
        assert "- The Transformer model relies on self-attention mechanisms." in note.body
        assert "- It dispenses with recurrence and convolutions entirely." in note.body

    sync_run(_test())


def test_build_note_render_failure_raises_render_error(test_deps: Deps):
    async def _test():
        # Malformed capture with invalid received_at causing render failure
        class BrokenCapture:
            kind = CaptureKind.TEXT
            local_path = None
            filename = None
            text = "test"
            caption = None
            received_at = "not-a-datetime"

        cls = Classification(
            category=Category.REFERENCE,
            title="Test Note",
            summary="Summary text.",
        )
        with pytest.raises(IngestError) as exc_info:
            await build_note(BrokenCapture(), cls, test_deps)  # type: ignore
        assert exc_info.value.stage == "render"

    sync_run(_test())


# ===========================================================================
# Stage 5: ingest tests
# ===========================================================================

def test_ingest_college_document_end_to_end(test_deps: Deps):
    async def _test():
        with tempfile.NamedTemporaryFile("w", suffix=".pdf", delete=False) as tf:
            tf.write("marksheet content")
            local_pdf = Path(tf.name)

        try:
            capture = Capture(
                chat_id=101,
                message_id=30,
                kind=CaptureKind.DOCUMENT,
                local_path=local_pdf,
                filename="marksheet.pdf",
                received_at=datetime(2026, 11, 3, 12, 0, 0),
            )
            proposal = await ingest(capture, test_deps)
            assert proposal.category == Category.COLLEGE
            assert proposal.note.filename.endswith(".md")
            assert "Engineering Physics Marksheet" in proposal.preview

            # Note must be present in inbox
            inbox_note = test_deps.vault.root / "inbox" / proposal.note.filename
            assert inbox_note.exists()

            # Attachment stored
            att = test_deps.vault.root / "attachments" / "college-documents" / "marksheet.pdf"
            assert att.exists()

            # File proposal
            filed_path = file_proposal(proposal, Category.COLLEGE, test_deps)
            assert not inbox_note.exists()
            assert filed_path.exists()
            assert filed_path.parent == test_deps.vault.root / "college-documents"
            assert "status: filed" in filed_path.read_text(encoding="utf-8")
        finally:
            if local_pdf.exists():
                local_pdf.unlink()

    sync_run(_test())


def test_ingest_project_triggers_expansion_and_writes_inbox():
    async def _test():
        temp_dir = tempfile.mkdtemp()
        vault = FileVault(temp_dir)
        llm = FakeLlm(
            payload={
                "category": "possible-projects",
                "title": "Autonomous Garden Watering",
                "summary": "Soil sensor and pump control project.",
                "tags": ["iot", "gardening"],
                "document_type": None,
                "issuer": None,
                "issued_date": None,
                "reference_ids": [],
            },
            text=(
                "## Description\nAn automated watering system.\n\n"
                "## Similar projects and competitors\n"
                "- **OpenSprinkler** — Open-source irrigation. https://opensprinkler.com\n"
            ),
        )
        search = FakeSearch([
            SearchResult(
                title="OpenSprinkler",
                url="https://opensprinkler.com",
                snippet="Open-source smart sprinkler controller",
            )
        ])
        deps = Deps(
            llm=llm,
            ocr=FakeOcr(),
            text=FakeText(),
            search=search,
            vault=vault,
        )

        capture = Capture(
            chat_id=101,
            message_id=31,
            kind=CaptureKind.TEXT,
            text="Autonomous garden watering system",
            received_at=datetime(2026, 11, 3, 14, 0, 0),
        )
        proposal = await ingest(capture, deps)
        assert proposal.category == Category.PROJECT
        assert "Autonomous Garden Watering" in proposal.preview

        # Note written to inbox
        inbox_file = vault.root / "inbox" / proposal.note.filename
        assert inbox_file.exists()
        content = inbox_file.read_text(encoding="utf-8")
        assert "https://opensprinkler.com" in content
        assert "category: possible-projects" in content

    sync_run(_test())


def test_ingest_vault_write_failure_raises_write_error(test_deps: Deps):
    async def _test():
        class FailingVault:
            def write_note(self, note: Note, folder: str) -> Path:
                raise OSError("Disk full: cannot write note to inbox")

            def move_note(self, src: Path, folder: str) -> Path:
                return src

            def store_attachment(self, src: Path, category: str) -> Path:
                return src

        deps = Deps(
            llm=test_deps.llm,
            ocr=test_deps.ocr,
            text=test_deps.text,
            search=test_deps.search,
            vault=FailingVault(),  # type: ignore
        )
        capture = Capture(chat_id=101, message_id=32, kind=CaptureKind.TEXT, text="Some text")
        with pytest.raises(IngestError) as exc_info:
            await ingest(capture, deps)
        assert exc_info.value.stage == "write"

    sync_run(_test())


def test_ingest_attachment_failure_cleans_up_inbox_note(test_deps: Deps):
    async def _test():
        temp_dir = tempfile.mkdtemp()
        real_vault = FileVault(temp_dir)

        class FailingAttachmentVault:
            root = real_vault.root

            def write_note(self, note: Note, folder: str) -> Path:
                return real_vault.write_note(note, folder)

            def move_note(self, src: Path, folder: str) -> Path:
                return real_vault.move_note(src, folder)

            def store_attachment(self, src: Path, category: str, filename: str | None = None) -> Path:
                raise OSError("Permission denied: cannot store attachment")

        deps = Deps(
            llm=test_deps.llm,
            ocr=test_deps.ocr,
            text=test_deps.text,
            search=test_deps.search,
            vault=FailingAttachmentVault(),  # type: ignore
        )

        with tempfile.NamedTemporaryFile("w", suffix=".pdf", delete=False) as tf:
            tf.write("content")
            local_pdf = Path(tf.name)

        try:
            capture = Capture(
                chat_id=101,
                message_id=33,
                kind=CaptureKind.DOCUMENT,
                local_path=local_pdf,
                filename="doc.pdf",
            )
            with pytest.raises(IngestError) as exc_info:
                await ingest(capture, deps)
            assert exc_info.value.stage == "write"

            # Verify no partial note was left behind in inbox (FR-11)
            inbox_files = list((real_vault.root / "inbox").iterdir())
            assert len(inbox_files) == 0
        finally:
            if local_pdf.exists():
                local_pdf.unlink()
            shutil.rmtree(temp_dir, ignore_errors=True)

    sync_run(_test())


# ===========================================================================
# Stage 6: file_proposal tests
# ===========================================================================

def test_file_proposal_with_correction_logs_jsonl(test_deps: Deps):
    async def _test():
        capture = Capture(chat_id=101, message_id=40, kind=CaptureKind.TEXT, text="Physics marksheet")
        proposal = await ingest(capture, test_deps)

        # File into reference instead of college
        filed_path = file_proposal(proposal, Category.REFERENCE, test_deps)
        assert filed_path.parent == test_deps.vault.root / "reference"

        # Verify _corrections.jsonl has a record
        corr_file = test_deps.vault.root / "_corrections.jsonl"
        assert corr_file.exists()
        corr_content = corr_file.read_text(encoding="utf-8")
        assert "college-documents" in corr_content
        assert "reference" in corr_content

    sync_run(_test())


def test_file_proposal_no_correction_when_same_category(test_deps: Deps):
    async def _test():
        capture = Capture(chat_id=101, message_id=41, kind=CaptureKind.TEXT, text="Physics marksheet")
        proposal = await ingest(capture, test_deps)

        # File into original category
        filed_path = file_proposal(proposal, Category.COLLEGE, test_deps)
        assert filed_path.parent == test_deps.vault.root / "college-documents"

        # _corrections.jsonl must not be written when approved without category change
        corr_file = test_deps.vault.root / "_corrections.jsonl"
        assert not corr_file.exists()

    sync_run(_test())


def test_file_proposal_with_correction_moves_attachment(test_deps: Deps):
    async def _test():
        with tempfile.NamedTemporaryFile("w", suffix=".pdf", delete=False) as tf:
            tf.write("marksheet content")
            local_pdf = Path(tf.name)

        try:
            capture = Capture(
                chat_id=101,
                message_id=42,
                kind=CaptureKind.DOCUMENT,
                local_path=local_pdf,
                filename="marksheet.pdf",
                received_at=datetime(2026, 11, 3, 15, 0, 0),
            )
            proposal = await ingest(capture, test_deps)
            initial_att = test_deps.vault.root / "attachments" / "college-documents" / "marksheet.pdf"
            assert initial_att.exists()

            # User corrects category to reference
            filed_path = file_proposal(proposal, Category.REFERENCE, test_deps)
            assert filed_path.parent == test_deps.vault.root / "reference"

            # Attachment relocated to reference and removed from college
            new_att = test_deps.vault.root / "attachments" / "reference" / "marksheet.pdf"
            assert new_att.exists()
            assert not initial_att.exists()

            # Correction logged
            corr_file = test_deps.vault.root / "_corrections.jsonl"
            assert corr_file.exists()
            assert "college-documents" in corr_file.read_text(encoding="utf-8")
            assert "reference" in corr_file.read_text(encoding="utf-8")
        finally:
            if local_pdf.exists():
                local_pdf.unlink()

    sync_run(_test())


def test_file_proposal_vault_move_failure_raises_write_error(test_deps: Deps):
    async def _test():
        class FailingMoveVault:
            root = test_deps.vault.root

            def move_note(self, src: Path, folder: str) -> Path:
                raise OSError("Move note permission denied")

        deps = Deps(
            llm=test_deps.llm,
            ocr=test_deps.ocr,
            text=test_deps.text,
            search=test_deps.search,
            vault=FailingMoveVault(),  # type: ignore
        )

        capture = Capture(chat_id=101, message_id=43, kind=CaptureKind.TEXT, text="Test text")
        proposal = await ingest(capture, test_deps)

        with pytest.raises(IngestError) as exc_info:
            file_proposal(proposal, Category.COLLEGE, deps)
        assert exc_info.value.stage == "write"

    sync_run(_test())


def test_ingest_attachment_collision_synchronizes_frontmatter(test_deps: Deps):
    """Test that attachment collision renaming is accurately synchronized in the note frontmatter."""
    async def _test():
        # Pre-seed an attachment collision
        att_dir = test_deps.vault.root / "attachments" / "college-documents"
        att_dir.mkdir(parents=True, exist_ok=True)
        (att_dir / "marksheet.pdf").write_bytes(b"existing-file")

        with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as tf:
            tf.write(b"new-marksheet-content")
            local_pdf = Path(tf.name)

        try:
            capture = Capture(
                chat_id=101,
                message_id=44,
                kind=CaptureKind.DOCUMENT,
                local_path=local_pdf,
                filename="marksheet.pdf",
            )
            proposal = await ingest(capture, test_deps)
            # Frontmatter attachment link must match the newly assigned collision filename: marksheet-2.pdf
            assert proposal.note.frontmatter["attachment"] == "attachments/college-documents/marksheet-2.pdf"
            assert (att_dir / "marksheet-2.pdf").exists()
        finally:
            if local_pdf.exists():
                local_pdf.unlink()

    sync_run(_test())
