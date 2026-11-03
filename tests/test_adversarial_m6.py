"""Tier 5 Adversarial Coverage Hardening Suite for Milestone 6.

Empirical verification and white-box stress testing across:
1. pipeline.py: Corrupted files, network drops, empty strings, mismatched schema fields,
   two-stage repair flows, atomic transaction rollback on attachment failure, and category relocation.
2. bot.py: Duplicate callbacks, 64-byte callback_data boundary conditions, token hash collision
   resolution without chat_data, malformed Telegram update structures, size limit checks, markdown
   parse fallbacks, and typed IngestError presentation.
3. vault.py: Extreme collision depth (> 20), deeply nested directory hierarchies, non-ASCII/Unicode
   filenames (Devanagari, Tamil, Japanese, Arabic, Emojis), simultaneous note & attachment collisions on archive,
   and correction log schema resilience.
4. sarvam_llm.py, sarvam_ocr.py, sarvam_text.py, search.py: Direct HTTP mode, 401/500 errors, conversational
   padding with nested braces, hard sentence splitting, language ID fallbacks, and resilient web search.
5. models.py, render.py, prompts.py, main.py: Schema validation limits, YAML quoting subtleties, abbreviation
   sentence splitting, and application wiring.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from pydantic import ValidationError
import pytest
from telegram import InlineKeyboardMarkup, Message, PhotoSize, Update
from telegram.error import BadRequest
from telegram.ext import Application, ContextTypes

from ingest.adapters.ports import LlmPort, OcrPort, SearchPort, TextPort, VaultPort
from ingest.adapters.sarvam_llm import SarvamLlm
from ingest.adapters.sarvam_ocr import SarvamOcr
from ingest.adapters.sarvam_text import SarvamText
from ingest.adapters.search import SearchAdapter, TavilySearch
from ingest.adapters.vault import FileVault
from ingest.bot import (
    MAX_FILE_SIZE,
    _callback_token,
    _make_callback_data,
    _read_category_from_note,
    _resolve_inbox_file,
    build_category_choice_keyboard,
    build_proposal_keyboard,
    on_callback,
    on_document,
    on_photo,
    on_text,
    setup_bot_handlers,
    start_command,
)
from ingest.config import Settings
from ingest.main import build_deps, main
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
from ingest.prompts import classify_prompt, extract_prompt, project_prompt
from ingest.render import (
    _format_yaml_scalar,
    _needs_yaml_quotes,
    build_frontmatter,
    dump_frontmatter,
    filename_for,
    get_usage,
    render_body,
    render_markdown,
    slug,
    split_sentences,
)
from tests.fakes import FakeLlm, FakeOcr, FakeSearch, FakeText


def sync_run(coro):
    """Executes a coroutine chain synchronously without socketpair / loopback overhead."""
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
def temp_vault() -> FileVault:
    temp_dir = tempfile.mkdtemp(prefix="adv_vault_")
    vault = FileVault(temp_dir)
    return vault


# ===========================================================================
# SECTION 1: Pipeline Adversarial Edge Cases (pipeline.py)
# ===========================================================================

def test_pipeline_normalise_missing_or_corrupt_files(temp_vault: FileVault):
    """Verify normalise rejects media captures with missing or non-existent files."""
    deps = Deps(
        llm=FakeLlm(),
        ocr=FakeOcr(),
        text=FakeText(),
        search=FakeSearch(),
        vault=temp_vault,
    )

    async def _test():
        # Document with no local path
        cap_no_path = Capture(
            chat_id=1,
            message_id=1,
            kind=CaptureKind.DOCUMENT,
            local_path=None,
        )
        with pytest.raises(IngestError) as exc_info:
            await normalise(cap_no_path, deps)
        assert exc_info.value.stage == "ocr"

        # Document with non-existent local path
        cap_ghost = Capture(
            chat_id=1,
            message_id=2,
            kind=CaptureKind.PHOTO,
            local_path=Path("non_existent_photo_12345.jpg"),
        )
        ocr_failing = FakeOcr(text=FileNotFoundError("Missing on disk"))
        deps_failing = Deps(
            llm=FakeLlm(),
            ocr=ocr_failing,
            text=FakeText(),
            search=FakeSearch(),
            vault=temp_vault,
        )
        with pytest.raises(IngestError) as exc_info2:
            await normalise(cap_ghost, deps_failing)
        assert exc_info2.value.stage == "ocr"

    sync_run(_test())


def test_pipeline_normalise_empty_inputs_and_whitespace(temp_vault: FileVault):
    """Verify normalise cleanly handles empty strings, None text, and whitespace captions."""
    deps = Deps(
        llm=FakeLlm(),
        ocr=FakeOcr(text=""),
        text=FakeText(),
        search=FakeSearch(),
        vault=temp_vault,
    )

    async def _test():
        # Text capture with None text and whitespace caption
        cap1 = Capture(
            chat_id=1,
            message_id=1,
            kind=CaptureKind.TEXT,
            text=None,
            caption="   \n\t  ",
        )
        orig, trans = await normalise(cap1, deps)
        assert orig == ""
        assert trans is None

        # Text capture with pure whitespace text
        cap2 = Capture(
            chat_id=1,
            message_id=2,
            kind=CaptureKind.TEXT,
            text="   ",
            caption="Valid Caption",
        )
        orig2, trans2 = await normalise(cap2, deps)
        assert orig2 == "Valid Caption"
        assert trans2 is None

        # Media capture with empty OCR text but non-empty caption
        doc_path = temp_vault.root / "inbox" / "empty.txt"
        doc_path.write_text("", encoding="utf-8")
        cap3 = Capture(
            chat_id=1,
            message_id=3,
            kind=CaptureKind.DOCUMENT,
            local_path=doc_path,
            caption="Receipt from mess",
        )
        orig3, trans3 = await normalise(cap3, deps)
        assert orig3 == "Receipt from mess"
        assert trans3 is None

    sync_run(_test())


def test_pipeline_normalise_network_drops_and_error_propagation(temp_vault: FileVault):
    """Verify network timeouts and pre-existing IngestErrors are properly handled."""
    async def _test():
        # OCR raises network TimeoutError -> wrapped in IngestError("ocr")
        ocr_timeout = FakeOcr(text=httpx.ConnectTimeout("Network dropped during OCR"))
        deps1 = Deps(
            llm=FakeLlm(),
            ocr=ocr_timeout,
            text=FakeText(),
            search=FakeSearch(),
            vault=temp_vault,
        )
        cap = Capture(chat_id=1, message_id=1, kind=CaptureKind.DOCUMENT, local_path=Path("dummy.pdf"))
        with pytest.raises(IngestError) as exc1:
            await normalise(cap, deps1)
        assert exc1.value.stage == "ocr"

        # OCR raises IngestError directly -> propagated directly without re-wrapping
        ocr_direct = FakeOcr(text=IngestError("ocr", "Internal OCR Failure"))
        deps2 = Deps(
            llm=FakeLlm(),
            ocr=ocr_direct,
            text=FakeText(),
            search=FakeSearch(),
            vault=temp_vault,
        )
        with pytest.raises(IngestError) as exc2:
            await normalise(cap, deps2)
        assert exc2.value.stage == "ocr"
        assert str(exc2.value.cause) == "Internal OCR Failure"

        # Text detection raises error on TEXT capture -> wrapped in stage "download"
        text_failing = MagicMock()
        text_failing.detect = AsyncMock(side_effect=ConnectionResetError("Socket reset"))
        deps3 = Deps(
            llm=FakeLlm(),
            ocr=FakeOcr(),
            text=text_failing,
            search=FakeSearch(),
            vault=temp_vault,
        )
        cap_text = Capture(chat_id=1, message_id=2, kind=CaptureKind.TEXT, text="Hello World")
        with pytest.raises(IngestError) as exc3:
            await normalise(cap_text, deps3)
        assert exc3.value.stage == "download"

    sync_run(_test())


def test_pipeline_classify_schema_mismatch_and_repair_flow(temp_vault: FileVault):
    """Verify classify triggers automatic repair when initial LLM output violates schema."""
    async def _test():
        cap = Capture(chat_id=1, message_id=1, kind=CaptureKind.TEXT, text="Project Idea")

        call_count = 0
        valid_payload = {
            "category": "possible-projects",
            "title": "Automated Plant Watering",
            "summary": "An automated system to water plants using soil moisture sensors.",
            "tags": ["hardware", "plants"],
            "document_type": None,
            "issuer": None,
            "issued_date": None,
            "reference_ids": [],
        }

        class RepairingLlm:
            async def complete_json(self, system: str, user: str) -> dict[str, Any]:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # Invalid: missing category
                    return {"title": "Missing Category", "summary": "No category here"}
                # Repair call
                assert "Previous response was invalid" in user
                return valid_payload

            async def complete_text(self, system: str, user: str) -> str:
                return "brief"

        deps_repair = Deps(
            llm=RepairingLlm(),
            ocr=FakeOcr(),
            text=FakeText(),
            search=FakeSearch(),
            vault=temp_vault,
        )

        cls = await classify(cap, "Project Idea", None, deps_repair)
        assert call_count == 2
        assert cls.category == Category.PROJECT
        assert cls.title == "Automated Plant Watering"

        # Scenario B: Both initial and repair JSON violate schema -> raises IngestError("classify")
        call_count_fail = 0

        class UnrecoverableLlm:
            async def complete_json(self, system: str, user: str) -> dict[str, Any]:
                nonlocal call_count_fail
                call_count_fail += 1
                return {"title": "Still Invalid", "tags": "not-a-list"}

            async def complete_text(self, system: str, user: str) -> str:
                return ""

        deps_fail = Deps(
            llm=UnrecoverableLlm(),
            ocr=FakeOcr(),
            text=FakeText(),
            search=FakeSearch(),
            vault=temp_vault,
        )

        with pytest.raises(IngestError) as exc_info:
            await classify(cap, "Project Idea", None, deps_fail)
        assert exc_info.value.stage == "classify"
        assert call_count_fail == 2

    sync_run(_test())


def test_pipeline_expand_project_adversarial_paths(temp_vault: FileVault):
    """Verify expand_project survives empty search results and wraps external search failures."""
    async def _test():
        deps_empty_search = Deps(
            llm=FakeLlm(text="## Description\nFallback brief."),
            ocr=FakeOcr(),
            text=FakeText(),
            search=FakeSearch(results=[]),
            vault=temp_vault,
        )
        brief = await expand_project("Solar Drone", deps_empty_search)
        assert "Fallback brief" in brief

        search_failing = MagicMock()
        search_failing.search = AsyncMock(side_effect=RuntimeError("Search API Quota Exceeded"))
        deps_failing = Deps(
            llm=FakeLlm(),
            ocr=FakeOcr(),
            text=FakeText(),
            search=search_failing,
            vault=temp_vault,
        )
        with pytest.raises(IngestError) as exc_info:
            await expand_project("Solar Drone", deps_failing)
        assert exc_info.value.stage == "search"

        search_direct = MagicMock()
        search_direct.search = AsyncMock(side_effect=IngestError("search", "Pre-wrapped failure"))
        deps_direct = Deps(
            llm=FakeLlm(),
            ocr=FakeOcr(),
            text=FakeText(),
            search=search_direct,
            vault=temp_vault,
        )
        with pytest.raises(IngestError) as exc_info2:
            await expand_project("Solar Drone", deps_direct)
        assert exc_info2.value.stage == "search"
        assert str(exc_info2.value.cause) == "Pre-wrapped failure"

    sync_run(_test())


def test_pipeline_build_note_attachment_variations(temp_vault: FileVault):
    """Verify build_note correctly assigns or omits attachment paths based on Capture kind."""
    deps = Deps(
        llm=FakeLlm(),
        ocr=FakeOcr(),
        text=FakeText(),
        search=FakeSearch(),
        vault=temp_vault,
    )
    cls = Classification(
        category=Category.COLLEGE,
        title="BTech Transcript",
        summary="Final transcript for degree.",
        tags=["college", "grades"],
        document_type="transcript",
        issuer="University",
        issued_date="2026-06-01",
        reference_ids=["TR-101"],
    )

    async def _test():
        # 1. TEXT capture with filename: attachment MUST be None
        cap_text = Capture(
            chat_id=1,
            message_id=1,
            kind=CaptureKind.TEXT,
            filename="notes.txt",
            text="Transcript details",
        )
        note_text = await build_note(cap_text, cls, deps)
        assert "attachment" not in note_text.frontmatter

        # 2. DOCUMENT capture with local_path: attachment has category folder
        cap_doc = Capture(
            chat_id=1,
            message_id=2,
            kind=CaptureKind.DOCUMENT,
            local_path=Path("/tmp/transcript.pdf"),
            filename="transcript.pdf",
        )
        note_doc = await build_note(cap_doc, cls, deps)
        assert note_doc.frontmatter["attachment"] == "attachments/college-documents/transcript.pdf"

        # 3. DOCUMENT capture with filename only (no local_path): attachment preserved
        cap_fn_only = Capture(
            chat_id=1,
            message_id=3,
            kind=CaptureKind.DOCUMENT,
            filename="scanned_grade.png",
        )
        note_fn = await build_note(cap_fn_only, cls, deps)
        assert note_fn.frontmatter["attachment"] == "attachments/college-documents/scanned_grade.png"

    sync_run(_test())


def test_pipeline_ingest_atomic_transaction_cleanup_on_attachment_failure(temp_vault: FileVault):
    """Verify that if store_attachment fails during ingest, written inbox note is unlinked."""
    temp_dir = Path(tempfile.mkdtemp(prefix="adv_attach_test_"))
    try:
        sample_doc = temp_dir / "sample.pdf"
        sample_doc.write_bytes(b"%PDF-1.4 dummy binary content")

        real_vault = temp_vault
        failing_vault = MagicMock(wraps=real_vault)
        failing_vault.store_attachment.side_effect = OSError("Disk full during attachment store")

        deps = Deps(
            llm=FakeLlm(
                payload={
                    "category": "college-documents",
                    "title": "Semester Grade Report",
                    "summary": "Grade report for semester.",
                    "tags": ["college"],
                    "document_type": "marksheet",
                    "issuer": "College",
                    "issued_date": "2026-01-01",
                    "reference_ids": [],
                }
            ),
            ocr=FakeOcr(text="Marksheet Text"),
            text=FakeText(),
            search=FakeSearch(),
            vault=failing_vault,
        )

        cap = Capture(
            chat_id=1,
            message_id=1,
            kind=CaptureKind.DOCUMENT,
            local_path=sample_doc,
            filename="sample.pdf",
        )

        async def _test():
            with pytest.raises(IngestError) as exc_info:
                await ingest(cap, deps)
            assert exc_info.value.stage == "write"

            inbox_files = list((real_vault.root / "inbox").glob("*.md"))
            assert len(inbox_files) == 0, f"Expected inbox to be empty after rollback, found: {inbox_files}"

        sync_run(_test())
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_pipeline_file_proposal_redirection_and_attachment_relocation(temp_vault: FileVault):
    """Verify file_proposal relocates attachment and logs correction when category changes."""
    note = Note(
        frontmatter={
            "title": "Robotics Blueprint",
            "category": "college-documents",
            "status": "inbox",
            "attachment": "attachments/college-documents/blueprint.png",
        },
        body="## Body content\nBlueprint for robot.",
        filename="2026-11-03 - Robotics-Blueprint.md",
    )
    written_note = temp_vault.write_note(note, folder="inbox")

    att_dir = temp_vault.root / "attachments" / "college-documents"
    att_dir.mkdir(parents=True, exist_ok=True)
    att_file = att_dir / "blueprint.png"
    att_file.write_bytes(b"\x89PNG dummy blueprint data")

    proposal = Proposal(
        note=note,
        category=Category.COLLEGE,
        preview="*Robotics Blueprint*",
    )

    deps = Deps(
        llm=FakeLlm(),
        ocr=FakeOcr(),
        text=FakeText(),
        search=FakeSearch(),
        vault=temp_vault,
    )

    dest = file_proposal(proposal, Category.PROJECT, deps)

    assert dest.exists()
    assert dest.parent == temp_vault.root / "possible-projects"
    assert not written_note.exists()

    new_att = temp_vault.root / "attachments" / "possible-projects" / "blueprint.png"
    assert new_att.exists()
    assert not att_file.exists(), "Old attachment must be unlinked after relocation"

    corr_file = temp_vault.root / "_corrections.jsonl"
    assert corr_file.exists()
    corrections = [json.loads(line) for line in corr_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(corrections) == 1
    assert corrections[0]["original_category"] == "college-documents"
    assert corrections[0]["chosen_category"] == "possible-projects"


# ===========================================================================
# SECTION 2: Telegram Bot Imperative Shell Adversarial Edge Cases (bot.py)
# ===========================================================================

def test_bot_callback_data_boundary_conditions():
    """Verify _make_callback_data strictly obeys Telegram's 64-byte payload limit."""
    prefix = "approve"
    ident_56 = "a" * 56
    data_64 = _make_callback_data(prefix, ident_56)
    assert len(data_64.encode("utf-8")) == 64
    assert data_64 == f"{prefix}:{ident_56}"

    ident_57 = "a" * 57
    data_short = _make_callback_data(prefix, ident_57)
    assert len(data_short.encode("utf-8")) == 24
    assert len(data_short.encode("utf-8")) <= 64
    token = _callback_token(ident_57)
    assert data_short == f"{prefix}:{token}"

    unicode_title = "परीक्षा-परिणाम-२०२६-इंजीनियरिंग-भौतिकी-" * 10
    unicode_data = _make_callback_data("choose:college-documents", unicode_title)
    assert len(unicode_data.encode("utf-8")) <= 64

    long_prefix = "p" * 70
    fallback_data = _make_callback_data(long_prefix, "ident")
    assert len(fallback_data.encode("utf-8")) <= 64


def test_bot_duplicate_and_simultaneous_callbacks(temp_vault: FileVault):
    """Verify simultaneous duplicate callback queries do not crash the bot."""
    note = Note(
        frontmatter={"title": "Rapid Tap Test", "status": "inbox", "category": "reference"},
        body="Body text",
        filename="2026-11-03 - Rapid-Tap.md",
    )
    written = temp_vault.write_note(note, folder="inbox")
    proposal = Proposal(note=note, category=Category.REFERENCE, preview="*Rapid Tap Test*")

    deps = Deps(
        llm=FakeLlm(),
        ocr=FakeOcr(),
        text=FakeText(),
        search=FakeSearch(),
        vault=temp_vault,
    )

    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    context.application = MagicMock()
    context.application.bot_data = {"deps": deps}
    context.chat_data = {note.filename: proposal}

    query1 = MagicMock()
    query1.data = f"approve:{note.filename}"
    query1.answer = AsyncMock()
    query1.edit_message_text = AsyncMock()

    update1 = MagicMock()
    update1.callback_query = query1

    # First callback fires
    sync_run(on_callback(update1, context))
    query1.answer.assert_awaited_once()
    assert "✅ Filed to `reference/`" in query1.edit_message_text.call_args[0][0]

    # Proposal removed or note already filed
    context.chat_data.pop(note.filename, None)

    # Second callback arrives immediately after (duplicate tap)
    query2 = MagicMock()
    query2.data = f"approve:{note.filename}"
    query2.answer = AsyncMock()
    query2.edit_message_text = AsyncMock()

    update2 = MagicMock()
    update2.callback_query = query2

    sync_run(on_callback(update2, context))
    query2.answer.assert_awaited_once()
    assert "Note filed" in query2.edit_message_text.call_args[0][0]


def test_bot_token_hash_resolution_without_chat_data(temp_vault: FileVault):
    """Verify _resolve_inbox_file recovers inbox files using hash token after bot restart."""
    long_filename = "2026-11-03 - Extremely-Long-Title-That-Forces-A-Hash-Token-In-Telegram-Callbacks.md"
    note = Note(
        frontmatter={"title": "Long Title", "status": "inbox", "category": "possible-projects"},
        body="Body content",
        filename=long_filename,
    )
    written = temp_vault.write_note(note, folder="inbox")
    assert written.exists()

    token = _callback_token(long_filename)
    inbox_dir = temp_vault.root / "inbox"

    resolved = _resolve_inbox_file(inbox_dir, token)
    assert resolved is not None
    assert resolved.resolve() == written.resolve()

    ghost = _resolve_inbox_file(inbox_dir, "0000000000000000")
    assert ghost is None

    missing_dir = temp_vault.root / "missing_folder"
    assert _resolve_inbox_file(missing_dir, token) is None


def test_bot_malformed_update_structures(temp_vault: FileVault):
    """Verify handlers handle None and missing fields in Telegram Update objects cleanly."""
    deps = Deps(
        llm=FakeLlm(),
        ocr=FakeOcr(),
        text=FakeText(),
        search=FakeSearch(),
        vault=temp_vault,
    )
    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    context.application = MagicMock()
    context.application.bot_data = {"deps": deps}
    context.chat_data = {}

    async def _test():
        empty_update = MagicMock()
        empty_update.effective_message = None
        empty_update.callback_query = None

        await start_command(empty_update, context)
        await on_document(empty_update, context)
        await on_photo(empty_update, context)
        await on_text(empty_update, context)
        await on_callback(empty_update, context)

        doc_update_empty = MagicMock()
        doc_update_empty.effective_message.document = None
        await on_document(doc_update_empty, context)

        photo_update_empty = MagicMock()
        photo_update_empty.effective_message.photo = []
        await on_photo(photo_update_empty, context)

        text_update_empty = MagicMock()
        text_update_empty.effective_message.text = None
        await on_text(text_update_empty, context)

        query_bad = MagicMock()
        query_bad.data = "action_without_colon"
        query_bad.answer = AsyncMock()
        cb_update_bad = MagicMock()
        cb_update_bad.callback_query = query_bad
        await on_callback(cb_update_bad, context)
        query_bad.answer.assert_awaited_once()

    sync_run(_test())


def test_bot_media_size_validation_boundaries(temp_vault: FileVault):
    """Verify document and photo size limit checks (> 20 MB)."""
    deps = Deps(
        llm=FakeLlm(),
        ocr=FakeOcr(),
        text=FakeText(),
        search=FakeSearch(),
        vault=temp_vault,
    )
    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    context.application = MagicMock()
    context.application.bot_data = {"deps": deps}
    context.chat_data = {}

    async def _test():
        ack_doc = MagicMock()
        ack_doc.edit_text = AsyncMock()

        msg_doc = MagicMock()
        msg_doc.reply_text = AsyncMock(return_value=ack_doc)
        msg_doc.document.file_size = 20 * 1024 * 1024 + 1

        update_doc = MagicMock()
        update_doc.effective_message = msg_doc

        await on_document(update_doc, context)
        ack_doc.edit_text.assert_awaited_once()
        assert "exceeds the 20 MB limit" in ack_doc.edit_text.call_args[0][0]

        ack_photo = MagicMock()
        ack_photo.edit_text = AsyncMock()

        item_photo = MagicMock()
        item_photo.file_size = 21 * 1024 * 1024
        msg_photo = MagicMock()
        msg_photo.reply_text = AsyncMock(return_value=ack_photo)
        msg_photo.photo = [item_photo]

        update_photo = MagicMock()
        update_photo.effective_message = msg_photo

        await on_photo(update_photo, context)
        ack_photo.edit_text.assert_awaited_once()
        assert "exceeds the 20 MB limit" in ack_photo.edit_text.call_args[0][0]

    sync_run(_test())


def test_bot_markdown_parsing_error_fallback(temp_vault: FileVault):
    """Verify bot falls back to parse_mode=None when Telegram rejects unescaped Markdown."""
    deps = Deps(
        llm=FakeLlm(
            payload={
                "category": "reference",
                "title": "Unescaped *Markdown* & [Entities]",
                "summary": "Summary with unclosed *asterisk and _underscore.",
                "tags": ["testing"],
                "document_type": None,
                "issuer": None,
                "issued_date": None,
                "reference_ids": [],
            }
        ),
        ocr=FakeOcr(),
        text=FakeText(),
        search=FakeSearch(),
        vault=temp_vault,
    )
    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    context.application = MagicMock()
    context.application.bot_data = {"deps": deps}
    context.chat_data = {}

    async def _test():
        ack = MagicMock()
        ack.edit_text = AsyncMock(
            side_effect=[
                BadRequest("Can't parse entities: character '*' is reserved"),
                None,
            ]
        )

        msg = MagicMock()
        msg.reply_text = AsyncMock(return_value=ack)
        msg.text = "Unescaped Markdown Test Message"
        msg.chat_id = 123
        msg.message_id = 456

        update = MagicMock()
        update.effective_message = msg

        await on_text(update, context)

        assert ack.edit_text.await_count == 2
        assert ack.edit_text.await_args_list[0].kwargs.get("parse_mode") == "Markdown"
        assert ack.edit_text.await_args_list[1].kwargs.get("parse_mode") is None

    sync_run(_test())


def test_bot_typed_error_reporting(temp_vault: FileVault):
    """Verify typed IngestError produces helpful stage error messages."""
    async def _test():
        stages = ["download", "ocr", "classify", "search", "write"]
        for stage in stages:
            ack = MagicMock()
            ack.edit_text = AsyncMock()
            msg = MagicMock()
            msg.reply_text = AsyncMock(return_value=ack)
            msg.text = f"Test {stage}"
            msg.chat_id = 1
            msg.message_id = 1
            update = MagicMock()
            update.effective_message = msg

            failing_llm = FakeLlm(payload=IngestError(stage, f"{stage} crashed"))
            deps = Deps(
                llm=failing_llm,
                ocr=FakeOcr(),
                text=FakeText(),
                search=FakeSearch(),
                vault=temp_vault,
            )
            context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
            context.application = MagicMock()
            context.application.bot_data = {"deps": deps}
            context.chat_data = {}

            await on_text(update, context)
            ack.edit_text.assert_awaited_once()
            assert f"Couldn't process that — {stage} failed. Send it again." in ack.edit_text.call_args[0][0]

    sync_run(_test())


# ===========================================================================
# SECTION 3: Vault Adapter Extreme Adversarial Edge Cases (vault.py)
# ===========================================================================

def test_vault_extreme_collision_depth_greater_than_20(temp_vault: FileVault):
    """Stress test FileVault with 25 sequential collisions across write, store, move, archive."""
    depth = 25
    stem = "2026-11-03 - Extreme-Collision"

    # 1. Sequential write collisions: base, -2, -3, ..., -25
    written_paths = []
    for i in range(1, depth + 1):
        note = Note(
            frontmatter={"title": "Extreme Collision", "status": "inbox", "ver": i},
            body=f"Content ver {i}",
            filename=f"{stem}.md",
        )
        p = temp_vault.write_note(note, folder="inbox")
        written_paths.append(p)

    assert written_paths[0].name == f"{stem}.md"
    for i in range(2, depth + 1):
        assert written_paths[i - 1].name == f"{stem}-{i}.md"
        assert written_paths[i - 1].exists()

    # 2. Sequential store_attachment collisions: base, -2, ..., -25
    dummy_att = temp_vault.root / "inbox" / "avatar.png"
    dummy_att.write_bytes(b"\x89PNG dummy avatar")
    stored_paths = []
    for i in range(depth):
        p = temp_vault.store_attachment(dummy_att, category="reference", filename="avatar.png")
        stored_paths.append(p)

    assert stored_paths[0].name == "avatar.png"
    for i in range(2, depth + 1):
        assert stored_paths[i - 1].name == f"avatar-{i}.png"
        assert stored_paths[i - 1].exists()

    # 3. Move collisions: Pre-populate destination with 25 files, then move one more -> gets -26
    dest_dir = temp_vault.root / "college-documents"
    base_col = "2026-11-03 - Subject"
    (dest_dir / f"{base_col}.md").write_text("v1", encoding="utf-8")
    for i in range(2, 26):
        (dest_dir / f"{base_col}-{i}.md").write_text(f"v{i}", encoding="utf-8")

    src_to_move = temp_vault.root / "inbox" / f"{base_col}.md"
    src_to_move.write_text("inbox file", encoding="utf-8")

    moved = temp_vault.move_note(src_to_move, folder="college-documents")
    assert moved.name == f"{base_col}-26.md"
    assert moved.exists()


def test_vault_deeply_nested_paths_and_non_ascii_filenames(temp_vault: FileVault):
    """Verify FileVault works across 5-level nested paths and non-ASCII Unicode filenames."""
    deep_folder = "college-documents/ug/sem-4/exams/results"

    titles = [
        "2026-11-03 - इंजीनियरिंग-भौतिकी-अंकतालिका.md",
        "2026-11-03 - தேர்வு-முடிவுகள்.md",
        "2026-11-03 - プロジェクト概要.md",
        "2026-11-03 - وثيقة-الجامعة.md",
        "2026-11-03 - 🚀 Super-Project #1 (Draft).md",
    ]

    for title in titles:
        note = Note(
            frontmatter={"title": title, "status": "inbox"},
            body=f"Body for {title}",
            filename=title,
        )
        written = temp_vault.write_note(note, folder=deep_folder)
        assert written.exists()
        assert (temp_vault.root / deep_folder).is_dir()
        content = written.read_text(encoding="utf-8")
        assert f"Body for {title}" in content

        # Moving to folder="_archive" strictly updates status to "archived"
        archived = temp_vault.move_note(written, folder="_archive")
        assert archived.exists()
        assert not written.exists()
        assert "status: archived" in archived.read_text(encoding="utf-8")


def test_vault_simultaneous_note_and_attachment_collisions_on_archive(temp_vault: FileVault):
    """Verify archive_note increments collisions on both note and attachment simultaneously."""
    archive_dir = temp_vault.root / "_archive"
    archive_att_dir = archive_dir / "attachments"

    (archive_dir / "Exam.md").write_text("Old 1", encoding="utf-8")
    (archive_dir / "Exam-2.md").write_text("Old 2", encoding="utf-8")

    (archive_att_dir / "scan.pdf").write_bytes(b"Old Att 1")
    (archive_att_dir / "scan-2.pdf").write_bytes(b"Old Att 2")

    src_note = temp_vault.root / "inbox" / "Exam.md"
    src_note.write_text("---\ntitle: Exam\nstatus: inbox\n---\n\n## Body\nContent.", encoding="utf-8")

    src_att = temp_vault.root / "inbox" / "scan.pdf"
    src_att.write_bytes(b"Fresh PDF Content")

    dest_note, dest_att = temp_vault.archive_note(src_note, src_att)

    assert dest_note.name == "Exam-3.md"
    assert dest_att is not None
    assert dest_att.name == "scan-3.pdf"
    assert dest_note.exists()
    assert dest_att.exists()
    assert not src_note.exists()
    assert not src_att.exists()
    assert "status: archived" in dest_note.read_text(encoding="utf-8")


def test_vault_append_correction_schema_flexibility(temp_vault: FileVault):
    """Verify append_correction accepts dicts, kwargs, and handles missing timestamps."""
    temp_vault.append_correction({"original": "college-documents", "chosen": "reference"})

    custom_ts = "2026-01-01T00:00:00Z"
    temp_vault.append_correction({"original": "reference", "chosen": "projects", "timestamp": custom_ts})

    temp_vault.append_correction(
        original="college-documents",
        chosen="possible-projects",
        note_title="इंजीनियरिंग प्रोजेक्ट",
        user_id=12345,
    )

    corr_file = temp_vault.root / "_corrections.jsonl"
    lines = [json.loads(line) for line in corr_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 3

    assert "timestamp" in lines[0]
    assert lines[1]["timestamp"] == custom_ts
    assert lines[2]["note_title"] == "इंजीनियरिंग प्रोजेक्ट"
    assert lines[2]["user_id"] == 12345


# ===========================================================================
# SECTION 4: Sarvam AI & Search Adapters (sarvam_llm, ocr, text, search)
# ===========================================================================

def test_sarvam_llm_direct_http_and_error_handling():
    """Verify SarvamLlm direct HTTP completions and repair retry under network errors."""
    async def _test():
        llm_http = SarvamLlm(client=None, api_key="sk-test", base_url="https://api.test")

        mock_resp_json = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "category": "reference",
                            "title": "HTTP Completion Title",
                            "summary": "Summary via direct HTTP call.",
                            "tags": ["http", "direct"],
                        })
                    }
                }
            ]
        }

        # Mock httpx.AsyncClient context manager directly
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_instance = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client_instance

            mock_res = MagicMock()
            mock_res.raise_for_status = MagicMock()
            mock_res.json.return_value = mock_resp_json
            mock_client_instance.post.return_value = mock_res

            res = await llm_http.complete_json("system", "user")
            assert res["title"] == "HTTP Completion Title"

        # 401 Unauthorized -> raises IngestError("classify")
        with patch("httpx.AsyncClient") as mock_client_cls_err:
            mock_client_instance_err = AsyncMock()
            mock_client_cls_err.return_value.__aenter__.return_value = mock_client_instance_err

            mock_err_res = MagicMock()
            mock_err_res.raise_for_status.side_effect = httpx.HTTPStatusError(
                "401 Unauthorized", request=MagicMock(), response=MagicMock(status_code=401)
            )
            mock_client_instance_err.post.return_value = mock_err_res

            with pytest.raises(IngestError) as exc_info:
                await llm_http.complete_json("system", "user")
            assert exc_info.value.stage == "classify"

        # Conversational padding with complex nested braces
        raw_conversational = (
            "Here is the classification you requested:\n\n"
            "{\n"
            '  "category": "possible-projects",\n'
            '  "title": "Smart Irrigation",\n'
            '  "summary": "IoT based irrigation {with sensors} inside.",\n'
            '  "tags": ["iot", "irrigation"]\n'
            "}\n\nHope this helps!"
        )
        parsed = SarvamLlm._parse_and_extract_json(raw_conversational)
        assert parsed["title"] == "Smart Irrigation"

        # Unrecoverable junk -> raises ValueError
        with pytest.raises(ValueError):
            SarvamLlm._parse_and_extract_json("I cannot fulfill this request as an AI assistant.")

    sync_run(_test())


def test_sarvam_ocr_edge_cases_and_mime_types():
    """Verify SarvamOcr MIME types, plain text fast paths, and fallback mechanisms."""
    temp_dir = Path(tempfile.mkdtemp(prefix="adv_ocr_"))
    try:
        ocr = SarvamOcr(api_key="sk-ocr-test", base_url="https://api.test")

        async def _test():
            # Fast path: .csv and .json files
            csv_path = temp_dir / "data.csv"
            csv_path.write_text("col1,col2\nval1,val2", encoding="utf-8")
            csv_text = await ocr.extract_text(csv_path)
            assert csv_text == "col1,col2\nval1,val2"

            json_path = temp_dir / "meta.json"
            json_path.write_text('{"id": 42}', encoding="utf-8")
            json_text = await ocr.extract_text(json_path)
            assert json_text == '{"id": 42}'

            # Non-existent file -> raises IngestError("ocr")
            with pytest.raises(IngestError) as exc_info:
                await ocr.extract_text(temp_dir / "non_existent.png")
            assert exc_info.value.stage == "ocr"

            # Corrupt PDF (not valid PDF bytes): fitz.open fails, calls _call_sarvam_ocr
            corrupt_pdf = temp_dir / "broken.pdf"
            corrupt_pdf.write_bytes(b"Not a real PDF header")

            with patch.object(ocr, "_call_sarvam_ocr", new_callable=AsyncMock) as mock_api:
                mock_api.return_value = "Recovered text from broken PDF"
                extracted = await ocr.extract_text(corrupt_pdf)
                assert extracted == "Recovered text from broken PDF"
                mock_api.assert_awaited_once_with(corrupt_pdf)

        sync_run(_test())
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_sarvam_text_edge_cases():
    """Verify SarvamText chunk boundary conditions, empty inputs, and language ID fallbacks."""
    st = SarvamText(chunk_chars=100)

    exact_100 = "A" * 100
    chunks = st.chunk_text(exact_100)
    assert len(chunks) == 1
    assert chunks[0] == exact_100

    hard_250 = "B" * 250
    chunks_hard = st.chunk_text(hard_250)
    assert len(chunks_hard) == 3
    assert chunks_hard[0] == "B" * 100
    assert chunks_hard[1] == "B" * 100
    assert chunks_hard[2] == "B" * 50
    assert "".join(chunks_hard) == hard_250

    async def _test():
        assert await st.detect("") == "en-IN"
        assert await st.detect("   \t\n  ") == "en-IN"

        st_faulty = SarvamText(client=MagicMock())
        st_faulty.client.text.identify_language.side_effect = RuntimeError("Service Down")
        assert await st_faulty.detect("Some English sentence") == "en-IN"

        st_http = SarvamText(api_key="sk-test", base_url="https://api.test")
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_inst = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_inst
            mock_res = MagicMock()
            mock_res.is_success = False
            mock_inst.post.return_value = mock_res
            lang = await st_http.detect("Testing language detection")
            assert lang == "en-IN"

        assert await st.to_english("Sample text", source="hi") == "Sample text"

    sync_run(_test())


def test_search_adapter_adversarial_queries_and_fallbacks():
    """Verify SearchAdapter handles missing keys, API errors, and empty results with strict grounding."""
    sa = SearchAdapter(api_key="", brave_api_key="")

    async def _test():
        # Missing API keys must raise IngestError("search")
        with pytest.raises(IngestError) as exc_info:
            await sa.search("AI & Robotics: Future! @2026", k=2)
        assert exc_info.value.stage == "search"
        assert "not configured" in str(exc_info.value)

        # HTTP error must raise IngestError("search")
        sa_tavily = SearchAdapter(api_key="tvly_live_key_123", provider="tavily")
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_inst = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_inst
            mock_inst.post.side_effect = httpx.HTTPStatusError("500 Error", request=MagicMock(), response=MagicMock())

            with pytest.raises(IngestError) as exc_info2:
                await sa_tavily.search("Quantum Computing", k=3)
            assert exc_info2.value.stage == "search"

        # Empty results return empty list [] without hallucination
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_inst = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_inst
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"results": []}
            mock_resp.raise_for_status = MagicMock()
            mock_inst.post.return_value = mock_resp

            empty_res = await sa_tavily.search("Very Obscure Term", k=3)
            assert empty_res == []

    sync_run(_test())


# ===========================================================================
# SECTION 5: Models, Render, Prompts & Main Entrypoint
# ===========================================================================

def test_models_and_render_edge_cases():
    """Verify field constraints, YAML quoting subtleties, and sentence splitter edge cases."""
    # 1. Classification title validator
    with pytest.raises(ValidationError):
        Classification(
            category=Category.REFERENCE,
            title="",
            summary="A summary.",
        )

    with pytest.raises(ValidationError):
        Classification(
            category=Category.REFERENCE,
            title="   \n\t  ",
            summary="A summary.",
        )

    cls = Classification(
        category=Category.REFERENCE,
        title="  First   Line \n Second \t Line  ",
        summary="A summary.",
    )
    assert cls.title == "First Line Second Line"

    # 2. YAML quoting edge cases
    assert _needs_yaml_quotes("true") is True
    assert _needs_yaml_quotes("false") is True
    assert _needs_yaml_quotes("null") is True
    assert _needs_yaml_quotes("~") is True
    assert _needs_yaml_quotes("12345") is True
    assert _needs_yaml_quotes("3.1415") is True
    assert _needs_yaml_quotes("- Item") is True
    assert _needs_yaml_quotes("Key: Value") is True
    assert _needs_yaml_quotes("") is True
    assert _needs_yaml_quotes("Plain Title") is False

    # 3. Frontmatter serialization with nested structures
    fm = {
        "title": "YAML Test",
        "empty_list": [],
        "empty_dict": {},
        "boolean_true": True,
        "boolean_false": False,
        "date_field": date(2026, 11, 3),
        "string_with_quotes": 'She said "Hello"',
    }
    dumped = dump_frontmatter(fm)
    assert "empty_list: []" in dumped
    assert "empty_dict: {}" in dumped
    assert "boolean_true: true" in dumped
    assert "boolean_false: false" in dumped
    assert "date_field: 2026-11-03" in dumped

    # 4. Sentence splitting across complex abbreviations
    text = (
        "Dr. A. P. J. Abdul Kalam was the 11th President of India. "
        "He studied physics at St. Joseph's College, Tiruchirappalli. "
        "E.g. his work in DRDO and ISRO inspired millions. "
        "Is this sentence split cleanly? Yes, absolutely!"
    )
    sentences = split_sentences(text)
    assert len(sentences) >= 3
    assert not any(s.strip() == "Dr." for s in sentences)
    assert not any(s.strip() == "E.g." for s in sentences)

    # 5. Document usage lookup
    assert "Academic document preserved in vault" in get_usage(None)
    assert "Academic document preserved in vault" in get_usage("")
    assert "Official comprehensive record" in get_usage("Transcript")
    assert "Official academic paperwork (UnknownType)" in get_usage("UnknownType")


def test_main_wiring_and_entrypoint():
    """Verify build_deps instantiates all concrete adapters and main handles missing config."""
    settings = Settings(
        telegram_bot_token="123456:FAKE-TOKEN-FOR-TESTING",
        sarvam_api_key="sk-fake-sarvam-key",
        search_api_key="fake-search-key",
        vault_path=Path(tempfile.mkdtemp()),
    )

    deps = build_deps(settings)
    assert isinstance(deps.llm, SarvamLlm)
    assert isinstance(deps.ocr, SarvamOcr)
    assert isinstance(deps.text, SarvamText)
    assert isinstance(deps.search, SearchAdapter)
    assert isinstance(deps.vault, FileVault)

    # Verify main handles configuration failure cleanly
    with patch("ingest.main.Settings", side_effect=Exception("Missing environment")):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
