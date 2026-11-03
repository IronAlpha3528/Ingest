"""Unit tests for Telegram bot shell, keyboards, and callback routing."""

from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import InlineKeyboardMarkup, Message, PhotoSize, Update
from telegram.ext import Application, ContextTypes

from ingest.adapters.vault import FileVault
from ingest.bot import (
    MAX_FILE_SIZE,
    _callback_token,
    _make_callback_data,
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
from ingest.main import build_deps
from ingest.models import (
    Capture,
    CaptureKind,
    Category,
    Classification,
    IngestError,
    Note,
    Proposal,
)
from ingest.pipeline import Deps
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


def make_fake_deps() -> Deps:
    td = tempfile.mkdtemp(prefix="ingest_test_vault_")
    vault = FileVault(td)
    return Deps(
        llm=FakeLlm(
            payload={
                "category": "college-documents",
                "title": "Engineering Physics Marksheet",
                "summary": "Official engineering physics semester marksheet.",
                "tags": ["college", "physics"],
                "document_type": "marksheet",
                "issuer": "State University",
                "issued_date": "2026-05-10",
                "reference_ids": ["PHY-01"],
            }
        ),
        ocr=FakeOcr("Physics Final Marks State University"),
        text=FakeText(),
        search=FakeSearch(),
        vault=vault,
    )


def test_build_proposal_keyboard():
    filename = "2026-11-03 - Final-Marksheet.md"
    kb = build_proposal_keyboard(filename)
    buttons = kb.inline_keyboard[0]
    assert len(buttons) == 3
    assert buttons[0].text == "Approve"
    assert buttons[0].callback_data == f"approve:{filename}"
    assert buttons[1].text == "Change folder"
    assert buttons[1].callback_data == f"change:{filename}"
    assert buttons[2].text == "Discard"
    assert buttons[2].callback_data == f"discard:{filename}"

    # Telegram callback data limit is 64 bytes
    for b in buttons:
        assert len(b.callback_data.encode("utf-8")) <= 64


def test_build_category_choice_keyboard():
    filename = "2026-11-03 - Sample-Note.md"
    kb = build_category_choice_keyboard(filename)
    rows = kb.inline_keyboard
    assert len(rows) == 3
    assert rows[0][0].text == "College Documents"
    assert rows[0][0].callback_data == f"choose:college-documents:{filename}"
    assert rows[1][0].text == "Possible Projects"
    assert rows[1][0].callback_data == f"choose:possible-projects:{filename}"
    assert rows[2][0].text == "Reference"
    assert rows[2][0].callback_data == f"choose:reference:{filename}"

    for r in rows:
        assert len(r[0].callback_data.encode("utf-8")) <= 64


def test_keyboards_with_long_and_unicode_filenames_strictly_fit_64_bytes():
    # Very long English filename (80+ characters)
    long_name = "2026-11-03 - An-Extremely-Long-Note-Filename-That-Easily-Exceeds-Telegram-Callback-Byte-Limit.md"
    kb_prop = build_proposal_keyboard(long_name)
    for b in kb_prop.inline_keyboard[0]:
        encoded = b.callback_data.encode("utf-8")
        assert len(encoded) <= 64

    kb_cat = build_category_choice_keyboard(long_name)
    for r in kb_cat.inline_keyboard:
        encoded = r[0].callback_data.encode("utf-8")
        assert len(encoded) <= 64

    # Unicode / Indic multi-byte characters
    indic_name = "2026-11-03 - अभियांत्रिकी-भौतिकी-अंतिम-अंकतालिका-विश्वविद्यालय-दस्तावेज़.md"
    kb_indic_prop = build_proposal_keyboard(indic_name)
    for b in kb_indic_prop.inline_keyboard[0]:
        encoded = b.callback_data.encode("utf-8")
        assert len(encoded) <= 64

    kb_indic_cat = build_category_choice_keyboard(indic_name)
    for r in kb_indic_cat.inline_keyboard:
        encoded = r[0].callback_data.encode("utf-8")
        assert len(encoded) <= 64


def test_setup_bot_handlers():
    deps = make_fake_deps()
    app = MagicMock(spec=Application)
    app.bot_data = {}

    setup_bot_handlers(app, deps)
    assert app.bot_data["deps"] == deps
    assert app.add_handler.call_count >= 5


def test_start_command():
    async def _test():
        update = MagicMock(spec=Update)
        message = MagicMock(spec=Message)
        message.reply_text = AsyncMock()
        update.effective_message = message
        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

        await start_command(update, context)
        assert message.reply_text.call_count == 1
        call_text = message.reply_text.call_args[0][0]
        assert "Welcome to Ingest" in call_text

        # Test safety when effective_message is None
        update_none = MagicMock(spec=Update)
        update_none.effective_message = None
        await start_command(update_none, context)

    sync_run(_test())


def test_on_text_success():
    deps = make_fake_deps()

    async def _test():
        update = MagicMock(spec=Update)
        message = MagicMock(spec=Message)
        message.chat_id = 12345
        message.message_id = 999
        message.text = "Build an automated hostel mess billing app"
        message.caption = None

        ack_msg = MagicMock(spec=Message)
        ack_msg.edit_text = AsyncMock()
        message.reply_text = AsyncMock(return_value=ack_msg)
        update.effective_message = message

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.application = MagicMock()
        context.application.bot_data = {"deps": deps}
        context.chat_data = {}

        await on_text(update, context)

        # 1. Fast receipt acknowledgment "Thinking…" sent first
        message.reply_text.assert_called_once_with("Thinking…")

        # 2. Preview message edited with proposal preview and keyboard
        assert ack_msg.edit_text.call_count == 1
        edit_kwargs = ack_msg.edit_text.call_args[1]
        assert edit_kwargs.get("parse_mode") == "Markdown"
        assert isinstance(edit_kwargs.get("reply_markup"), InlineKeyboardMarkup)

        # 3. Proposal saved in context.chat_data
        assert len(context.chat_data) >= 1

    sync_run(_test())


def test_on_text_markdown_parse_error_fallback():
    deps = make_fake_deps()

    async def _test():
        update = MagicMock(spec=Update)
        message = MagicMock(spec=Message)
        message.chat_id = 12345
        message.message_id = 999
        message.text = "A simple reference note with special symbols"
        message.caption = None

        ack_msg = MagicMock(spec=Message)
        # First call with parse_mode="Markdown" raises an exception (simulating BadRequest)
        # Second call with parse_mode=None succeeds
        ack_msg.edit_text = AsyncMock(side_effect=[Exception("BadRequest: Can't parse entities"), None])
        message.reply_text = AsyncMock(return_value=ack_msg)
        update.effective_message = message

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.application = MagicMock()
        context.application.bot_data = {"deps": deps}
        context.chat_data = {}

        await on_text(update, context)

        assert ack_msg.edit_text.call_count == 2
        fallback_kwargs = ack_msg.edit_text.call_args_list[1][1]
        assert fallback_kwargs.get("parse_mode") is None

    sync_run(_test())


def test_on_text_ingest_error():
    deps = make_fake_deps()

    async def _test():
        update = MagicMock(spec=Update)
        message = MagicMock(spec=Message)
        message.chat_id = 12345
        message.message_id = 999
        message.text = "Hostel app"
        message.caption = None

        ack_msg = MagicMock(spec=Message)
        ack_msg.edit_text = AsyncMock()
        message.reply_text = AsyncMock(return_value=ack_msg)
        update.effective_message = message

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.application = MagicMock()
        context.application.bot_data = {"deps": deps}
        context.chat_data = {}

        with patch("ingest.bot.ingest", AsyncMock(side_effect=IngestError("classify", "LLM rate limited"))):
            await on_text(update, context)

        ack_msg.edit_text.assert_called_once_with("Couldn't process that — classify failed. Send it again.")

    sync_run(_test())


def test_on_text_unexpected_error():
    deps = make_fake_deps()

    async def _test():
        update = MagicMock(spec=Update)
        message = MagicMock(spec=Message)
        message.chat_id = 12345
        message.message_id = 999
        message.text = "Hostel app"
        message.caption = None

        ack_msg = MagicMock(spec=Message)
        ack_msg.edit_text = AsyncMock()
        message.reply_text = AsyncMock(return_value=ack_msg)
        update.effective_message = message

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.application = MagicMock()
        context.application.bot_data = {"deps": deps}
        context.chat_data = {}

        with patch("ingest.bot.ingest", AsyncMock(side_effect=RuntimeError("Unexpected crash"))):
            await on_text(update, context)

        ack_msg.edit_text.assert_called_once_with("Couldn't process that — an unexpected error occurred. Send it again.")

    sync_run(_test())


def test_on_text_empty_does_nothing():
    deps = make_fake_deps()

    async def _test():
        update = MagicMock(spec=Update)
        update.effective_message = None
        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        await on_text(update, context)

        msg = MagicMock(spec=Message)
        msg.text = None
        update.effective_message = msg
        await on_text(update, context)

    sync_run(_test())


def test_on_document_success():
    deps = make_fake_deps()

    async def _test():
        update = MagicMock(spec=Update)
        message = MagicMock(spec=Message)
        message.chat_id = 12345
        message.message_id = 100
        message.caption = "Engineering Marksheet PDF"

        doc = MagicMock()
        doc.file_name = "physics_marksheet.pdf"
        doc.file_size = 500 * 1024  # 500 KB (well under 20 MB)

        tg_file = MagicMock()
        async def fake_download(path):
            Path(path).write_bytes(b"dummy-pdf-content")
        tg_file.download_to_drive = AsyncMock(side_effect=fake_download)
        doc.get_file = AsyncMock(return_value=tg_file)
        message.document = doc

        ack_msg = MagicMock(spec=Message)
        ack_msg.edit_text = AsyncMock()
        message.reply_text = AsyncMock(return_value=ack_msg)
        update.effective_message = message

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.application = MagicMock()
        context.application.bot_data = {"deps": deps}
        context.chat_data = {}

        await on_document(update, context)

        # 1. Fast receipt acknowledgment "Reading…"
        message.reply_text.assert_called_once_with("Reading…")

        # 2. File was downloaded
        assert tg_file.download_to_drive.call_count == 1

        # 3. Preview presented
        assert ack_msg.edit_text.call_count == 1
        edit_kwargs = ack_msg.edit_text.call_args[1]
        assert edit_kwargs.get("parse_mode") == "Markdown"

        # 4. Proposal stored in chat_data
        assert len(context.chat_data) >= 1

    sync_run(_test())


def test_on_document_oversized_rejected():
    deps = make_fake_deps()

    async def _test():
        update = MagicMock(spec=Update)
        message = MagicMock(spec=Message)
        message.chat_id = 12345
        message.message_id = 101
        message.caption = None

        doc = MagicMock()
        doc.file_name = "huge_video.mp4"
        doc.file_size = 25 * 1024 * 1024  # 25 MB > 20 MB limit
        doc.get_file = AsyncMock()
        message.document = doc

        ack_msg = MagicMock(spec=Message)
        ack_msg.edit_text = AsyncMock()
        message.reply_text = AsyncMock(return_value=ack_msg)
        update.effective_message = message

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.application = MagicMock()
        context.application.bot_data = {"deps": deps}

        await on_document(update, context)

        # Fast receipt sent, then edited with 20 MB limit message
        message.reply_text.assert_called_once_with("Reading…")
        ack_msg.edit_text.assert_called_once_with(
            "Couldn't process that — file exceeds the 20 MB limit. Please send a smaller file."
        )
        assert doc.get_file.call_count == 0

    sync_run(_test())


def test_on_document_ingest_error():
    deps = make_fake_deps()

    async def _test():
        update = MagicMock(spec=Update)
        message = MagicMock(spec=Message)
        message.chat_id = 12345
        message.message_id = 102
        message.caption = None

        doc = MagicMock()
        doc.file_name = "doc.pdf"
        doc.file_size = 1024
        tg_file = MagicMock()
        async def fake_download(path):
            Path(path).write_bytes(b"content")
        tg_file.download_to_drive = AsyncMock(side_effect=fake_download)
        doc.get_file = AsyncMock(return_value=tg_file)
        message.document = doc

        ack_msg = MagicMock(spec=Message)
        ack_msg.edit_text = AsyncMock()
        message.reply_text = AsyncMock(return_value=ack_msg)
        update.effective_message = message

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.application = MagicMock()
        context.application.bot_data = {"deps": deps}

        with patch("ingest.bot.ingest", AsyncMock(side_effect=IngestError("ocr", "Document OCR failed"))):
            await on_document(update, context)

        ack_msg.edit_text.assert_called_once_with("Couldn't process that — ocr failed. Send it again.")

    sync_run(_test())


def test_on_photo_success():
    deps = make_fake_deps()

    async def _test():
        update = MagicMock(spec=Update)
        message = MagicMock(spec=Message)
        message.chat_id = 12345
        message.message_id = 103
        message.caption = "Marksheet snapshot"

        # Provide small and large photo resolutions
        small_photo = MagicMock(spec=PhotoSize)
        small_photo.file_size = 100 * 1024
        large_photo = MagicMock(spec=PhotoSize)
        large_photo.file_size = 600 * 1024
        large_photo.file_unique_id = "uniq123"

        tg_file = MagicMock()
        async def fake_download(path):
            Path(path).write_bytes(b"jpeg-bytes")
        tg_file.download_to_drive = AsyncMock(side_effect=fake_download)
        large_photo.get_file = AsyncMock(return_value=tg_file)

        message.photo = [small_photo, large_photo]

        ack_msg = MagicMock(spec=Message)
        ack_msg.edit_text = AsyncMock()
        message.reply_text = AsyncMock(return_value=ack_msg)
        update.effective_message = message

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.application = MagicMock()
        context.application.bot_data = {"deps": deps}
        context.chat_data = {}

        await on_photo(update, context)

        # 1. Fast receipt acknowledgment "Reading…"
        message.reply_text.assert_called_once_with("Reading…")

        # 2. Largest photo was downloaded
        assert tg_file.download_to_drive.call_count == 1

        # 3. Preview presented
        assert ack_msg.edit_text.call_count == 1
        edit_kwargs = ack_msg.edit_text.call_args[1]
        assert edit_kwargs.get("parse_mode") == "Markdown"

    sync_run(_test())


def test_on_photo_oversized_rejected():
    deps = make_fake_deps()

    async def _test():
        update = MagicMock(spec=Update)
        message = MagicMock(spec=Message)
        message.chat_id = 12345
        message.message_id = 104
        message.caption = None

        large_photo = MagicMock(spec=PhotoSize)
        large_photo.file_size = 22 * 1024 * 1024  # > 20 MB
        large_photo.get_file = AsyncMock()
        message.photo = [large_photo]

        ack_msg = MagicMock(spec=Message)
        ack_msg.edit_text = AsyncMock()
        message.reply_text = AsyncMock(return_value=ack_msg)
        update.effective_message = message

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.application = MagicMock()
        context.application.bot_data = {"deps": deps}

        await on_photo(update, context)

        message.reply_text.assert_called_once_with("Reading…")
        ack_msg.edit_text.assert_called_once_with(
            "Couldn't process that — photo exceeds the 20 MB limit. Please send a smaller photo."
        )
        assert large_photo.get_file.call_count == 0

    sync_run(_test())


def test_on_photo_ingest_error():
    deps = make_fake_deps()

    async def _test():
        update = MagicMock(spec=Update)
        message = MagicMock(spec=Message)
        message.chat_id = 12345
        message.message_id = 105
        message.caption = None

        photo = MagicMock(spec=PhotoSize)
        photo.file_size = 1024
        photo.file_unique_id = "testphoto"
        tg_file = MagicMock()
        async def fake_download(path):
            Path(path).write_bytes(b"jpeg")
        tg_file.download_to_drive = AsyncMock(side_effect=fake_download)
        photo.get_file = AsyncMock(return_value=tg_file)
        message.photo = [photo]

        ack_msg = MagicMock(spec=Message)
        ack_msg.edit_text = AsyncMock()
        message.reply_text = AsyncMock(return_value=ack_msg)
        update.effective_message = message

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.application = MagicMock()
        context.application.bot_data = {"deps": deps}

        with patch("ingest.bot.ingest", AsyncMock(side_effect=IngestError("download", "Download socket timed out"))):
            await on_photo(update, context)

        ack_msg.edit_text.assert_called_once_with("Couldn't process that — download failed. Send it again.")

    sync_run(_test())


def test_on_callback_approve():
    deps = make_fake_deps()

    async def _test():
        note_name = "2026-11-03 - Final-Marksheet.md"
        # Write dummy note to inbox
        inbox_note = Note(
            frontmatter={"title": "Final Marksheet", "category": "college-documents", "status": "inbox"},
            body="## Summary\nMarksheet",
            filename=note_name,
        )
        deps.vault.write_note(inbox_note, "inbox")

        proposal = Proposal(
            note=inbox_note,
            category=Category.COLLEGE,
            preview="*Final Marksheet*",
        )

        update = MagicMock(spec=Update)
        query = MagicMock()
        query.data = f"approve:{note_name}"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        update.callback_query = query

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.application = MagicMock()
        context.application.bot_data = {"deps": deps}
        context.chat_data = {note_name: proposal}

        await on_callback(update, context)

        query.answer.assert_called_once()
        query.edit_message_text.assert_called_once()
        msg_text = query.edit_message_text.call_args[0][0]
        assert "Filed to `college-documents/`" in msg_text
        assert (deps.vault.root / "college-documents" / note_name).exists()

    sync_run(_test())


def test_on_callback_change_displays_category_keyboard():
    deps = make_fake_deps()

    async def _test():
        note_name = "2026-11-03 - Sample-Note.md"
        update = MagicMock(spec=Update)
        query = MagicMock()
        query.data = f"change:{note_name}"
        query.answer = AsyncMock()
        query.edit_message_reply_markup = AsyncMock()
        update.callback_query = query

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.application = MagicMock()
        context.application.bot_data = {"deps": deps}

        await on_callback(update, context)

        query.answer.assert_called_once()
        query.edit_message_reply_markup.assert_called_once()
        reply_markup = query.edit_message_reply_markup.call_args[1]["reply_markup"]
        assert isinstance(reply_markup, InlineKeyboardMarkup)
        # Should have rows for the 3 categories
        assert len(reply_markup.inline_keyboard) == 3

    sync_run(_test())


def test_on_callback_choose():
    deps = make_fake_deps()

    async def _test():
        note_name = "2026-11-03 - Multidisciplinary-Notes.md"
        inbox_note = Note(
            frontmatter={"title": "Multidisciplinary Notes", "category": "college-documents", "status": "inbox"},
            body="## Summary\nNotes",
            filename=note_name,
        )
        deps.vault.write_note(inbox_note, "inbox")

        proposal = Proposal(
            note=inbox_note,
            category=Category.COLLEGE,
            preview="*Multidisciplinary Notes*",
        )

        update = MagicMock(spec=Update)
        query = MagicMock()
        query.data = f"choose:reference:{note_name}"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        update.callback_query = query

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.application = MagicMock()
        context.application.bot_data = {"deps": deps}
        context.chat_data = {note_name: proposal}

        await on_callback(update, context)

        query.answer.assert_called_once()
        query.edit_message_text.assert_called_once()
        msg_text = query.edit_message_text.call_args[0][0]
        assert "Filed to `reference/`" in msg_text
        assert (deps.vault.root / "reference" / note_name).exists()

    sync_run(_test())


def test_on_callback_discard():
    deps = make_fake_deps()

    async def _test():
        note_name = "2026-11-03 - Unwanted-Draft.md"
        inbox_note = Note(
            frontmatter={"title": "Unwanted Draft", "category": "reference", "status": "inbox"},
            body="## Summary\nDiscard me",
            filename=note_name,
        )
        deps.vault.write_note(inbox_note, "inbox")

        proposal = Proposal(
            note=inbox_note,
            category=Category.REFERENCE,
            preview="*Unwanted Draft*",
        )

        update = MagicMock(spec=Update)
        query = MagicMock()
        query.data = f"discard:{note_name}"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        update.callback_query = query

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.application = MagicMock()
        context.application.bot_data = {"deps": deps}
        context.chat_data = {note_name: proposal}

        await on_callback(update, context)

        query.answer.assert_called_once()
        query.edit_message_text.assert_called_once()
        msg_text = query.edit_message_text.call_args[0][0]
        assert "Discarded and moved to `_archive/`" in msg_text
        assert (deps.vault.root / "_archive" / note_name).exists()

    sync_run(_test())


def test_on_callback_recovers_from_inbox_if_chat_data_cleared():
    deps = make_fake_deps()

    async def _test():
        note_name = "2026-11-03 - Recovered-Note.md"
        inbox_note = Note(
            frontmatter={"title": "Recovered Note", "category": "college-documents", "status": "inbox"},
            body="## Summary\nRecovered after bot restart",
            filename=note_name,
        )
        deps.vault.write_note(inbox_note, "inbox")

        update = MagicMock(spec=Update)
        query = MagicMock()
        query.data = f"approve:{note_name}"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        update.callback_query = query

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.application = MagicMock()
        context.application.bot_data = {"deps": deps}
        context.chat_data = {}  # Empty chat_data (e.g. restart)

        await on_callback(update, context)

        query.answer.assert_called_once()
        query.edit_message_text.assert_called_once()
        assert (deps.vault.root / "college-documents" / note_name).exists()

    sync_run(_test())


def test_on_callback_token_identifier():
    deps = make_fake_deps()

    async def _test():
        long_filename = "2026-11-03 - Very-Long-Title-That-Gets-A-Token-Assigned.md"
        inbox_note = Note(
            frontmatter={"title": "Long Title", "category": "possible-projects", "status": "inbox"},
            body="## Summary\nTesting token callbacks",
            filename=long_filename,
        )
        deps.vault.write_note(inbox_note, "inbox")
        token = _callback_token(long_filename)

        proposal = Proposal(
            note=inbox_note,
            category=Category.PROJECT,
            preview="*Long Title*",
        )

        update = MagicMock(spec=Update)
        query = MagicMock()
        query.data = f"approve:{token}"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        update.callback_query = query

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.application = MagicMock()
        context.application.bot_data = {"deps": deps}
        context.chat_data = {token: proposal}

        await on_callback(update, context)

        query.answer.assert_called_once()
        query.edit_message_text.assert_called_once()
        assert (deps.vault.root / "possible-projects" / long_filename).exists()

    sync_run(_test())


def test_on_callback_error_handling():
    deps = make_fake_deps()

    async def _test():
        update = MagicMock(spec=Update)
        query = MagicMock()
        query.data = "approve:nonexistent.md"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        update.callback_query = query

        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.application = MagicMock()
        context.application.bot_data = {"deps": deps}
        context.chat_data = {}

        # Simulating file_proposal error
        with patch("ingest.bot.file_proposal", side_effect=IngestError("write", "I/O failure")):
            fake_prop = MagicMock()
            fake_prop.category = Category.REFERENCE
            context.chat_data = {"nonexistent.md": fake_prop}
            await on_callback(update, context)
            query.edit_message_text.assert_called_once_with("Couldn't process that — write failed.")

    sync_run(_test())


def test_build_deps_main():
    with tempfile.TemporaryDirectory() as td:
        settings = Settings(
            telegram_bot_token="fake_token",
            sarvam_api_key="fake_sarvam_key",
            search_api_key="fake_search_key",
            vault_path=Path(td),
        )
        deps = build_deps(settings)
        assert isinstance(deps, Deps)
        assert deps.llm is not None
        assert deps.ocr is not None
        assert deps.text is not None
        assert deps.search is not None
        assert deps.vault is not None
