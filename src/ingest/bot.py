"""Telegram Bot interface and event handlers for the ingest pipeline.

Imperative shell: translates Telegram events into Capture models, invokes the pipeline,
and renders interactive inline keyboards for confirmation.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import re
import shutil
import tempfile
from typing import TYPE_CHECKING, Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ingest.models import Capture, CaptureKind, Category, IngestError, Proposal
from ingest.pipeline import Deps, file_proposal, ingest

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Maximum upload limit per Telegram Bot API specification
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


def _callback_token(identifier: str) -> str:
    """Generate a stable 16-character hex hash token for long identifiers."""
    return hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:16]


def _make_callback_data(prefix: str, identifier: str, max_bytes: int = 64) -> str:
    """Ensure callback data fits within Telegram's 64-byte payload limit.

    If the full prefix:identifier fits within 64 bytes, it is preserved directly.
    If it exceeds 64 bytes, identifier is replaced with a compact 16-character hash token.
    """
    candidate = f"{prefix}:{identifier}"
    if len(candidate.encode("utf-8")) <= max_bytes:
        return candidate

    token = _callback_token(identifier)
    shortened = f"{prefix}:{token}"
    if len(shortened.encode("utf-8")) <= max_bytes:
        return shortened

    # Fallback byte slice if prefix itself was exceptionally long
    return shortened.encode("utf-8")[:max_bytes].decode("utf-8", "ignore")


def build_proposal_keyboard(note_filename: str) -> InlineKeyboardMarkup:
    """Build the 3-button confirmation keyboard: Approve, Change folder, Discard.

    Ensures all callback data values strictly obey Telegram's 64-byte limit.
    """
    approve_data = _make_callback_data("approve", note_filename)
    change_data = _make_callback_data("change", note_filename)
    discard_data = _make_callback_data("discard", note_filename)

    keyboard = [
        [
            InlineKeyboardButton("Approve", callback_data=approve_data),
            InlineKeyboardButton("Change folder", callback_data=change_data),
            InlineKeyboardButton("Discard", callback_data=discard_data),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def build_category_choice_keyboard(note_filename: str) -> InlineKeyboardMarkup:
    """Build the folder selection keyboard for user correction.

    Ensures all callback data values strictly obey Telegram's 64-byte limit.
    """
    college_data = _make_callback_data(f"choose:{Category.COLLEGE.value}", note_filename)
    project_data = _make_callback_data(f"choose:{Category.PROJECT.value}", note_filename)
    reference_data = _make_callback_data(f"choose:{Category.REFERENCE.value}", note_filename)

    keyboard = [
        [
            InlineKeyboardButton("College Documents", callback_data=college_data),
        ],
        [
            InlineKeyboardButton("Possible Projects", callback_data=project_data),
        ],
        [
            InlineKeyboardButton("Reference", callback_data=reference_data),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def _resolve_inbox_file(inbox_dir: Path, note_key: str) -> Path | None:
    """Resolve an inbox file from either its exact filename, collision stem, or hash token."""
    if not inbox_dir.exists():
        return None

    # 1. Exact path match
    direct = inbox_dir / note_key
    if direct.exists():
        return direct

    # 2. Collision suffix matches (e.g. Note-2.md)
    stem = Path(note_key).stem
    suffix = Path(note_key).suffix or ".md"
    matches = list(inbox_dir.glob(f"{stem}*{suffix}"))
    if matches:
        return matches[0]

    # 3. Hash token match against files in inbox
    for f in inbox_dir.glob("*.md"):
        if _callback_token(f.name) == note_key:
            return f

    return None


def _read_category_from_note(note_path: Path) -> Category | None:
    """Extract category from note frontmatter if available."""
    try:
        content = note_path.read_text(encoding="utf-8")
        match = re.search(r"^category:\s*([^\s\n]+)", content, flags=re.MULTILINE)
        if match:
            return Category(match.group(1).strip())
    except Exception:
        pass
    return None


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command greeting the user."""
    if not update.effective_message:
        return
    await update.effective_message.reply_text(
        "👋 Welcome to Ingest!\n\n"
        "Send or forward me:\n"
        "• Academic documents or scans (PDFs, images)\n"
        "• Project ideas (text)\n"
        "• General notes and references\n\n"
        "I will extract, classify, research, and file them into your Obsidian vault with one tap."
    )


async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle document uploads (PDFs, docx, scans) with size validation and fast receipt."""
    message = update.effective_message
    if not message or not message.document:
        return

    deps: Deps = context.application.bot_data["deps"]

    # Fast receipt acknowledgement within 2 seconds
    ack = await message.reply_text("Reading…")

    # Document size validation (<= 20 MB)
    file_size = getattr(message.document, "file_size", None)
    if isinstance(file_size, int) and file_size > MAX_FILE_SIZE:
        await ack.edit_text("Couldn't process that — file exceeds the 20 MB limit. Please send a smaller file.")
        return

    temp_dir = Path(tempfile.mkdtemp(prefix="ingest_doc_"))
    try:
        tg_file = await message.document.get_file()
        raw_name = getattr(message.document, "file_name", None)
        file_name = raw_name if isinstance(raw_name, str) and raw_name else "document.pdf"
        local_path = temp_dir / file_name
        await tg_file.download_to_drive(str(local_path))

        # Secondary size validation after download
        if local_path.exists() and local_path.stat().st_size > MAX_FILE_SIZE:
            await ack.edit_text("Couldn't process that — file exceeds the 20 MB limit. Please send a smaller file.")
            return

        caption_val = message.caption if isinstance(message.caption, str) else None
        capture = Capture(
            chat_id=message.chat_id,
            message_id=message.message_id,
            kind=CaptureKind.DOCUMENT,
            local_path=local_path,
            filename=file_name,
            caption=caption_val,
        )

        proposal: Proposal = await ingest(capture, deps)
        context.chat_data[proposal.note.filename] = proposal
        context.chat_data[_callback_token(proposal.note.filename)] = proposal
        logger.info("✨ Proposal ready: '%s' [%s]", proposal.note.filename, proposal.category.value)

        # Present proposal with inline keyboard; fallback to plain text if Markdown parsing fails
        try:
            await ack.edit_text(
                proposal.preview,
                parse_mode="Markdown",
                reply_markup=build_proposal_keyboard(proposal.note.filename),
            )
        except Exception as pe:
            logger.warning("Markdown formatting failed (%s); sending plain text preview", pe)
            await ack.edit_text(
                proposal.preview,
                parse_mode=None,
                reply_markup=build_proposal_keyboard(proposal.note.filename),
            )
    except IngestError as e:
        logger.error(
            "❌ Pipeline failed at stage '%s' during document processing: %s (cause: %s)",
            e.stage,
            e,
            getattr(e, "cause", None),
            exc_info=True,
        )
        await ack.edit_text(f"Couldn't process that — {e.stage} failed. Send it again.")
    except Exception as e:
        logger.exception("❌ Unexpected error in on_document: %s", e)
        await ack.edit_text("Couldn't process that — an unexpected error occurred. Send it again.")
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle photo messages with size validation and fast receipt."""
    message = update.effective_message
    if not message or not message.photo:
        return

    deps: Deps = context.application.bot_data["deps"]

    # Fast receipt acknowledgement within 2 seconds
    ack = await message.reply_text("Reading…")

    # Photo item selection (highest resolution is at the end)
    photo_item = message.photo[-1]

    # Photo size validation (<= 20 MB)
    p_size = getattr(photo_item, "file_size", None)
    if isinstance(p_size, int) and p_size > MAX_FILE_SIZE:
        await ack.edit_text("Couldn't process that — photo exceeds the 20 MB limit. Please send a smaller photo.")
        return

    temp_dir = Path(tempfile.mkdtemp(prefix="ingest_photo_"))
    try:
        tg_file = await photo_item.get_file()
        file_unique = getattr(photo_item, "file_unique_id", "photo")
        if not isinstance(file_unique, str):
            file_unique = "photo"
        file_name = f"photo_{file_unique}.jpg"
        local_path = temp_dir / file_name
        await tg_file.download_to_drive(str(local_path))

        # Secondary size validation after download
        if local_path.exists() and local_path.stat().st_size > MAX_FILE_SIZE:
            await ack.edit_text("Couldn't process that — photo exceeds the 20 MB limit. Please send a smaller photo.")
            return

        caption_val = message.caption if isinstance(message.caption, str) else None
        capture = Capture(
            chat_id=message.chat_id,
            message_id=message.message_id,
            kind=CaptureKind.PHOTO,
            local_path=local_path,
            filename=file_name,
            caption=caption_val,
        )

        proposal: Proposal = await ingest(capture, deps)
        context.chat_data[proposal.note.filename] = proposal
        context.chat_data[_callback_token(proposal.note.filename)] = proposal

        try:
            await ack.edit_text(
                proposal.preview,
                parse_mode="Markdown",
                reply_markup=build_proposal_keyboard(proposal.note.filename),
            )
        except Exception as pe:
            logger.warning("Markdown formatting failed (%s); sending plain text preview", pe)
            await ack.edit_text(
                proposal.preview,
                parse_mode=None,
                reply_markup=build_proposal_keyboard(proposal.note.filename),
            )
    except IngestError as e:
        logger.error(
            "❌ Pipeline failed at stage '%s' during photo processing: %s (cause: %s)",
            e.stage,
            e,
            getattr(e, "cause", None),
            exc_info=True,
        )
        await ack.edit_text(f"Couldn't process that — {e.stage} failed. Send it again.")
    except Exception as e:
        logger.exception("❌ Unexpected error in on_photo: %s", e)
        await ack.edit_text("Couldn't process that — an unexpected error occurred. Send it again.")
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle plain text messages and project ideas with fast receipt acknowledgement."""
    message = update.effective_message
    if not message or not message.text or not isinstance(message.text, str):
        return

    deps: Deps = context.application.bot_data["deps"]

    # Fast receipt acknowledgement within 2 seconds
    ack = await message.reply_text("Thinking…")

    try:
        capture = Capture(
            chat_id=message.chat_id,
            message_id=message.message_id,
            kind=CaptureKind.TEXT,
            text=message.text,
        )

        proposal: Proposal = await ingest(capture, deps)
        context.chat_data[proposal.note.filename] = proposal
        context.chat_data[_callback_token(proposal.note.filename)] = proposal

        try:
            await ack.edit_text(
                proposal.preview,
                parse_mode="Markdown",
                reply_markup=build_proposal_keyboard(proposal.note.filename),
            )
        except Exception as pe:
            logger.warning("Markdown formatting failed (%s); sending plain text preview", pe)
            await ack.edit_text(
                proposal.preview,
                parse_mode=None,
                reply_markup=build_proposal_keyboard(proposal.note.filename),
            )
    except IngestError as e:
        logger.error(
            "❌ Pipeline failed at stage '%s' during text processing: %s (cause: %s)",
            e.stage,
            e,
            getattr(e, "cause", None),
            exc_info=True,
        )
        await ack.edit_text(f"Couldn't process that — {e.stage} failed. Send it again.")
    except Exception as e:
        logger.exception("❌ Unexpected error in on_text: %s", e)
        await ack.edit_text("Couldn't process that — an unexpected error occurred. Send it again.")


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route inline keyboard callbacks (Approve, Change folder, Choose, Discard)."""
    query = update.callback_query
    if not query or not query.data:
        return

    await query.answer()
    data = query.data
    deps: Deps = context.application.bot_data["deps"]

    if ":" not in data:
        return

    parts = data.split(":", 2)
    action = parts[0]

    try:
        if action == "approve":
            note_key = parts[1]
            proposal: Proposal | None = context.chat_data.get(note_key)
            if proposal:
                dest = file_proposal(proposal, proposal.category, deps)
                await query.edit_message_text(f"✅ Filed to `{proposal.category.value}/` as `{dest.name}`")
            else:
                # Fallback: attempt resolution from inbox filesystem
                inbox_dir = getattr(deps.vault, "root", Path(".")) / "inbox"
                resolved = _resolve_inbox_file(inbox_dir, note_key)
                if resolved and resolved.exists():
                    cat = _read_category_from_note(resolved) or Category.REFERENCE
                    dest = deps.vault.move_note(resolved, cat.value)
                    await query.edit_message_text(f"✅ Filed to `{cat.value}/` as `{dest.name}`")
                else:
                    await query.edit_message_text("Note filed ✓ (already moved from inbox)")

        elif action == "change":
            note_key = parts[1]
            await query.edit_message_reply_markup(
                reply_markup=build_category_choice_keyboard(note_key)
            )

        elif action == "choose":
            cat_str = parts[1]
            note_key = parts[2]
            chosen_cat = Category(cat_str)
            proposal = context.chat_data.get(note_key)
            if proposal:
                dest = file_proposal(proposal, chosen_cat, deps)
                await query.edit_message_text(f"✅ Filed to `{chosen_cat.value}/` as `{dest.name}`")
            else:
                inbox_dir = getattr(deps.vault, "root", Path(".")) / "inbox"
                resolved = _resolve_inbox_file(inbox_dir, note_key)
                if resolved and resolved.exists():
                    orig_cat = _read_category_from_note(resolved) or Category.REFERENCE
                    dest = deps.vault.move_note(resolved, chosen_cat.value)
                    if orig_cat != chosen_cat and hasattr(deps.vault, "append_correction"):
                        deps.vault.append_correction(
                            original=orig_cat.value,
                            chosen=chosen_cat.value,
                            note_title=resolved.stem,
                        )
                    await query.edit_message_text(f"✅ Filed to `{chosen_cat.value}/` as `{dest.name}`")
                else:
                    await query.edit_message_text("Note filed ✓ (already moved from inbox)")

        elif action == "discard":
            note_key = parts[1]
            proposal = context.chat_data.get(note_key)
            inbox_dir = getattr(deps.vault, "root", Path(".")) / "inbox"
            resolved_file = _resolve_inbox_file(inbox_dir, note_key)

            att_path: Path | None = None
            if proposal and proposal.note.frontmatter.get("attachment"):
                vault_root = getattr(deps.vault, "root", Path("."))
                candidate_att = vault_root / proposal.note.frontmatter["attachment"]
                if candidate_att.exists():
                    att_path = candidate_att

            if resolved_file and resolved_file.exists():
                if hasattr(deps.vault, "archive_note"):
                    deps.vault.archive_note(resolved_file, att_path)
                elif hasattr(deps.vault, "archive"):
                    deps.vault.archive(resolved_file)
                else:
                    archive_dir = getattr(deps.vault, "root", Path(".")) / "_archive"
                    archive_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(resolved_file), str(archive_dir / resolved_file.name))

                await query.edit_message_text("🗑️ Discarded and moved to `_archive/`")
            else:
                await query.edit_message_text("⚠️ Note not found in inbox — it may have already been filed or archived.")
    except IngestError as e:
        logger.error(
            "❌ Pipeline failed at stage '%s' during callback handling: %s (cause: %s)",
            e.stage,
            e,
            getattr(e, "cause", None),
            exc_info=True,
        )
        await query.edit_message_text(f"Couldn't process that — {e.stage} failed.")
    except Exception as e:
        logger.exception("❌ Unexpected error in on_callback: %s", e)
        await query.edit_message_text("Couldn't process that — an unexpected error occurred.")


def setup_bot_handlers(app: Application, deps: Deps) -> None:
    """Register all command, message, and callback query handlers."""
    app.bot_data["deps"] = deps

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.Document.ALL, on_document))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(CallbackQueryHandler(on_callback))
