"""Pipeline module orchestrating the ingestion flow.

Implements normalise -> classify -> expand_project -> build_note -> ingest -> file_proposal.
Communicates strictly through ports and typed models.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ingest.models import (
    Capture,
    CaptureKind,
    Category,
    Classification,
    IngestError,
    Note,
    Proposal,
)
from ingest.prompts import classify_prompt, project_prompt
from ingest.render import build_note as render_build_note

if TYPE_CHECKING:
    from ingest.adapters.ports import LlmPort, OcrPort, SearchPort, TextPort, VaultPort

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Deps:
    """Dependency injection container holding port instances."""

    llm: LlmPort
    ocr: OcrPort
    text: TextPort
    search: SearchPort
    vault: VaultPort


async def normalise(capture: Capture, deps: Deps) -> tuple[str, str | None]:
    """Extract and normalise inbound content into (original_text, english_translation)."""
    try:
        if capture.kind == CaptureKind.TEXT:
            original = capture.text or ""
        else:
            if not capture.local_path:
                raise ValueError("Media capture has no local file path")
            original = await deps.ocr.extract_text(Path(capture.local_path))

        if capture.caption and capture.caption.strip():
            cap = capture.caption.strip()
            original = f"{original}\n\n{cap}".strip() if original else cap

        # Language check - detect and bypass translation per English-only directive
        await deps.text.detect(original)
        translation: str | None = None

        return original, translation
    except IngestError:
        raise
    except Exception as e:
        stage = "ocr" if capture.kind != CaptureKind.TEXT else "download"
        raise IngestError(stage, e) from e


async def classify(
    capture: Capture,
    original: str,
    translation: str | None,
    deps: Deps,
) -> Classification:
    """Classify extracted content and produce structured metadata."""
    try:
        system_prompt, user_prompt = classify_prompt(capture, original, translation)
        raw_json = await deps.llm.complete_json(system_prompt, user_prompt)
        try:
            return Classification.model_validate(raw_json)
        except Exception as val_err:
            # Design §9: LLM JSON that fails validation gets one repair attempt
            repair_user = (
                f"{user_prompt}\n\n"
                f"Previous response was invalid: {val_err}\n"
                "Please return valid JSON matching the schema."
            )
            repair_json = await deps.llm.complete_json(system_prompt, repair_user)
            return Classification.model_validate(repair_json)
    except IngestError:
        raise
    except Exception as e:
        raise IngestError("classify", e) from e


async def expand_project(idea: str, deps: Deps) -> str:
    """Expand a project idea with web research into a structured Markdown brief."""
    try:
        results = await deps.search.search(idea, k=5)
        system_prompt, user_prompt = project_prompt(idea, results)
        try:
            return await deps.llm.complete_text(system_prompt, user_prompt, stage="search")
        except TypeError:
            return await deps.llm.complete_text(system_prompt, user_prompt)
    except IngestError:
        raise
    except Exception as e:
        raise IngestError("search", e) from e


async def build_note(
    capture: Capture,
    cls: Classification,
    deps: Deps,
    body_extra: str | None = None,
    attachment: str | None = None,
    source_language: str | None = None,
) -> Note:
    """Render a Note with frontmatter and body."""
    try:
        attachment_rel: str | None = attachment
        if attachment_rel is None:
            if capture.local_path:
                filename = capture.filename or Path(capture.local_path).name
                attachment_rel = f"attachments/{cls.category.value}/{filename}"
            elif capture.filename and capture.kind != CaptureKind.TEXT:
                attachment_rel = f"attachments/{cls.category.value}/{capture.filename}"

        return render_build_note(
            capture=capture,
            cls=cls,
            attachment=attachment_rel,
            body_extra=body_extra,
            source_language=source_language,
        )
    except IngestError:
        raise
    except Exception as e:
        raise IngestError("render", e) from e


async def ingest(capture: Capture, deps: Deps) -> Proposal:
    """Execute complete ingestion pipeline and write proposal note to inbox."""
    # 1. Normalise
    original, translation = await normalise(capture, deps)

    source_lang: str | None = None
    try:
        detected = await deps.text.detect(original[:500])
        if detected and detected != "en-IN":
            source_lang = detected
    except Exception:
        pass

    # 2. Classify
    cls = await classify(capture, original, translation, deps)

    # 3. Expand if project
    body_extra: str | None = None
    if cls.category == Category.PROJECT:
        idea_text = capture.text or capture.caption or cls.summary
        body_extra = await expand_project(idea_text, deps)
        if not body_extra or not body_extra.strip():
            logger.warning("Project expansion returned empty research brief for idea: %s", idea_text)

    # 4. Store attachment first if present to resolve exact stored filename (handles collisions)
    stored_att_path: Path | None = None
    attachment_rel: str | None = None
    if capture.local_path and Path(capture.local_path).exists():
        try:
            try:
                stored_att_path = deps.vault.store_attachment(
                    Path(capture.local_path),
                    cls.category.value,
                    filename=capture.filename,
                )
            except TypeError:
                stored_att_path = deps.vault.store_attachment(Path(capture.local_path), cls.category.value)

            if stored_att_path:
                attachment_rel = f"attachments/{cls.category.value}/{stored_att_path.name}"
        except IngestError:
            raise
        except Exception as e:
            raise IngestError("write", e) from e
    elif capture.filename and capture.kind != CaptureKind.TEXT:
        attachment_rel = f"attachments/{cls.category.value}/{capture.filename}"

    # 5. Render Note
    note = await build_note(
        capture,
        cls,
        deps,
        body_extra=body_extra,
        attachment=attachment_rel,
        source_language=source_lang,
    )

    # 6. Write to inbox
    try:
        written_path = deps.vault.write_note(note, "inbox")
    except Exception as e:
        # Rollback: Clean up stored attachment if note writing failed
        if stored_att_path and stored_att_path.exists():
            try:
                stored_att_path.unlink()
            except OSError:
                pass
        if isinstance(e, IngestError):
            raise
        raise IngestError("write", e) from e

    if isinstance(written_path, Path) and written_path.name != note.filename:
        note = note.model_copy(update={"filename": written_path.name})

    # 7. Build Proposal
    preview = f"*{cls.title}*\n\nCategory: `{cls.category.value}`\n\n{cls.summary}"
    return Proposal(note=note, category=cls.category, preview=preview)


def file_proposal(proposal: Proposal, category: Category, deps: Deps) -> Path:
    """File an approved or redirected proposal from inbox into target category folder."""
    try:
        inbox_dir = getattr(deps.vault, "root", Path(".")) / "inbox"
        src_path = inbox_dir / proposal.note.filename
        if not src_path.exists():
            # Check with counter suffixes if collision occurred
            stem = Path(proposal.note.filename).stem
            suffix = Path(proposal.note.filename).suffix or ".md"
            matches = sorted(inbox_dir.glob(f"{stem}*{suffix}"))
            if matches:
                src_path = matches[0]

        target_cat = category.value if hasattr(category, "value") else str(category)
        dest = deps.vault.move_note(src_path, target_cat)

        # Log correction if category differed from initial classification
        orig_cat = (
            proposal.category.value
            if hasattr(proposal.category, "value")
            else str(proposal.category)
        )
        if target_cat != orig_cat:
            if hasattr(deps.vault, "append_correction"):
                deps.vault.append_correction(
                    original=orig_cat,
                    chosen=target_cat,
                    note_title=proposal.note.frontmatter.get("title"),
                )
            elif hasattr(deps.vault, "_append_correction"):
                deps.vault._append_correction(
                    original=orig_cat,
                    chosen=target_cat,
                    note_title=proposal.note.frontmatter.get("title") or "",
                )

            # Relocate attachment if present under old category
            attachment_rel = proposal.note.frontmatter.get("attachment")
            if attachment_rel:
                vault_root = getattr(deps.vault, "root", Path("."))
                old_att_path = vault_root / attachment_rel
                if old_att_path.exists() and hasattr(deps.vault, "store_attachment"):
                    try:
                        deps.vault.store_attachment(old_att_path, target_cat, filename=old_att_path.name)
                        try:
                            old_att_path.unlink()
                        except OSError:
                            pass
                    except Exception:
                        pass

        return dest
    except IngestError:
        raise
    except Exception as e:
        raise IngestError("write", e) from e
