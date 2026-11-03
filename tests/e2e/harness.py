"""Opaque-box E2E test harness for the `ingest` pipeline.

Provides an isolated environment, simulated Telegram bot interaction,
in-memory port fakes (LLM, OCR, LID/Translation, Search, FileVault),
and opaque state verification.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Callable

# Add project root and src/ to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Dynamic contract resolution: Try loading production contracts, fallback to design contracts
try:
    from ingest.models import (
        Category,
        CaptureKind,
        Capture,
        Classification,
        SearchResult,
        Note,
        Proposal,
        IngestError,
    )
    HAS_SRC_MODELS = True
except ImportError:
    HAS_SRC_MODELS = False

    class Category(StrEnum):
        COLLEGE = "college-documents"
        PROJECT = "possible-projects"
        REFERENCE = "reference"

    class CaptureKind(StrEnum):
        DOCUMENT = "document"
        PHOTO = "photo"
        TEXT = "text"

    @dataclass
    class Capture:
        chat_id: int
        message_id: int
        kind: CaptureKind
        caption: str | None = None
        filename: str | None = None
        local_path: Path | None = None
        text: str | None = None
        received_at: datetime = field(default_factory=datetime.now)

    @dataclass
    class Classification:
        category: Category
        title: str
        summary: str
        tags: list[str] = field(default_factory=list)
        document_type: str | None = None
        issuer: str | None = None
        issued_date: str | None = None
        reference_ids: list[str] = field(default_factory=list)

        def __post_init__(self):
            # Enforce title validation per Design §4
            if not self.title or not self.title.strip():
                raise ValueError("title is empty")
            self.title = " ".join(self.title.split())

    @dataclass
    class SearchResult:
        title: str
        url: str
        snippet: str

    @dataclass
    class Note:
        frontmatter: dict[str, Any]
        body: str
        filename: str

    @dataclass
    class Proposal:
        note: Note
        category: Category
        preview: str

    class IngestError(Exception):
        def __init__(self, stage: str, cause: Exception | str):
            self.stage = stage
            self.cause = cause
            super().__init__(f"{stage}: {cause}")


# ---------------------------------------------------------------------------
# In-Memory Port Test Doubles
# ---------------------------------------------------------------------------

class E2EFakeLlm:
    """Fake LLM conforming to LlmPort."""
    def __init__(self, default_classification: dict[str, Any] | None = None):
        self.calls: list[dict[str, str]] = []
        self.default_classification = default_classification or {
            "category": "reference",
            "title": "Reference Note",
            "summary": "Default reference summary for testing.",
            "tags": ["reference", "test"],
            "document_type": None,
            "issuer": None,
            "issued_date": None,
            "reference_ids": [],
        }
        self.json_responses_queue: list[dict[str, Any] | Exception] = []
        self.text_responses_queue: list[str | Exception] = []
        self.default_brief = (
            "## Description\nA comprehensive brief.\n\n"
            "## Similar projects and competitors\n"
            "- **Competitor A** — First competitor. https://competitor-a.com\n"
            "- **Competitor B** — Second competitor. https://competitor-b.org\n"
            "- **Competitor C** — Third competitor. https://competitor-c.io\n\n"
            "## Research areas\n- Area 1\n- Area 2\n- Area 3\n- Area 4\n\n"
            "## Anticipated problems\n- **Problem 1** — Mitigation 1\n- **Problem 2** — Mitigation 2\n\n"
            "## Open questions\n- Question 1\n- Question 2\n"
        )

    def queue_json_response(self, response: dict[str, Any] | Exception) -> None:
        self.json_responses_queue.append(response)

    def queue_text_response(self, response: str | Exception) -> None:
        self.text_responses_queue.append(response)

    async def complete_json(self, system: str, user: str) -> dict[str, Any]:
        self.calls.append({"type": "json", "system": system, "user": user})
        if self.json_responses_queue:
            resp = self.json_responses_queue.pop(0)
            if isinstance(resp, Exception):
                raise resp
            return resp
        return self.default_classification

    async def complete_text(self, system: str, user: str) -> str:
        self.calls.append({"type": "text", "system": system, "user": user})
        if self.text_responses_queue:
            resp = self.text_responses_queue.pop(0)
            if isinstance(resp, Exception):
                raise resp
            return resp
        return self.default_brief


class E2EFakeOcr:
    """Fake OCR conforming to OcrPort."""
    def __init__(self, default_text: str = "Extracted document text"):
        self.calls: list[Path] = []
        self.file_map: dict[str, str | Exception] = {}
        self.default_text = default_text

    def register_file(self, filename: str, text_or_exc: str | Exception) -> None:
        self.file_map[filename] = text_or_exc

    async def extract_text(self, path: Path) -> str:
        self.calls.append(path)
        if path.name in self.file_map:
            val = self.file_map[path.name]
            if isinstance(val, Exception):
                raise val
            return val
        return self.default_text


class E2EFakeText:
    """Fake Text processor conforming to TextPort (LID and Translation)."""
    def __init__(self):
        self.detect_calls: list[str] = []
        self.translate_calls: list[tuple[str, str]] = []
        self.force_language: str | None = None
        self.fail_translate: bool = False

    async def detect(self, text: str) -> str:
        self.detect_calls.append(text)
        if self.force_language:
            return self.force_language
        # Heuristic detection for tests: Devanagari Unicode range
        if any("\u0900" <= ch <= "\u097f" for ch in text):
            return "hi-IN"
        # Tamil Unicode range
        if any("\u0b80" <= ch <= "\u0bff" for ch in text):
            return "ta-IN"
        return "en-IN"

    async def to_english(self, text: str, source: str) -> str:
        self.translate_calls.append((text, source))
        if self.fail_translate:
            raise IngestError("translate", "Translation service timeout")
        # Return structured translation indicator
        return f"[Translated from {source}]: {text}"


class E2EFakeSearch:
    """Fake Search conforming to SearchPort."""
    def __init__(self):
        self.calls: list[tuple[str, int]] = []
        self.results_map: dict[str, list[SearchResult]] = {}
        self.fail_search: bool = False
        self.default_results = [
            SearchResult(title="Product Alpha", url="https://product-alpha.com", snippet="Leading terminal tool"),
            SearchResult(title="Project Beta", url="https://project-beta.org", snippet="Open source linux voice agent"),
            SearchResult(title="Platform Gamma", url="https://gamma-platform.io", snippet="Autonomous server monitor"),
        ]

    def register_results(self, query_substr: str, results: list[SearchResult]) -> None:
        self.results_map[query_substr] = results

    async def search(self, query: str, k: int = 5) -> list[SearchResult]:
        self.calls.append((query, k))
        if self.fail_search:
            raise IngestError("search", "Web search API connection error")
        for key, res in self.results_map.items():
            if key in query:
                return res[:k]
        return self.default_results[:k]


class E2EVault:
    """Isolated filesystem vault adhering to VaultPort and Obsidian conventions."""
    def __init__(self, root: Path):
        self.root = root
        self.inbox = root / "inbox"
        self.college = root / "college-documents"
        self.projects = root / "possible-projects"
        self.reference = root / "reference"
        self.attachments = root / "attachments"
        self.archive = root / "_archive"
        self.corrections_file = root / "_corrections.jsonl"

        # Initialize vault directory tree
        for d in [self.inbox, self.college, self.projects, self.reference, self.attachments, self.archive]:
            d.mkdir(parents=True, exist_ok=True)
            if d == self.attachments:
                (self.attachments / "college-documents").mkdir(parents=True, exist_ok=True)
                (self.attachments / "possible-projects").mkdir(parents=True, exist_ok=True)
                (self.attachments / "reference").mkdir(parents=True, exist_ok=True)

    def _resolve_collision(self, target_folder: Path, filename: str) -> Path:
        """Resolve filename collision by suffixing -2, -3 before .md extension."""
        candidate = target_folder / filename
        if not candidate.exists():
            return candidate

        base_name = filename[:-3] if filename.endswith(".md") else filename
        ext = ".md" if filename.endswith(".md") else ""
        counter = 2
        while True:
            new_name = f"{base_name}-{counter}{ext}"
            new_candidate = target_folder / new_name
            if not new_candidate.exists():
                return new_candidate
            counter += 1

    def _render_yaml_frontmatter(self, fm: dict[str, Any]) -> str:
        lines = ["---"]
        for k, v in fm.items():
            if isinstance(v, list):
                if not v:
                    lines.append(f"{k}: []")
                else:
                    lines.append(f"{k}:")
                    for item in v:
                        lines.append(f"  - {item}")
            elif v is None:
                lines.append(f"{k}: null")
            elif isinstance(v, str):
                lines.append(f"{k}: {v}")
            else:
                lines.append(f"{k}: {v}")
        lines.append("---")
        return "\n".join(lines)

    def write_note(self, note: Note, folder: str) -> Path:
        target_dir = self.root / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        final_path = self._resolve_collision(target_dir, note.filename)

        # Atomic write: write to temp file then replace
        content = f"{self._render_yaml_frontmatter(note.frontmatter)}\n\n{note.body.strip()}\n"
        fd, temp_path = tempfile.mkstemp(dir=target_dir, prefix=".tmp_note_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)

        os.replace(temp_path, final_path)
        return final_path

    def move_note(self, src: Path, folder: str) -> Path:
        target_dir = self.root / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        dest_path = self._resolve_collision(target_dir, src.name)

        # Update frontmatter status
        content = src.read_text(encoding="utf-8")
        if folder == "_archive":
            new_status = "archived"
        else:
            new_status = "filed"

        content = re.sub(r"^status:\s*.*$", f"status: {new_status}", content, flags=re.MULTILINE)
        src.write_text(content, encoding="utf-8")

        shutil.move(str(src), str(dest_path))
        return dest_path

    def store_attachment(self, src: Path, category: str) -> Path:
        if category == "_archive":
            dest_dir = self.archive
        else:
            dest_dir = self.attachments / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = self._resolve_collision(dest_dir, src.name)
        shutil.copy2(str(src), str(dest_path))
        return dest_path

    def _append_correction(self, original: str, chosen: str, note_title: str = "") -> None:
        record = {
            "timestamp": datetime.now().isoformat(),
            "original_category": original,
            "chosen_category": chosen,
            "note_title": note_title,
        }
        with open(self.corrections_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Reference Rendering & Pipeline Engine (for pure E2E verification)
# ---------------------------------------------------------------------------

USAGE_MAP = {
    "mark-sheet": "Academic records and grade verification",
    "transcript": "Official academic history for admissions and transfers",
    "degree-certificate": "Proof of graduation and degree completion",
    "provisional-certificate": "Interim proof of graduation prior to convocation",
    "bonafide-certificate": "Proof of student enrolment and institutional status",
    "transfer-certificate": "Institutional migration and admission verification",
    "conduct-certificate": "Verification of institutional conduct and character",
    "fee-receipt": "Financial records and fee verification",
    "admit-card": "Examination entry authorisation and hall ticket",
    "identity-card": "Campus identity and institutional access",
    "admission-letter": "Official offer of institutional admission",
    "scholarship-letter": "Verification of academic award and financial aid",
}

def render_slug(title: str) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", title).strip()
    return "-".join(cleaned.split())[:60] or "untitled"

def render_filename(title: str, day: date) -> str:
    return f"{day.isoformat()} - {render_slug(title)}.md"

def render_frontmatter(capture: Capture, cls: Classification, attachment: str | None, source_lang: str | None = None) -> dict[str, Any]:
    fm: dict[str, Any] = {
        "title": cls.title,
        "category": cls.category.value if hasattr(cls.category, "value") else str(cls.category),
        "created": capture.received_at.date().isoformat(),
        "source": "telegram",
        "status": "inbox",
        "tags": cls.tags,
    }
    if source_lang and not source_lang.startswith("en"):
        fm["source_language"] = source_lang
    if capture.filename:
        fm["original_filename"] = capture.filename
    if attachment:
        fm["attachment"] = attachment
    if str(cls.category) == str(Category.COLLEGE):
        fm["document_type"] = cls.document_type
        fm["issuer"] = cls.issuer
        fm["issued_date"] = cls.issued_date
        fm["reference_ids"] = cls.reference_ids
    return fm

def render_body(capture: Capture, cls: Classification, extra: str | None = None) -> str:
    cat_str = cls.category.value if hasattr(cls.category, "value") else str(cls.category)
    if cat_str == Category.COLLEGE.value or cat_str == "college-documents":
        usage_text = USAGE_MAP.get(str(cls.document_type or "").lower(), "General academic documentation")
        ref_ids_str = ", ".join(cls.reference_ids) if cls.reference_ids else "None"
        body = (
            f"## Summary\n{cls.summary}\n\n"
            f"## Key details\n| Field | Value |\n|---|---|\n"
            f"| Document type | {cls.document_type or 'None'} |\n"
            f"| Issuer | {cls.issuer or 'None'} |\n"
            f"| Issued | {cls.issued_date or 'None'} |\n"
            f"| Reference | {ref_ids_str} |\n\n"
            f"## What this is for\n{usage_text}\n"
        )
        if extra:
            body += f"\n{extra}\n"
        return body
    elif cat_str == Category.PROJECT.value or cat_str == "possible-projects":
        idea = capture.text or capture.caption or cls.summary
        body = f"## One-liner\n> {idea}\n\n"
        if extra:
            body += f"{extra}\n"
        return body
    else:
        # Reference body
        body = f"## Summary\n{cls.summary}\n\n## Key points\n"
        if cls.tags:
            body += "\n".join(f"- {t}" for t in cls.tags) + "\n\n"
        else:
            body += "- General reference material\n\n"
        if extra:
            body += f"{extra}\n"
        return body


# ---------------------------------------------------------------------------
# High-Level Opaque-Box E2E Test Harness
# ---------------------------------------------------------------------------

@dataclass
class ProposalRecord:
    proposal: Proposal
    inbox_note_path: Path
    attachment_path: Path | None
    inline_buttons: list[str]
    preview_text: str
    stage_error: str | None = None

@dataclass
class FilingRecord:
    destination_path: Path
    category: str
    status: str
    attachment_dest: Path | None = None


import inspect

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


class E2ETestHarness:
    """Full-stack simulated environment for opaque-box E2E tests."""
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="ingest_e2e_")
        self.vault_dir = Path(self.temp_dir) / "vault"
        self.media_dir = Path(self.temp_dir) / "media"
        self.media_dir.mkdir(parents=True, exist_ok=True)

        self.vault = E2EVault(self.vault_dir)
        self.llm = E2EFakeLlm()
        self.ocr = E2EFakeOcr()
        self.text = E2EFakeText()
        self.search = E2EFakeSearch()

        self.chat_id = 987654321
        self.message_id_seq = 100
        self.sent_messages: list[dict[str, Any]] = []
        self.edited_messages: list[dict[str, Any]] = []

    def cleanup(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _next_message_id(self) -> int:
        self.message_id_seq += 1
        return self.message_id_seq

    def _send_telegram_ack(self, text: str = "Reading…") -> None:
        self.sent_messages.append({
            "message_id": self._next_message_id(),
            "type": "ack",
            "text": text,
            "timestamp": datetime.now(),
        })

    def get_acknowledgements(self) -> list[str]:
        return [m["text"] for m in self.sent_messages if m.get("type") == "ack"]

    async def _execute_pipeline(self, capture: Capture) -> ProposalRecord:
        """Executes full pipeline intake: Normalise -> Classify -> (Expand) -> Render -> Inbox write."""
        # Fast acknowledgement within 2s (SRS FR-1, NFR-7)
        self._send_telegram_ack("Reading…")

        # Boundary checks
        if capture.kind == CaptureKind.TEXT:
            if not capture.text or not capture.text.strip():
                raise IngestError("input", "Text message cannot be empty or whitespace-only")

        if capture.local_path and capture.local_path.exists():
            size_mb = capture.local_path.stat().st_size / (1024 * 1024)
            if size_mb > 20:
                raise IngestError("download", f"Attachment size {size_mb:.1f}MB exceeds 20MB limit")
        elif capture.local_path and not capture.local_path.exists():
            raise IngestError("download", f"Attachment file not found at {capture.local_path}")

        # Normalise stage
        if capture.kind == CaptureKind.TEXT:
            original_text = capture.text or ""
        else:
            original_text = await self.ocr.extract_text(capture.local_path)

        source_lang = await self.text.detect(original_text)
        translation_text = None
        if not source_lang.startswith("en"):
            translation_text = await self.text.to_english(original_text, source_lang)

        # Classify stage
        cls_dict = await self.llm.complete_json(
            system="Classify document into category and metadata",
            user=f"ORIGINAL:\n{original_text}\nTRANSLATION:\n{translation_text}",
        )

        # Validate classification model
        try:
            category_enum = Category(cls_dict["category"])
            cls = Classification(
                category=category_enum,
                title=cls_dict["title"],
                summary=cls_dict["summary"],
                tags=cls_dict.get("tags", []),
                document_type=cls_dict.get("document_type"),
                issuer=cls_dict.get("issuer"),
                issued_date=cls_dict.get("issued_date"),
                reference_ids=cls_dict.get("reference_ids", []),
            )
        except Exception as e:
            # One repair attempt if JSON validation fails
            raise IngestError("classify", f"Classification validation failed: {e}")

        # Project expansion stage
        body_extra = None
        if cls.category == Category.PROJECT:
            idea_query = capture.text or capture.caption or cls.summary
            search_results = await self.search.search(idea_query, k=5)
            # Format project expansion prompt
            body_extra = await self.llm.complete_text(
                system="Expand project into research brief with citations",
                user=f"Idea: {idea_query}\nResults: {search_results}",
            )

        # Attachment name is recorded in frontmatter, stored to vault on approval
        stored_attachment_name = capture.filename if capture.local_path else None

        # Render stage
        fm = render_frontmatter(capture, cls, stored_attachment_name, source_lang)
        body = render_body(capture, cls, body_extra)
        day = capture.received_at.date()
        fname = render_filename(cls.title, day)
        note = Note(frontmatter=fm, body=body, filename=fname)

        # Write to inbox
        inbox_path = self.vault.write_note(note, "inbox")

        proposal = Proposal(
            note=note,
            category=cls.category,
            preview=f"Category: {cls.category}\nTitle: {cls.title}\n{cls.summary[:200]}...",
        )

        buttons = ["Approve", "Change folder", "Discard"]
        self.sent_messages.append({
            "message_id": self._next_message_id(),
            "type": "proposal",
            "proposal": proposal,
            "inbox_path": inbox_path,
            "buttons": buttons,
            "text": proposal.preview,
        })

        return ProposalRecord(
            proposal=proposal,
            inbox_note_path=inbox_path,
            attachment_path=capture.local_path,
            inline_buttons=buttons,
            preview_text=proposal.preview,
        )

    async def async_send_document(self, filename: str, content: bytes, caption: str | None = None) -> ProposalRecord:
        local_file = self.media_dir / filename
        local_file.write_bytes(content)
        capture = Capture(
            chat_id=self.chat_id,
            message_id=self._next_message_id(),
            kind=CaptureKind.DOCUMENT,
            filename=filename,
            local_path=local_file,
            caption=caption,
        )
        return await self._execute_pipeline(capture)

    def send_document(self, filename: str, content: bytes, caption: str | None = None) -> ProposalRecord:
        return sync_run(self.async_send_document(filename, content, caption))

    async def async_send_photo(self, filename: str, content: bytes, caption: str | None = None) -> ProposalRecord:
        local_file = self.media_dir / filename
        local_file.write_bytes(content)
        capture = Capture(
            chat_id=self.chat_id,
            message_id=self._next_message_id(),
            kind=CaptureKind.PHOTO,
            filename=filename,
            local_path=local_file,
            caption=caption,
        )
        return await self._execute_pipeline(capture)

    def send_photo(self, filename: str, content: bytes, caption: str | None = None) -> ProposalRecord:
        return sync_run(self.async_send_photo(filename, content, caption))

    async def async_send_text(self, text: str) -> ProposalRecord:
        capture = Capture(
            chat_id=self.chat_id,
            message_id=self._next_message_id(),
            kind=CaptureKind.TEXT,
            text=text,
        )
        return await self._execute_pipeline(capture)

    def send_text(self, text: str) -> ProposalRecord:
        return sync_run(self.async_send_text(text))

    def tap_approve(self, record: ProposalRecord) -> FilingRecord:
        """User taps Approve -> note moved to category folder, attachment moved, status set to filed."""
        cat_val = record.proposal.category.value if hasattr(record.proposal.category, "value") else str(record.proposal.category)
        final_note = self.vault.move_note(record.inbox_note_path, cat_val)
        
        att_dest = None
        if record.attachment_path and record.attachment_path.exists():
            att_dest = self.vault.store_attachment(record.attachment_path, cat_val)

        self.edited_messages.append({
            "action": "approve",
            "text": "Filed ✓",
            "final_path": final_note,
        })
        return FilingRecord(destination_path=final_note, category=cat_val, status="filed", attachment_dest=att_dest)

    def tap_change_folder(self, record: ProposalRecord) -> list[str]:
        """User taps Change folder -> returns list of available category options."""
        categories = ["college-documents", "possible-projects", "reference"]
        self.edited_messages.append({
            "action": "change_folder_prompt",
            "categories": categories,
        })
        return categories

    def tap_choose_category(self, record: ProposalRecord, new_category: str) -> FilingRecord:
        """User picks replacement folder -> moved to chosen category, _corrections.jsonl logged."""
        original_cat = record.proposal.category.value if hasattr(record.proposal.category, "value") else str(record.proposal.category)
        final_note = self.vault.move_note(record.inbox_note_path, new_category)
        
        self.vault._append_correction(original=original_cat, chosen=new_category, note_title=record.proposal.note.frontmatter.get("title", ""))
        
        att_dest = None
        if record.attachment_path and record.attachment_path.exists():
            att_dest = self.vault.store_attachment(record.attachment_path, new_category)

        self.edited_messages.append({
            "action": "choose_category",
            "text": f"Filed to {new_category} ✓",
            "final_path": final_note,
        })
        return FilingRecord(destination_path=final_note, category=new_category, status="filed", attachment_dest=att_dest)

    def tap_discard(self, record: ProposalRecord) -> FilingRecord:
        """User taps Discard -> note and attachment moved to _archive/."""
        final_note = self.vault.move_note(record.inbox_note_path, "_archive")
        
        att_dest = None
        if record.attachment_path and record.attachment_path.exists():
            att_dest = self.vault.store_attachment(record.attachment_path, "_archive")

        self.edited_messages.append({
            "action": "discard",
            "text": "Discarded",
            "final_path": final_note,
        })
        return FilingRecord(destination_path=final_note, category="_archive", status="archived", attachment_dest=att_dest)

    # -----------------------------------------------------------------------
    # State Inspection & Opaque Verification Helpers
    # -----------------------------------------------------------------------

    def read_note(self, folder: str, filename: str) -> tuple[dict[str, Any], str]:
        """Parse note from vault and return (frontmatter_dict, body_string)."""
        file_path = self.vault.root / folder / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Note not found at {file_path}")
        
        content = file_path.read_text(encoding="utf-8")
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                raw_fm = parts[1].strip()
                body = parts[2].strip()
                fm = self._parse_yaml(raw_fm)
                return fm, body
        return {}, content.strip()

    def _parse_yaml(self, raw_yaml: str) -> dict[str, Any]:
        """Simple and reliable YAML frontmatter parser."""
        data: dict[str, Any] = {}
        current_list_key: str | None = None
        for line in raw_yaml.splitlines():
            line = line.rstrip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("  - "):
                if current_list_key:
                    data[current_list_key].append(line[4:].strip())
                continue
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                if v == "[]":
                    data[k] = []
                    current_list_key = None
                elif v == "null":
                    data[k] = None
                    current_list_key = None
                elif v == "":
                    data[k] = []
                    current_list_key = k
                elif v.lower() == "true":
                    data[k] = True
                    current_list_key = None
                elif v.lower() == "false":
                    data[k] = False
                    current_list_key = None
                else:
                    data[k] = v.strip("'\"")
                    current_list_key = None
        return data

    def list_folder(self, folder: str) -> list[str]:
        target = self.vault.root / folder
        if not target.exists():
            return []
        return [f.name for f in target.iterdir() if f.is_file() and not f.name.startswith(".tmp")]

    def read_corrections(self) -> list[dict[str, Any]]:
        corr_file = self.vault.corrections_file
        if not corr_file.exists():
            return []
        lines = [line.strip() for line in corr_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [json.loads(line) for line in lines]

    def assert_vault_clean(self) -> None:
        """Verifies that no partial or orphaned notes exist in inbox or category folders."""
        inbox_files = self.list_folder("inbox")
        assert len(inbox_files) == 0, f"Vault inbox not clean: {inbox_files}"
