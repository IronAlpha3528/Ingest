"""Unit tests for the Obsidian FileVault adapter."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import pytest

from ingest.adapters.vault import FileVault
from ingest.models import IngestError, Note
from ingest.render import dump_frontmatter


@pytest.fixture
def tmp_path() -> Path:
    """Fixture providing a temporary directory Path compatible with pytest-compat."""
    return Path(tempfile.mkdtemp())


@pytest.fixture
def temp_vault() -> FileVault:
    """Fixture providing an initialized FileVault in a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    return FileVault(temp_dir)


def test_vault_init_creates_directories(temp_vault: FileVault):
    assert (temp_vault.root / "inbox").is_dir()
    assert (temp_vault.root / "college-documents").is_dir()
    assert (temp_vault.root / "possible-projects").is_dir()
    assert (temp_vault.root / "projects").is_dir()
    assert (temp_vault.root / "reference").is_dir()
    assert (temp_vault.root / "_archive").is_dir()
    assert (temp_vault.root / "_archive" / "attachments").is_dir()
    assert (temp_vault.root / "attachments" / "college-documents").is_dir()
    assert (temp_vault.root / "attachments" / "possible-projects").is_dir()
    assert (temp_vault.root / "attachments" / "projects").is_dir()
    assert (temp_vault.root / "attachments" / "reference").is_dir()


def test_write_note_creates_file_atomic(temp_vault: FileVault):
    note = Note(
        frontmatter={"title": "Test Note", "status": "inbox", "tags": ["test"]},
        body="## Summary\nThis is a test body.",
        filename="2026-11-03 - Test-Note.md",
    )
    # Default folder = "inbox"
    path = temp_vault.write_note(note)
    assert path.exists()
    assert path.parent == temp_vault.root / "inbox"
    assert path.name == "2026-11-03 - Test-Note.md"

    expected_content = f"{dump_frontmatter(note.frontmatter)}\n\n{note.body}\n"
    content = path.read_text(encoding="utf-8")
    assert content == expected_content


def test_write_note_custom_folder(temp_vault: FileVault):
    note = Note(
        frontmatter={"title": "Custom Folder Note", "status": "inbox"},
        body="Body content here",
        filename="2026-11-03 - Custom-Folder.md",
    )
    path = temp_vault.write_note(note, folder="college-documents")
    assert path.exists()
    assert path.parent == temp_vault.root / "college-documents"
    assert "title: Custom Folder Note" in path.read_text(encoding="utf-8")


def test_write_note_collision_appends_counter(temp_vault: FileVault):
    note1 = Note(
        frontmatter={"title": "Test Collision", "status": "inbox"},
        body="Body 1",
        filename="2026-11-03 - Test-Collision.md",
    )
    note2 = Note(
        frontmatter={"title": "Test Collision", "status": "inbox"},
        body="Body 2",
        filename="2026-11-03 - Test-Collision.md",
    )
    note3 = Note(
        frontmatter={"title": "Test Collision", "status": "inbox"},
        body="Body 3",
        filename="2026-11-03 - Test-Collision.md",
    )
    p1 = temp_vault.write_note(note1, "inbox")
    p2 = temp_vault.write_note(note2, "inbox")
    p3 = temp_vault.write_note(note3, "inbox")

    assert p1.name == "2026-11-03 - Test-Collision.md"
    assert p2.name == "2026-11-03 - Test-Collision-2.md"
    assert p3.name == "2026-11-03 - Test-Collision-3.md"
    assert p1.exists()
    assert p2.exists()
    assert p3.exists()
    assert "Body 1" in p1.read_text(encoding="utf-8")
    assert "Body 2" in p2.read_text(encoding="utf-8")
    assert "Body 3" in p3.read_text(encoding="utf-8")


def test_move_note_updates_status_to_filed(temp_vault: FileVault):
    note = Note(
        frontmatter={"title": "Moving Note", "status": "inbox"},
        body="Body to move",
        filename="2026-11-03 - Moving-Note.md",
    )
    inbox_path = temp_vault.write_note(note, "inbox")
    assert inbox_path.exists()
    assert "status: inbox" in inbox_path.read_text(encoding="utf-8")

    dest_path = temp_vault.move_note(inbox_path, "college-documents")
    assert not inbox_path.exists()
    assert dest_path.exists()
    assert dest_path.parent == temp_vault.root / "college-documents"
    content = dest_path.read_text(encoding="utf-8")
    assert "status: filed" in content
    assert "status: inbox" not in content


def test_move_note_collision_handling(temp_vault: FileVault):
    # Prepare note already in destination folder
    dest_dir = temp_vault.root / "college-documents"
    existing_dest = dest_dir / "2026-11-03 - Same-Name.md"
    existing_dest.write_text("Existing note content", encoding="utf-8")

    # Prepare note in inbox with identical filename
    note = Note(
        frontmatter={"title": "Same Name", "status": "inbox"},
        body="New note content",
        filename="2026-11-03 - Same-Name.md",
    )
    inbox_path = temp_vault.write_note(note, "inbox")

    moved_path = temp_vault.move_note(inbox_path, "college-documents")
    assert moved_path.name == "2026-11-03 - Same-Name-2.md"
    assert moved_path.exists()
    assert existing_dest.exists()
    assert "status: filed" in moved_path.read_text(encoding="utf-8")
    assert "Existing note content" in existing_dest.read_text(encoding="utf-8")


def test_store_attachment(temp_vault: FileVault, tmp_path: Path):
    source_pdf = tmp_path / "test_document.pdf"
    source_pdf.write_text("Simulated PDF content", encoding="utf-8")

    dest = temp_vault.store_attachment(source_pdf, "college-documents")
    assert dest.exists()
    assert dest.parent == temp_vault.root / "attachments" / "college-documents"
    assert dest.name == "test_document.pdf"
    assert dest.read_text(encoding="utf-8") == "Simulated PDF content"


def test_store_attachment_collision_handling(temp_vault: FileVault, tmp_path: Path):
    att1 = tmp_path / "paper.pdf"
    att1.write_text("Paper 1", encoding="utf-8")
    att2 = tmp_path / "sub" / "paper.pdf"
    att2.parent.mkdir(parents=True, exist_ok=True)
    att2.write_text("Paper 2", encoding="utf-8")
    att3 = tmp_path / "sub2" / "paper.pdf"
    att3.parent.mkdir(parents=True, exist_ok=True)
    att3.write_text("Paper 3", encoding="utf-8")

    dest1 = temp_vault.store_attachment(att1, "reference")
    dest2 = temp_vault.store_attachment(att2, "reference")
    dest3 = temp_vault.store_attachment(att3, "reference")

    assert dest1.name == "paper.pdf"
    assert dest2.name == "paper-2.pdf"
    assert dest3.name == "paper-3.pdf"
    assert dest1.read_text(encoding="utf-8") == "Paper 1"
    assert dest2.read_text(encoding="utf-8") == "Paper 2"
    assert dest3.read_text(encoding="utf-8") == "Paper 3"


def test_append_correction_dict(temp_vault: FileVault):
    entry = {
        "original_category": "college-documents",
        "chosen_category": "reference",
        "note_title": "Marksheet Classification",
    }
    temp_vault.append_correction(entry)

    corrections_file = temp_vault.root / "_corrections.jsonl"
    assert corrections_file.exists()
    lines = corrections_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["original_category"] == "college-documents"
    assert data["chosen_category"] == "reference"
    assert data["note_title"] == "Marksheet Classification"
    assert "timestamp" in data


def test_append_correction_kwargs(temp_vault: FileVault):
    temp_vault.append_correction(
        original="college-documents",
        chosen="reference",
        note_title="Misclassified Paper",
    )
    corrections_file = temp_vault.root / "_corrections.jsonl"
    assert corrections_file.exists()
    lines = corrections_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["original_category"] == "college-documents"
    assert data["chosen_category"] == "reference"
    assert data["note_title"] == "Misclassified Paper"
    assert "timestamp" in data


def test_archive_note_and_attachment(temp_vault: FileVault, tmp_path: Path):
    note = Note(
        frontmatter={"title": "To Discard", "status": "inbox"},
        body="Discarded content body",
        filename="2026-11-03 - To-Discard.md",
    )
    inbox_note = temp_vault.write_note(note, "inbox")

    att_path = tmp_path / "receipt.png"
    att_path.write_text("fake-image-bytes", encoding="utf-8")

    archived_note, archived_att = temp_vault.archive_note(inbox_note, att_path)

    # Inbox note removed, archived note in _archive/ with status: archived
    assert not inbox_note.exists()
    assert archived_note.exists()
    assert archived_note.parent == temp_vault.root / "_archive"
    note_content = archived_note.read_text(encoding="utf-8")
    assert "status: archived" in note_content
    assert "status: inbox" not in note_content

    # Attachment moved to _archive/attachments/
    assert not att_path.exists()
    assert archived_att is not None
    assert archived_att.exists()
    assert archived_att.parent == temp_vault.root / "_archive" / "attachments"
    assert archived_att.read_text(encoding="utf-8") == "fake-image-bytes"


def test_archive_note_without_attachment(temp_vault: FileVault):
    note = Note(
        frontmatter={"title": "Discard No Att", "status": "inbox"},
        body="Solo note body",
        filename="2026-11-03 - Discard-Solo.md",
    )
    inbox_note = temp_vault.write_note(note, "inbox")

    archived_note, archived_att = temp_vault.archive_note(inbox_note)
    assert not inbox_note.exists()
    assert archived_note.exists()
    assert archived_att is None
    assert "status: archived" in archived_note.read_text(encoding="utf-8")


def test_archive_note_collision_handling(temp_vault: FileVault):
    note1 = Note(
        frontmatter={"title": "Dupe Archive", "status": "inbox"},
        body="Archive 1",
        filename="2026-11-03 - Dupe-Archive.md",
    )
    note2 = Note(
        frontmatter={"title": "Dupe Archive", "status": "inbox"},
        body="Archive 2",
        filename="2026-11-03 - Dupe-Archive.md",
    )
    inbox_note1 = temp_vault.write_note(note1, "inbox")
    archived1, _ = temp_vault.archive_note(inbox_note1)

    inbox_note2 = temp_vault.write_note(note2, "inbox")
    archived2, _ = temp_vault.archive_note(inbox_note2)

    assert archived1.name == "2026-11-03 - Dupe-Archive.md"
    assert archived2.name == "2026-11-03 - Dupe-Archive-2.md"
    assert archived1.exists()
    assert archived2.exists()


def test_archive_alias_backward_compatibility(temp_vault: FileVault, tmp_path: Path):
    note = Note(
        frontmatter={"title": "Legacy Discard", "status": "inbox"},
        body="Discarded content",
        filename="2026-11-03 - Legacy-Discard.md",
    )
    inbox_note = temp_vault.write_note(note, "inbox")
    att = tmp_path / "legacy.pdf"
    att.write_text("pdf-content", encoding="utf-8")

    res = temp_vault.archive(inbox_note, att)
    assert isinstance(res, Path)
    assert res.exists()
    assert res.parent == temp_vault.root / "_archive"
    assert "status: archived" in res.read_text(encoding="utf-8")


def test_vault_ingest_error_wrapping(tmp_path: Path):
    fake_file = tmp_path / "not_a_dir"
    fake_file.write_text("blocker", encoding="utf-8")

    with pytest.raises(IngestError) as exc_info:
        FileVault(fake_file / "subfolder")

    assert exc_info.value.stage == "write"
