"""Obsidian Vault adapter implementing VaultPort with atomic file operations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from ingest.models import IngestError, Note
from ingest.render import dump_frontmatter, render_markdown


class FileVault:
    """Manages reading, writing, moving, and archiving files in an Obsidian vault."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        try:
            self._init_directories()
        except OSError as e:
            raise IngestError("write", e) from e

    def _init_directories(self) -> None:
        """Initialize standard directory hierarchy."""
        self.root.mkdir(parents=True, exist_ok=True)
        folders = [
            "inbox",
            "college-documents",
            "possible-projects",
            "projects",
            "reference",
            "_archive",
        ]
        for folder in folders:
            (self.root / folder).mkdir(parents=True, exist_ok=True)

        att_root = self.root / "attachments"
        att_root.mkdir(parents=True, exist_ok=True)
        for cat in ["college-documents", "possible-projects", "projects", "reference"]:
            (att_root / cat).mkdir(parents=True, exist_ok=True)

        (self.root / "_archive" / "attachments").mkdir(parents=True, exist_ok=True)

    def write_note(self, note: Note, folder: str = "inbox") -> Path:
        """Write note markdown atomically into the target folder, resolving filename collisions."""
        try:
            target_dir = self.root / folder
            target_dir.mkdir(parents=True, exist_ok=True)

            base_name = note.filename
            stem = Path(base_name).stem
            suffix = Path(base_name).suffix or ".md"

            dest = target_dir / f"{stem}{suffix}"
            counter = 2
            while dest.exists():
                dest = target_dir / f"{stem}-{counter}{suffix}"
                counter += 1

            content = f"{dump_frontmatter(note.frontmatter)}\n\n{note.body}\n"

            # Atomic write via NamedTemporaryFile + os.replace in same directory
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target_dir, delete=False) as tf:
                temp_name = tf.name
                tf.write(content)
                tf.flush()
                os.fsync(tf.fileno())

            os.replace(temp_name, dest)
            return dest
        except Exception as e:
            if "temp_name" in locals() and os.path.exists(temp_name):
                try:
                    os.remove(temp_name)
                except OSError:
                    pass
            raise IngestError("write", e) from e

    def move_note(self, src: Path, folder: str) -> Path:
        """Move note to target folder, update status to filed (or archived), and resolve collisions."""
        try:
            src = Path(src)
            target_dir = self.root / folder
            target_dir.mkdir(parents=True, exist_ok=True)

            dest = target_dir / src.name
            counter = 2
            stem = src.stem
            suffix = src.suffix or ".md"
            while dest.exists() and dest.resolve() != src.resolve():
                dest = target_dir / f"{stem}-{counter}{suffix}"
                counter += 1

            content = src.read_text(encoding="utf-8")
            new_status = "archived" if folder == "_archive" else "filed"

            if re.search(r"^status:\s*.*$", content, flags=re.MULTILINE):
                content_updated = re.sub(
                    r"^status:\s*.*$",
                    f"status: {new_status}",
                    content,
                    flags=re.MULTILINE,
                )
            elif content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    content_updated = f"---{parts[1].rstrip()}\nstatus: {new_status}\n---{parts[2]}"
                else:
                    content_updated = f"status: {new_status}\n{content}"
            else:
                content_updated = f"---\nstatus: {new_status}\n---\n\n{content}"

            # Atomic move / replace
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target_dir, delete=False) as tf:
                temp_name = tf.name
                tf.write(content_updated)
                tf.flush()
                os.fsync(tf.fileno())

            os.replace(temp_name, dest)
            if src.resolve() != dest.resolve() and src.exists():
                try:
                    src.unlink()
                except OSError:
                    pass

            return dest
        except Exception as e:
            if "temp_name" in locals() and os.path.exists(temp_name):
                try:
                    os.remove(temp_name)
                except OSError:
                    pass
            raise IngestError("write", e) from e

    def store_attachment(self, src: Path, category: str, filename: str | None = None) -> Path:
        """Store an attachment in attachments/<category>/ preserving original or custom filename."""
        try:
            src = Path(src)
            dest_dir = self.root / "attachments" / category
            dest_dir.mkdir(parents=True, exist_ok=True)

            name = filename or src.name
            dest = dest_dir / name
            counter = 2
            stem = Path(name).stem
            suffix = Path(name).suffix
            while dest.exists():
                dest = dest_dir / f"{stem}-{counter}{suffix}"
                counter += 1

            with tempfile.NamedTemporaryFile(dir=dest_dir, delete=False) as tf:
                temp_name = tf.name

            shutil.copy2(str(src), temp_name)
            os.replace(temp_name, dest)
            return dest
        except Exception as e:
            if "temp_name" in locals() and os.path.exists(temp_name):
                try:
                    os.remove(temp_name)
                except OSError:
                    pass
            raise IngestError("write", e) from e

    def append_correction(
        self,
        entry: dict[str, Any] | str | None = None,
        original: str | None = None,
        chosen: str | None = None,
        note_title: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Append a correction entry to _corrections.jsonl."""
        try:
            record: dict[str, Any]
            if isinstance(entry, dict):
                record = dict(entry)
                if "timestamp" not in record:
                    record["timestamp"] = datetime.now(timezone.utc).isoformat()
            else:
                orig_val = original if original is not None else (entry if isinstance(entry, str) else kwargs.get("original", ""))
                chosen_val = chosen or kwargs.get("chosen", "")
                title_val = note_title or kwargs.get("note_title")
                record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "original_category": orig_val,
                    "chosen_category": chosen_val,
                    "note_title": title_val,
                    **kwargs,
                }
            corrections_file = self.root / "_corrections.jsonl"
            with open(corrections_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            raise IngestError("write", e) from e

    def archive_note(
        self,
        note_path: Path,
        attachment_path: Path | None = None,
    ) -> tuple[Path, Path | None]:
        """Move note and optional attachment to _archive/, updating note status to 'archived'."""
        try:
            note_path = Path(note_path)
            archive_dir = self.root / "_archive"
            archive_dir.mkdir(parents=True, exist_ok=True)

            dest_note = archive_dir / note_path.name
            counter = 2
            stem = note_path.stem
            suffix = note_path.suffix or ".md"
            while dest_note.exists() and dest_note.resolve() != note_path.resolve():
                dest_note = archive_dir / f"{stem}-{counter}{suffix}"
                counter += 1

            content = note_path.read_text(encoding="utf-8")
            if re.search(r"^status:\s*.*$", content, flags=re.MULTILINE):
                content_updated = re.sub(
                    r"^status:\s*.*$",
                    "status: archived",
                    content,
                    flags=re.MULTILINE,
                )
            elif content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    content_updated = f"---{parts[1].rstrip()}\nstatus: archived\n---{parts[2]}"
                else:
                    content_updated = f"status: archived\n{content}"
            else:
                content_updated = f"---\nstatus: archived\n---\n\n{content}"

            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=archive_dir, delete=False) as tf:
                temp_name = tf.name
                tf.write(content_updated)
                tf.flush()
                os.fsync(tf.fileno())

            os.replace(temp_name, dest_note)
            if note_path.resolve() != dest_note.resolve() and note_path.exists():
                try:
                    note_path.unlink()
                except OSError:
                    pass

            archived_att: Path | None = None
            if attachment_path is not None:
                attachment_path = Path(attachment_path)
                if attachment_path.exists():
                    archive_att_dir = self.root / "_archive" / "attachments"
                    archive_att_dir.mkdir(parents=True, exist_ok=True)
                    att_name = attachment_path.name
                    att_stem = Path(att_name).stem
                    att_suffix = Path(att_name).suffix
                    dest_att = archive_att_dir / att_name
                    counter = 2
                    while dest_att.exists() and dest_att.resolve() != attachment_path.resolve():
                        dest_att = archive_att_dir / f"{att_stem}-{counter}{att_suffix}"
                        counter += 1

                    if attachment_path.resolve() != dest_att.resolve():
                        with tempfile.NamedTemporaryFile(dir=archive_att_dir, delete=False) as tf:
                            temp_att = tf.name
                        shutil.copy2(str(attachment_path), temp_att)
                        os.replace(temp_att, dest_att)
                        try:
                            attachment_path.unlink()
                        except OSError:
                            pass
                    archived_att = dest_att

            return (dest_note, archived_att)
        except Exception as e:
            if "temp_name" in locals() and os.path.exists(temp_name):
                try:
                    os.remove(temp_name)
                except OSError:
                    pass
            if "temp_att" in locals() and os.path.exists(temp_att):
                try:
                    os.remove(temp_att)
                except OSError:
                    pass
            raise IngestError("write", e) from e

    def archive(self, src: Path, attachment_src: Path | None = None) -> Path:
        """Move note (and optional attachment) to _archive/."""
        archived_note, _ = self.archive_note(src, attachment_src)
        return archived_note
