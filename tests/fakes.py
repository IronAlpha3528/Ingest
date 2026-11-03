"""In-memory and temporary directory test fakes for fast, isolated pipeline testing.

Provides zero-network implementations of all ports in src/ingest/adapters/ports.py.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from typing import Any

from ingest.models import Note, SearchResult
from ingest.render import render_markdown


class FakeLlm:
    """In-memory LLM port fake for structured JSON and text completions."""

    def __init__(self, payload: dict[str, Any] | Exception | None = None, text: str | Exception = "## Description\nA test."):
        self.payload = payload if payload is not None else {}
        self.text = text
        self.calls: list[tuple[str, str]] = []

    async def complete_json(self, system: str, user: str) -> dict[str, Any]:
        self.calls.append((system, user))
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload

    async def complete_text(self, system: str, user: str, stage: str = "classify") -> str:
        self.calls.append((system, user))
        if isinstance(self.text, Exception):
            raise self.text
        return self.text


class FakeOcr:
    """In-memory OCR port fake returning predefined extracted text."""

    def __init__(self, text: str | Exception = "Extracted document text"):
        self.text = text
        self.calls: list[Path] = []

    async def extract_text(self, path: Path) -> str:
        self.calls.append(path)
        if isinstance(self.text, Exception):
            raise self.text
        return self.text


class FakeText:
    """In-memory language identification and translation port fake."""

    def __init__(self, lang: str = "en-IN", translation: str | None = None):
        self.lang = lang
        self.translation = translation
        self.detect_calls: list[str] = []
        self.translate_calls: list[tuple[str, str]] = []

    async def detect(self, text: str) -> str:
        self.detect_calls.append(text)
        return self.lang

    async def to_english(self, text: str, source: str) -> str:
        self.translate_calls.append((text, source))
        return self.translation if self.translation is not None else text


class FakeSearch:
    """In-memory web search port fake returning preconfigured SearchResult objects."""

    def __init__(self, results: list[SearchResult] | None = None):
        self.results = results if results is not None else [
            SearchResult(
                title="Example Project",
                url="https://example.com",
                snippet="Example project competitor snippet",
            )
        ]
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, k: int = 5) -> list[SearchResult]:
        self.calls.append((query, k))
        return self.results


class TempVault:
    """Real filesystem vault fake operating inside an isolated temporary directory.

    Emulates all VaultPort operations with exact folder hierarchy and collision handling.
    """

    def __init__(self, root: Path | None = None):
        if root is None:
            self._temp_dir = tempfile.TemporaryDirectory()
            self.root = Path(self._temp_dir.name)
        else:
            self._temp_dir = None
            self.root = root

        # Initialize vault directory tree
        for folder in ["inbox", "college-documents", "possible-projects", "reference", "_archive"]:
            (self.root / folder).mkdir(parents=True, exist_ok=True)
        for cat in ["college-documents", "possible-projects", "reference"]:
            (self.root / "attachments" / cat).mkdir(parents=True, exist_ok=True)

    def write_note(self, note: Note, folder: str) -> Path:
        target_dir = self.root / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        base_name = note.filename
        stem = Path(base_name).stem
        suffix = Path(base_name).suffix

        target = target_dir / base_name
        counter = 2
        while target.exists():
            target = target_dir / f"{stem}-{counter}{suffix}"
            counter += 1

        content = render_markdown(note)
        target.write_text(content, encoding="utf-8")
        return target

    def move_note(self, src: Path, folder: str) -> Path:
        target_dir = self.root / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        dest = target_dir / src.name
        counter = 2
        stem = src.stem
        suffix = src.suffix
        while dest.exists():
            dest = target_dir / f"{stem}-{counter}{suffix}"
            counter += 1
        shutil.move(str(src), str(dest))
        return dest

    def store_attachment(self, src: Path, category: str) -> Path:
        dest_dir = self.root / "attachments" / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        shutil.copy2(str(src), str(dest))
        return dest

    def cleanup(self) -> None:
        if self._temp_dir:
            self._temp_dir.cleanup()
