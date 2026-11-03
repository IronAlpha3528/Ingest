from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from ingest.models import Note, SearchResult


class LlmPort(Protocol):
    async def complete_json(self, system: str, user: str) -> dict[str, Any]: ...
    async def complete_text(self, system: str, user: str, stage: str = "classify") -> str: ...


class OcrPort(Protocol):
    async def extract_text(self, path: Path) -> str: ...


class TextPort(Protocol):
    """Sarvam language identification and translation."""
    async def detect(self, text: str) -> str: ...                  # BCP-47, e.g. "hi-IN"
    async def to_english(self, text: str, source: str) -> str: ...


class SearchPort(Protocol):
    async def search(self, query: str, k: int = 5) -> list[SearchResult]: ...


class VaultPort(Protocol):
    def write_note(self, note: Note, folder: str) -> Path: ...
    def move_note(self, src: Path, folder: str) -> Path: ...
    def store_attachment(self, src: Path, category: str) -> Path: ...
