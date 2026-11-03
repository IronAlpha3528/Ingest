from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Category(StrEnum):
    COLLEGE = "college-documents"
    PROJECT = "possible-projects"
    REFERENCE = "reference"


class CaptureKind(StrEnum):
    DOCUMENT = "document"
    PHOTO = "photo"
    TEXT = "text"


class Capture(BaseModel):
    """One inbound message, normalised. Built by bot.py, consumed by pipeline.py."""
    chat_id: int
    message_id: int
    kind: CaptureKind
    caption: str | None = None
    filename: str | None = None
    local_path: Path | None = None      # set for DOCUMENT and PHOTO
    text: str | None = None             # set for TEXT
    received_at: datetime = Field(default_factory=datetime.now)


class Classification(BaseModel):
    """The LLM's verdict. The schema is also the prompt contract."""
    category: Category
    title: str = Field(max_length=80)
    summary: str = Field(max_length=600)
    tags: list[str] = Field(default_factory=list, max_length=6)
    document_type: str | None = None
    issuer: str | None = None
    issued_date: str | None = None
    reference_ids: list[str] = Field(default_factory=list)

    @field_validator("title", mode="before")
    @classmethod
    def title_is_single_line(cls, v: Any) -> str:
        if v is None:
            raise ValueError("title cannot be null")
        if not isinstance(v, str):
            v = str(v)
        v = " ".join(v.split())
        if not v:
            raise ValueError("title is empty")
        return v


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class Note(BaseModel):
    """A rendered note, not yet written."""
    frontmatter: dict[str, Any]
    body: str
    filename: str


class Proposal(BaseModel):
    """What bot.py needs to render a message and its buttons."""
    note: Note
    category: Category
    preview: str


class IngestError(Exception):
    def __init__(self, stage: str, cause: Exception | str):
        self.stage = stage
        self.cause = cause
        super().__init__(f"{stage}: {cause}")
