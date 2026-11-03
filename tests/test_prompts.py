from __future__ import annotations

import json
from datetime import datetime

from ingest.models import Capture, CaptureKind, Classification, SearchResult
from ingest.prompts import (
    CLASSIFY_SYSTEM,
    PROJECT_SYSTEM,
    classify_prompt,
    extract_prompt,
    project_prompt,
)


def _make_sample_capture(**kwargs) -> Capture:
    default_kwargs = {
        "chat_id": 1001,
        "message_id": 2001,
        "kind": CaptureKind.TEXT,
        "text": "Sample text",
        "received_at": datetime(2026, 11, 3, 12, 0, 0),
    }
    default_kwargs.update(kwargs)
    return Capture(**default_kwargs)


def test_classify_system_contains_schema():
    capture = _make_sample_capture()
    system, _ = classify_prompt(capture, "Sample text", None)

    # Asserts that the generated schema with category definition is embedded in system prompt
    assert "Schema:" in system
    assert '"category"' in system
    assert '"college-documents"' in system
    assert '"possible-projects"' in system
    assert '"reference"' in system


def test_classify_user_has_original_block():
    capture = _make_sample_capture(filename="doc.pdf", caption="Academic record")
    original_text = "Roll No: 12345, Marks: 95%"
    _, user = classify_prompt(capture, original_text, None)

    assert "Filename: doc.pdf" in user
    assert "Caption: Academic record" in user
    assert "ORIGINAL:" in user
    assert original_text in user


def test_classify_user_has_translation_block():
    capture = _make_sample_capture()
    original_text = "यह एक कॉलेज अंकतालिका है"
    translation_text = "This is a college marksheet"
    _, user = classify_prompt(capture, original_text, translation_text)

    assert "ORIGINAL:" in user
    assert original_text in user
    assert "ENGLISH TRANSLATION:" in user
    assert translation_text in user


def test_classify_no_translation_block_when_english():
    capture = _make_sample_capture()
    original_text = "English memo about project budget"
    _, user = classify_prompt(capture, original_text, None)

    assert "ORIGINAL:" in user
    assert original_text in user
    assert "ENGLISH TRANSLATION:" not in user


def test_project_prompt_contains_urls():
    results = [
        SearchResult(title="Tool A", url="https://example.com/tool-a", snippet="A tracker tool"),
        SearchResult(title="Service B", url="https://service.org/b", snippet="Mess billing system"),
    ]
    _, user = project_prompt("hostel mess bill tracker", results)

    assert "Idea: hostel mess bill tracker" in user
    assert "https://example.com/tool-a" in user
    assert "https://service.org/b" in user


def test_project_prompt_all_snippets():
    results = [
        SearchResult(title="Tool A", url="https://example.com/tool-a", snippet="A tracker tool"),
        SearchResult(title="Service B", url="https://service.org/b", snippet="Mess billing system"),
    ]
    _, user = project_prompt("hostel mess bill tracker", results)

    assert "A tracker tool" in user
    assert "Mess billing system" in user
    assert "Tool A" in user
    assert "Service B" in user


def test_extract_prompt_returns_tuple():
    original = "Details from document"
    translation = "English details"
    result = extract_prompt(original, translation)

    assert isinstance(result, tuple)
    assert len(result) == 2
    system, user = result
    assert isinstance(system, str) and len(system) > 0
    assert isinstance(user, str) and len(user) > 0
    assert "ORIGINAL:" in user
    assert "ENGLISH TRANSLATION:" in user


def test_project_prompt_url_constraint():
    # Phase 8 constraint validation (Design §6.2)
    assert "Use only URLs from the search results provided" in PROJECT_SYSTEM
