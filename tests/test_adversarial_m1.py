"""Adversarial stress-testing suite for pure core modules (render.py, models.py, prompts.py).

Executes empirical challenges against:
1. Devanagari / Indic script slugification (matra and diacritic preservation)
2. Hand-rolled YAML frontmatter serialization (syntax validity & injection attacks)
3. Classification model boundary conditions (None coercion, lengths, tags)
4. render_body sentence fragmentation & formatting under extreme inputs
5. Prompt formatting with malformed / boundary inputs
"""
from datetime import date, datetime
import pytest
import yaml

from ingest.models import (
    Capture,
    CaptureKind,
    Category,
    Classification,
    SearchResult,
)
from ingest.prompts import (
    classify_prompt,
    project_prompt,
    extract_prompt,
)
from ingest.render import (
    build_frontmatter,
    build_note,
    dump_frontmatter,
    filename_for,
    get_usage,
    render_body,
    render_markdown,
    slug,
)
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# Section 1: Extreme Title Strings and Slug Handling
# ---------------------------------------------------------------------------

def test_slug_pure_emojis_fallback():
    """Pure emoji titles must fall back to 'untitled'."""
    assert slug("🎉🚀🔥") == "untitled"


def test_slug_mixed_emojis_and_text():
    """Emojis stripped, alphanumeric retained."""
    assert slug("🚀 Rocket Launch Plan") == "Rocket-Launch-Plan"


def test_slug_symbols_only_fallback():
    """Symbols-only strings must fall back to 'untitled'."""
    assert slug("!@#$%^&*()") == "untitled"
    assert slug("   ??? --- !!!   ") == "untitled"
    assert slug("----") == "untitled"


def test_slug_length_capping_500_chars():
    """500-char string strictly capped at 60 characters."""
    raw = "A" * 500
    res = slug(raw)
    assert len(res) == 60
    assert res == "A" * 60


def test_slug_devanagari_indic_diacritics():
    """Verify Devanagari vowel signs (matras), halants, and candrabindus are preserved across Indic scripts."""
    assert slug("परीक्षा परिणाम 2026") == "परीक्षा-परिणाम-2026"
    assert slug("छत्रपति शिवाजी महाराज टर्मिनस") == "छत्रपति-शिवाजी-महाराज-टर्मिनस"
    assert slug("IIT बॉम्बे CS101 Semester 1 Marksheet") == "IIT-बॉम्बे-CS101-Semester-1-Marksheet"
    assert slug("பல்கலைக்கழக தேர்வு முடிவுகள் 2026") == "பல்கலைக்கழக-தேர்வு-முடிவுகள்-2026"
    assert slug("বাংলা परीक्षा ২০২৬") == "বাংলা-परीक्षा-২০২৬"


def test_slug_indic_clean_truncation():
    """Verify 60-character truncation does not sever combining mark clusters."""
    # Matra at index 60 is dropped along with its base consonant rather than severed
    assert slug("A" * 59 + "की") == "A" * 59
    # Exact fit at 60 chars preserves the full syllable
    assert slug("A" * 58 + "की") == "A" * 58 + "की"
    assert len(slug("A" * 58 + "की")) == 60


def test_slug_windows_forbidden_characters():
    """Windows invalid filename chars: < > : \" / \\ | ? * must never appear in slug."""
    forbidden = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    dirty_title = 'Project <Test> : Guide "Draft" / Path \\ Sub | Query ? Star *'
    clean = slug(dirty_title)
    for ch in forbidden:
        assert ch not in clean
    assert clean == "Project-Test-Guide-Draft-Path-Sub-Query-Star"


# ---------------------------------------------------------------------------
# Section 2: Classification Model Edge Cases
# ---------------------------------------------------------------------------

def test_classification_title_none_rejected():
    """title=None must raise ValidationError with 'title cannot be null'."""
    with pytest.raises(ValidationError) as exc_info:
        Classification(
            category=Category.COLLEGE,
            title=None,  # type: ignore[arg-type]
            summary="A certificate with null title provided.",
        )
    assert "title cannot be null" in str(exc_info.value)


def test_classification_title_empty_and_whitespace_rejected():
    """Empty or whitespace-only titles must raise ValidationError."""
    with pytest.raises(ValidationError):
        Classification(category=Category.REFERENCE, title="", summary="Valid summary.")

    with pytest.raises(ValidationError):
        Classification(category=Category.REFERENCE, title="   \t\n  ", summary="Valid summary.")


def test_classification_title_max_length_boundary():
    """80 chars accepted, 81 chars and 500 chars rejected."""
    cls_80 = Classification(category=Category.REFERENCE, title="A" * 80, summary="Valid.")
    assert len(cls_80.title) == 80

    with pytest.raises(ValidationError):
        Classification(category=Category.REFERENCE, title="A" * 81, summary="Valid.")

    with pytest.raises(ValidationError):
        Classification(category=Category.REFERENCE, title="A" * 500, summary="Valid.")


def test_classification_summary_max_length_boundary():
    """600 chars accepted, 601 chars rejected."""
    cls_600 = Classification(category=Category.REFERENCE, title="Valid", summary="A" * 600)
    assert len(cls_600.summary) == 600

    with pytest.raises(ValidationError):
        Classification(category=Category.REFERENCE, title="Valid", summary="A" * 601)


def test_classification_tags_max_length_boundary():
    """6 tags accepted, 7 tags rejected."""
    cls_6 = Classification(category=Category.REFERENCE, title="Valid", summary="Valid", tags=["t"] * 6)
    assert len(cls_6.tags) == 6

    with pytest.raises(ValidationError):
        Classification(category=Category.REFERENCE, title="Valid", summary="Valid", tags=["t"] * 7)


def test_classification_null_and_nonstandard_metadata():
    """Classification accepts null college metadata and arbitrary date strings."""
    cls = Classification(
        category=Category.COLLEGE,
        title="College Document",
        summary="Summary of college document.",
        document_type=None,
        issuer=None,
        issued_date="Yesterday morning",  # Accepts non-standard date format
        reference_ids=[],
    )
    assert cls.document_type is None
    assert cls.issuer is None
    assert cls.issued_date == "Yesterday morning"
    assert cls.reference_ids == []


# ---------------------------------------------------------------------------
# Section 3: YAML Frontmatter Injection & Parsing Integrity
# ---------------------------------------------------------------------------

def test_yaml_frontmatter_injection_failures():
    """Verify dump_frontmatter safely quotes reserved characters, directives, and paths for roundtrip PyYAML parsing."""
    adversarial_titles = [
        "*Important Note*",
        "[Draft] Project Proposal",
        "{WIP} System Specs",
        "!Urgent Notice",
        "%Discount Summary",
        "|Block Scalar Style",
        ">Folded Scalar Style",
        "Colon at end:",
        "Line 1\nLine 2",
        "- Draft Note",
        "yes",
        "no",
        "true",
        "false",
        "null",
        "~",
        "123",
        "042",
        "3.14",
        "-5",
        "  spaces  ",
        "",
        r"C:\Users\ashis\vault",
        "परीक्षा परिणाम 2026",
    ]
    for adversarial_title in adversarial_titles:
        fm = {
            "title": adversarial_title,
            "category": "reference",
            "created": "2026-11-03",
            "source": "telegram",
            "status": "inbox",
            "tags": [adversarial_title, "valid-tag"],
        }
        raw_yaml = dump_frontmatter(fm)
        parsed = yaml.safe_load(raw_yaml.replace("---", ""))
        assert parsed["title"] == adversarial_title
        assert parsed["tags"][0] == adversarial_title


def test_yaml_frontmatter_tags_boolean_coercion():
    """Verify tags 'yes' and 'no' are quoted and preserved as string types."""
    fm = {
        "title": "Tags Test",
        "category": "reference",
        "created": "2026-11-03",
        "source": "telegram",
        "status": "inbox",
        "tags": ["yes", "no", "valid-tag"],
    }
    raw_yaml = dump_frontmatter(fm)
    parsed = yaml.safe_load(raw_yaml.replace("---", ""))
    assert parsed["tags"] == ["yes", "no", "valid-tag"]
    assert all(isinstance(t, str) for t in parsed["tags"])


# ---------------------------------------------------------------------------
# Section 4: render_body Formatting & Edge Cases
# ---------------------------------------------------------------------------

def test_render_body_reference_abbreviation_preservation():
    """Sentence splitting must preserve abbreviations and not create single-word bullets."""
    capture = Capture(chat_id=1, message_id=1, kind=CaptureKind.TEXT, text="Note")
    cls = Classification(
        category=Category.REFERENCE,
        title="Experiment Notes",
        summary="Dr. Smith completed trial No. 42 at 3.14 PM. It cost Rs. 500 total! Promising results, e.g. 95% yield.",
    )
    body = render_body(capture, cls)
    assert "- Dr. Smith completed trial No. 42 at 3.14 PM." in body
    assert "- It cost Rs. 500 total!" in body
    assert "- Promising results, e.g. 95% yield." in body
    bullet_lines = [line.strip() for line in body.splitlines() if line.strip().startswith("- ")]
    assert bullet_lines == [
        "- Dr. Smith completed trial No. 42 at 3.14 PM.",
        "- It cost Rs. 500 total!",
        "- Promising results, e.g. 95% yield.",
    ]
    assert "- Dr." not in bullet_lines
    assert "- It cost Rs." not in bullet_lines
    assert "- Promising results, e.g." not in bullet_lines


def test_render_body_college_missing_optional_fields():
    """College documents with missing optional fields render cleanly without crashing."""
    capture = Capture(chat_id=1, message_id=1, kind=CaptureKind.DOCUMENT, received_at=datetime(2026, 11, 3))
    cls = Classification(
        category=Category.COLLEGE,
        title="College Doc",
        summary="Summary text.",
        document_type=None,
        issuer=None,
        issued_date=None,
        reference_ids=[],
    )
    body = render_body(capture, cls)
    assert "| Document type | Unknown |" in body
    assert "| Issuer | Unknown |" in body
    assert "| Issued | Unknown |" in body
    assert "| Reference | None |" in body
    assert "Academic document preserved in vault" in body


def test_render_body_project_multiline_idea():
    """Project body with multiline idea: lines after line 1 lose blockquote marker."""
    capture = Capture(
        chat_id=1,
        message_id=1,
        kind=CaptureKind.TEXT,
        text="First line of idea.\nSecond line of idea.\nThird line of idea.",
    )
    cls = Classification(category=Category.PROJECT, title="Project Idea", summary="Summary")
    body = render_body(capture, cls)
    assert "> First line of idea.\nSecond line of idea." in body


# ---------------------------------------------------------------------------
# Section 5: Prompts Edge Cases
# ---------------------------------------------------------------------------

def test_classify_prompt_with_none_and_empty_translation():
    """classify_prompt omits translation block when translation is None or whitespace."""
    capture = Capture(chat_id=1, message_id=1, kind=CaptureKind.TEXT)
    _, u_none = classify_prompt(capture, "orig", None)
    assert "ENGLISH TRANSLATION:" not in u_none

    _, u_empty = classify_prompt(capture, "orig", "   \n\t  ")
    assert "ENGLISH TRANSLATION:" not in u_empty


def test_project_prompt_empty_search_results():
    """project_prompt outputs explicit fallback when search results are empty."""
    _, u = project_prompt("Idea", [])
    assert "(No search results available)" in u


def test_extract_prompt_structure():
    """extract_prompt returns system instruction and user ORIGINAL block."""
    s, u = extract_prompt("Raw document content", "English version")
    assert "extract metadata" in s.lower()
    assert "ORIGINAL:\nRaw document content" in u
    assert "ENGLISH TRANSLATION:\nEnglish version" in u
