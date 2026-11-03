"""Pure note rendering and filename formatting engine.

Imports only models.py and the standard library. Zero I/O, zero async.
"""

from __future__ import annotations

from datetime import date
import json
import re
from typing import Any
import unicodedata

from ingest.models import Capture, Category, Classification, Note

# Fixed 12-entry lookup for academic paperwork usage
USAGE_MAP: dict[str, str] = {
    "marksheet": (
        "Proof of academic grades, marks, and subject completion for a specific academic "
        "semester or examination term."
    ),
    "transcript": (
        "Official comprehensive record of all courses taken, grades attained, and cumulative "
        "GPA across the full degree program."
    ),
    "degree certificate": (
        "Formal legal credential certifying graduation and conferral of the academic degree "
        "by the university or governing board."
    ),
    "provisional certificate": (
        "Interim certificate verifying completion of degree requirements prior to formal "
        "convocation and diploma issuance."
    ),
    "fee receipt": (
        "Financial proof of tuition, hostel, examination, or institutional payment for accounting, "
        "reimbursements, and tax deduction."
    ),
    "admit card": (
        "Admit card and examination hall ticket verifying candidate eligibility, roll number, seat allocation, "
        "and examination timetable."
    ),
    "id card": (
        "Institutional identity credential verifying current active student enrollment, campus "
        "facility access, and student privileges."
    ),
    "bonafide certificate": (
        "Institutional attestation certifying active student status, required for passport applications, "
        "student visas, and official concessions."
    ),
    "admission letter": (
        "Official institutional offer letter and enrollment confirmation for admission into an "
        "academic program."
    ),
    "transfer certificate": (
        "Official release document issued upon leaving an educational institution to facilitate "
        "admission into another institution."
    ),
    "migration certificate": (
        "Certificate issued by a university or education board permitting a student to transition "
        "to another university or jurisdiction."
    ),
    "character certificate": (
        "Official institutional statement attesting to student conduct, ethical standing, and "
        "discipline during their enrollment period."
    ),
}

COLLEGE_BODY = """\
## Summary
{summary}

## Key details
| Field | Value |
|---|---|
| Document type | {document_type} |
| Issuer | {issuer} |
| Issued | {issued_date} |
| Reference | {reference_ids} |

## What this is for
{usage}
"""

PROJECT_BODY = """\
## One-liner
> {idea}
"""

REFERENCE_BODY = """\
## Summary
{summary}

## Key points
{points}
"""


def get_usage(document_type: str | None) -> str:
    """Deterministic lookup for academic document purpose."""
    if not document_type:
        return "Academic document preserved in vault for administrative, educational, or professional reference."
    key = document_type.strip().lower()
    return USAGE_MAP.get(
        key,
        f"Official academic paperwork ({document_type}) preserved for institutional, educational, or legal verification.",
    )


def slug(title: str) -> str:
    """Alphanumeric slug sanitization with Indic diacritic preservation, 60-char cap, fallback to 'untitled'."""
    if not title or not isinstance(title, str):
        return "untitled"

    # Normalize to Unicode NFC standard form
    norm = unicodedata.normalize("NFC", title)

    # Retain Unicode letters (L*), numbers (N*), combining marks (M*), hyphens, underscores, and whitespace
    cleaned = [
        ch
        for ch in norm
        if unicodedata.category(ch)[0] in ("L", "N", "M") or ch in ("-", "_") or ch.isspace()
    ]
    slug_str = "-".join("".join(cleaned).split())

    # Strip any leading combining marks that have no base character
    while slug_str and unicodedata.category(slug_str[0]).startswith("M"):
        slug_str = slug_str[1:]
    slug_str = slug_str.strip("-")

    # Cleanly truncate to at most 60 characters without severing combining mark clusters
    if len(slug_str) > 60:
        cut = 60
        # If the discarded character at index 60 is a combining mark, it was attached to
        # the base character before it. Roll back past any preceding marks and the base
        # character itself to avoid leaving a severed/corrupted syllable.
        if unicodedata.category(slug_str[cut]).startswith("M"):
            while cut > 0 and unicodedata.category(slug_str[cut - 1]).startswith("M"):
                cut -= 1
            if cut > 0:
                cut -= 1
        slug_str = slug_str[:cut].strip("-")

    return slug_str or "untitled"


def filename_for(title: str, day: date) -> str:
    """Format standard Obsidian filename: YYYY-MM-DD - Slug.md."""
    return f"{day.isoformat()} - {slug(title)}.md"


def build_frontmatter(
    capture: Capture,
    cls: Classification,
    attachment: str | None = None,
    source_language: str | None = None,
) -> dict[str, Any]:
    """Construct ordered dictionary for Obsidian YAML frontmatter."""
    fm: dict[str, Any] = {
        "title": cls.title,
        "category": cls.category.value,
        "created": capture.received_at.date().isoformat(),
        "source": "telegram",
        "status": "inbox",
        "tags": cls.tags,
    }
    if capture.filename:
        fm["original_filename"] = capture.filename
    if attachment:
        fm["attachment"] = attachment
    if source_language:
        fm["source_language"] = source_language
    if cls.category == Category.COLLEGE:
        fm.update({
            "document_type": cls.document_type,
            "issuer": cls.issuer,
            "issued_date": cls.issued_date,
            "reference_ids": cls.reference_ids,
        })
    return fm


_YAML_BOOLEAN_NULL_LITERALS: set[str] = {
    "true", "false", "yes", "no", "on", "off", "y", "n", "null", "~"
}
_YAML_SPECIAL_PREFIXES: set[str] = set("*&!|%>@`[]{}?:,#~\"'")


def _needs_yaml_quotes(s: str) -> bool:
    """Determine whether a string scalar requires YAML quoting."""
    if not s or s.strip() != s:
        return True
    if s[0] in _YAML_SPECIAL_PREFIXES:
        return True
    if s.startswith("- ") or s == "-":
        return True
    if ": " in s or s.endswith(":") or ":\t" in s:
        return True
    if any(ch in s for ch in ('"', "'", "`", "@", "#", "\n", "\r", "\\")):
        return True
    if s.lower() in _YAML_BOOLEAN_NULL_LITERALS:
        return True
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        int(s, 0)
        return True
    except (ValueError, TypeError):
        pass
    if s.lower() in (".nan", ".inf", "+.inf", "-.inf"):
        return True
    return False


def _format_yaml_scalar(val: Any) -> str:
    """Format scalar value for strict YAML serialization, quoting when necessary."""
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, date):
        return val.isoformat()
    if isinstance(val, str):
        if _needs_yaml_quotes(val):
            return json.dumps(val, ensure_ascii=False)
        return val
    return str(val)


def dump_frontmatter(fm: dict[str, Any]) -> str:
    """Serialize frontmatter dictionary to strict YAML format preserving key insertion order."""
    lines = ["---"]
    for key, val in fm.items():
        if isinstance(val, (list, tuple)):
            if not val:
                lines.append(f"{key}: []")
            else:
                lines.append(f"{key}:")
                for item in val:
                    lines.append(f"  - {_format_yaml_scalar(item)}")
        elif isinstance(val, dict):
            if not val:
                lines.append(f"{key}: {{}}")
            else:
                lines.append(f"{key}:")
                for sk, sv in val.items():
                    lines.append(f"  {sk}: {_format_yaml_scalar(sv)}")
        else:
            lines.append(f"{key}: {_format_yaml_scalar(val)}")
    lines.append("---")
    return "\n".join(lines)


PREFIX_ABBREVIATIONS: frozenset[str] = frozenset({
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "rev", "st",
    "e.g", "eg", "i.e", "ie", "vs", "al", "fig", "no", "rs",
    "approx", "appx", "dept", "vol", "inc", "ltd", "co", "corp",
})


def split_sentences(text: str) -> list[str]:
    """Split text into complete sentences, avoiding abbreviation fragmentation and orphan tokens."""
    raw_segments = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not raw_segments:
        return []

    merged: list[str] = []
    current = raw_segments[0]

    for next_seg in raw_segments[1:]:
        words = current.split()
        last_word = words[-1].lower().rstrip(".!?,:;") if words else ""

        is_abbrev = last_word in PREFIX_ABBREVIATIONS
        is_initial = bool(re.fullmatch(r"[A-Za-z]\.", words[-1])) if words else False
        next_starts_lower = next_seg[0].islower() if next_seg else False
        curr_is_single = len(words) <= 1 or len(current.strip()) <= 3
        next_is_single = len(next_seg.split()) <= 1 or len(next_seg.strip()) <= 3

        if is_abbrev or is_initial or next_starts_lower or curr_is_single or next_is_single:
            current = f"{current} {next_seg}"
        else:
            merged.append(current)
            current = next_seg

    if len(current.split()) <= 1 or len(current.strip()) <= 3:
        if merged:
            merged[-1] = f"{merged[-1]} {current}"
        else:
            merged.append(current)
    else:
        merged.append(current)

    return merged


def render_body(capture: Capture, cls: Classification, extra: str | None = None) -> str:
    """Render the Markdown body for the given classification category."""
    extra_block = f"\n\n{extra.strip()}" if extra and extra.strip() else ""

    if cls.category == Category.COLLEGE:
        ref_ids = ", ".join(cls.reference_ids) if cls.reference_ids else "None"
        usage_text = get_usage(cls.document_type)
        base = COLLEGE_BODY.format(
            summary=cls.summary,
            document_type=cls.document_type or "Unknown",
            issuer=cls.issuer or "Unknown",
            issued_date=cls.issued_date or "Unknown",
            reference_ids=ref_ids,
            usage=usage_text,
        ).strip()
        return f"{base}{extra_block}"

    if cls.category == Category.PROJECT:
        idea_text = capture.text or capture.caption or cls.summary
        base = PROJECT_BODY.format(idea=idea_text.strip()).strip()
        return f"{base}{extra_block}"

    # Category.REFERENCE
    sentences = split_sentences(cls.summary)
    points_text = "\n".join(f"- {s}" for s in sentences) if sentences else f"- {cls.summary}"
    base = REFERENCE_BODY.format(summary=cls.summary, points=points_text).strip()
    return f"{base}{extra_block}"


def render_markdown(note: Note) -> str:
    """Format full Markdown note content with frontmatter and body."""
    fm_str = dump_frontmatter(note.frontmatter)
    return f"{fm_str}\n\n{note.body}\n"


def build_note(
    capture: Capture,
    cls: Classification,
    attachment: str | None = None,
    body_extra: str | None = None,
    source_language: str | None = None,
) -> Note:
    """Pure note constructor creating frontmatter, body, and filename."""
    fm = build_frontmatter(capture, cls, attachment=attachment, source_language=source_language)
    body = render_body(capture, cls, extra=body_extra)
    fn = filename_for(cls.title, capture.received_at.date())
    return Note(frontmatter=fm, body=body, filename=fn)
