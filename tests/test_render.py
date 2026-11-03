"""Phase 1 unit tests and golden file assertions for pure note renderer."""

from datetime import date, datetime
from pathlib import Path
import pytest

from ingest.models import Capture, CaptureKind, Category, Classification
from ingest.render import (
    build_frontmatter,
    build_note,
    filename_for,
    get_usage,
    render_markdown,
    slug,
)

GOLDEN_DIR = Path(__file__).parent / "golden"


@pytest.fixture
def college_fixture():
    capture = Capture(
        chat_id=123456,
        message_id=42,
        kind=CaptureKind.DOCUMENT,
        caption="Semester 6 marksheet",
        filename="marksheet_sem6.pdf",
        local_path=Path("/tmp/marksheet_sem6.pdf"),
        received_at=datetime(2026, 11, 3, 10, 30, 0),
    )
    cls = Classification(
        category=Category.COLLEGE,
        title="Semester 6 Final Marksheet",
        summary="Official university marksheet for Semester 6 Bachelor of Technology in Computer Science. Contains detailed subject-wise grades, total credits earned, and semester GPA.",
        tags=["college", "marksheet", "btech", "academics"],
        document_type="Marksheet",
        issuer="Mumbai University",
        issued_date="2026-06-15",
        reference_ids=["MU-2026-BTECH-88421"],
    )
    attachment = "attachments/college-documents/marksheet_sem6.pdf"
    return capture, cls, attachment


@pytest.fixture
def project_fixture():
    capture = Capture(
        chat_id=123456,
        message_id=43,
        kind=CaptureKind.TEXT,
        text="An app that tracks my hostel mess bills and splits extras",
        received_at=datetime(2026, 11, 3, 11, 0, 0),
    )
    cls = Classification(
        category=Category.PROJECT,
        title="Hostel Mess Bill Tracker",
        summary="A mobile application designed for university students to log and audit monthly hostel mess expenses. It tracks daily attendance, dietary extras, and mess rebates with automated bill reconciliation against administration notices.",
        tags=["project", "finance", "hostel", "utility"],
        document_type=None,
        issuer=None,
        issued_date=None,
        reference_ids=[],
    )
    body_extra = """\
## Description
A mobile application designed for university students to log and audit monthly hostel mess expenses. It tracks daily attendance, dietary extras, and mess rebates with automated bill reconciliation against administration notices.

Targeted at college residents who experience billing discrepancies or need transparent expense sharing for shared meal subscriptions.

## Similar projects and competitors
- **Splitwise** — General expense splitting for groups without mess-specific billing cycles. <https://www.splitwise.com>
- **MessWala** — Campus mess management focused on caterers rather than student auditing. <https://example.com/messwala>

## Research areas
- OCR parsing for physical mess coupon booklets and handwritten register slips
- Push notification triggers for monthly mess rebate submission deadlines
- Offline-first local data synchronization for poor campus hostel connectivity
- Integration with UPI deep links for settling room-level shared extras

## Anticipated problems
- **Mess invoice variety** — different hostels use wildly varying invoice formats and billing formulas. Support flexible formula templates.
- **Student compliance** — students forget to log daily extras. Implement quick one-tap lockscreen widgets.

## Open questions
- Should the app support multi-canteen switching for campus food courts?
- Can hostel student committees export aggregate discrepancy reports directly to mess administrators?"""
    return capture, cls, body_extra


@pytest.fixture
def reference_fixture():
    capture = Capture(
        chat_id=123456,
        message_id=44,
        kind=CaptureKind.TEXT,
        text="Functional Core, Imperative Shell architecture isolates pure domain logic from side effects. Tests run fast and deterministic because the core has zero I/O.",
        received_at=datetime(2026, 11, 3, 11, 15, 0),
    )
    cls = Classification(
        category=Category.REFERENCE,
        title="Functional Core Imperative Shell Architecture",
        summary="An architectural pattern that isolates pure business logic in an immutable core while confining side effects and I/O to outer boundaries. Pure functions make domain rules easy to reason about and fast to test without mocks. Shell components orchestrate dependencies and adapt external services.",
        tags=["architecture", "functional-programming", "software-design"],
        document_type=None,
        issuer=None,
        issued_date=None,
        reference_ids=[],
    )
    return capture, cls


def test_college_golden(college_fixture):
    capture, cls, attachment = college_fixture
    note = build_note(capture, cls, attachment=attachment)
    rendered = render_markdown(note)
    expected = (GOLDEN_DIR / "college.md").read_text(encoding="utf-8")
    assert rendered == expected


def test_project_golden(project_fixture):
    capture, cls, body_extra = project_fixture
    note = build_note(capture, cls, body_extra=body_extra)
    rendered = render_markdown(note)
    expected = (GOLDEN_DIR / "project.md").read_text(encoding="utf-8")
    assert rendered == expected


def test_reference_golden(reference_fixture):
    capture, cls = reference_fixture
    note = build_note(capture, cls)
    rendered = render_markdown(note)
    expected = (GOLDEN_DIR / "reference.md").read_text(encoding="utf-8")
    assert rendered == expected


def test_slug_basic():
    assert slug("Hello World!") == "Hello-World"


def test_slug_truncation():
    long_title = "A" * 80
    assert len(slug(long_title)) <= 60
    assert slug(long_title) == "A" * 60


def test_filename_format():
    assert filename_for("My Note", date(2026, 11, 3)) == "2026-11-03 - My-Note.md"


def test_frontmatter_college_fields(college_fixture):
    capture, cls, attachment = college_fixture
    fm = build_frontmatter(capture, cls, attachment=attachment)
    assert fm["document_type"] == "Marksheet"
    assert fm["issuer"] == "Mumbai University"
    assert fm["issued_date"] == "2026-06-15"
    assert fm["reference_ids"] == ["MU-2026-BTECH-88421"]


def test_frontmatter_no_college_fields(reference_fixture):
    capture, cls = reference_fixture
    fm = build_frontmatter(capture, cls)
    assert "document_type" not in fm
    assert "issuer" not in fm
    assert "issued_date" not in fm
    assert "reference_ids" not in fm


def test_classification_title_validator():
    # Extra whitespace is collapsed
    cls = Classification(
        category=Category.REFERENCE,
        title="  Multiple   Spaces   In   Title  ",
        summary="A short summary testing title validation behavior.",
        tags=["test"],
    )
    assert cls.title == "Multiple Spaces In Title"

    # Empty or whitespace-only title raises ValueError
    with pytest.raises(ValueError):
        Classification(
            category=Category.REFERENCE,
            title="   ",
            summary="A short summary testing empty title.",
            tags=["test"],
        )


def test_slug_special_characters_and_fallback():
    assert slug("   ?!!!   ") == "untitled"
    assert slug("Special / Characters & Symbols!") == "Special-Characters-Symbols"


def test_usage_lookup_and_fallback():
    assert "grades" in get_usage("Marksheet").lower()
    assert "admit card" in get_usage("Admit Card").lower()
    assert "educational" in get_usage("Unknown Document").lower()
    assert "administrative" in get_usage(None).lower()
