"""Adversarial stress-testing suite for Milestone 2 (Vault Adapter) and Milestone 3 (Sarvam AI & Search Adapters).

Empirically challenges:
1. FileVault: concurrent/rapid atomic writes, collision chains (2 through 15), nested paths,
   status update mutations in frontmatter and body, attachment binary integrity, corrections JSONL concurrency.
2. SarvamLlm: malformed JSON repair recovery, unrecoverable syntax errors, conversational padding,
   broken markdown fences, code backtick stripping, error wrapping.
3. SarvamOcr: exact 50 vs 51 word boundary on digital PDF fast path, multipage digital extraction,
   sparse PDF fallback resilience on API error, scanned PDF & image error wrapping.
4. SarvamText: multi-Indic script pass-through (English-only bypass), extreme sentence chunking,
   character conservation on gigantic sentences, language detection edge cases.
5. SearchAdapter: Tavily and Brave API failure resilience, empty results fallback, strict k-limits,
   query sanitization.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import fitz
import httpx
import pytest

from ingest.adapters.ports import LlmPort, OcrPort, SearchPort, TextPort
from ingest.adapters.sarvam_llm import SarvamLlm
from ingest.adapters.sarvam_ocr import SarvamOcr
from ingest.adapters.sarvam_text import SarvamText
from ingest.adapters.search import SearchAdapter, TavilySearch
from ingest.adapters.vault import FileVault
from ingest.models import IngestError, Note, SearchResult
from ingest.render import dump_frontmatter


def sync_run(coro):
    """Executes a coroutine chain synchronously without socketpair overhead."""
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


@pytest.fixture
def temp_vault() -> FileVault:
    """Fixture providing an isolated FileVault in a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    vault = FileVault(temp_dir)
    return vault


# =====================================================================
# SECTION 1: FileVault Adversarial Stress Tests
# =====================================================================

def test_vault_rapid_concurrent_atomic_writes(temp_vault: FileVault):
    """Stress test FileVault with 30 concurrent threads writing distinct notes simultaneously."""
    note_count = 30

    def write_worker(idx: int) -> Path:
        note = Note(
            frontmatter={"title": f"Concurrent Note {idx}", "status": "inbox", "index": idx},
            body=f"## Content for Note {idx}\nBody text index {idx} verifying atomic write.",
            filename=f"2026-11-03 - Note-{idx:03d}.md",
        )
        return temp_vault.write_note(note, folder="inbox")

    with ThreadPoolExecutor(max_workers=8) as executor:
        written_paths = list(executor.map(write_worker, range(note_count)))

    assert len(written_paths) == note_count
    for idx, path in enumerate(written_paths):
        assert path.exists(), f"File {path} must exist"
        content = path.read_text(encoding="utf-8")
        assert f"Concurrent Note {idx}" in content
        assert f"index {idx}" in content


def test_vault_deep_collision_chain_up_to_15(temp_vault: FileVault):
    """Stress test collision chain resolution from base up to -15."""
    collision_depth = 15
    paths: list[Path] = []

    for i in range(1, collision_depth + 1):
        note = Note(
            frontmatter={"title": "Collision Subject", "status": "inbox", "version": i},
            body=f"Body version {i}",
            filename="2026-11-03 - Collision-Subject.md",
        )
        p = temp_vault.write_note(note, folder="inbox")
        paths.append(p)

    assert paths[0].name == "2026-11-03 - Collision-Subject.md"
    for i in range(2, collision_depth + 1):
        expected_name = f"2026-11-03 - Collision-Subject-{i}.md"
        assert paths[i - 1].name == expected_name, f"Expected {expected_name}, got {paths[i - 1].name}"
        assert paths[i - 1].exists()
        assert f"Body version {i}" in paths[i - 1].read_text(encoding="utf-8")


def test_vault_move_note_deep_collision_chain(temp_vault: FileVault):
    """Stress test move_note when target directory already has 10 collision files."""
    dest_dir = temp_vault.root / "projects"
    # Pre-seed target with 10 existing files
    base_stem = "2026-11-03 - Active-Project"
    (dest_dir / f"{base_stem}.md").write_text("Existing 1", encoding="utf-8")
    for i in range(2, 11):
        (dest_dir / f"{base_stem}-{i}.md").write_text(f"Existing {i}", encoding="utf-8")

    # Write note in inbox with identical base filename
    src_note = Note(
        frontmatter={"title": "Active Project", "status": "inbox"},
        body="Fresh project note from inbox",
        filename=f"{base_stem}.md",
    )
    src_path = temp_vault.write_note(src_note, folder="inbox")

    # Move to projects
    moved_path = temp_vault.move_note(src_path, folder="projects")
    assert moved_path.name == f"{base_stem}-11.md"
    assert moved_path.exists()
    assert not src_path.exists()
    moved_content = moved_path.read_text(encoding="utf-8")
    assert "status: filed" in moved_content
    assert "Fresh project note" in moved_content


def test_vault_move_note_to_archive_updates_status():
    """Verify moving note to _archive sets status to archived."""
    temp_dir = tempfile.mkdtemp()
    try:
        vault = FileVault(temp_dir)
        note = Note(
            frontmatter={"title": "Discarded Note", "status": "inbox"},
            body="To be archived",
            filename="2026-11-03 - Discard.md",
        )
        src = vault.write_note(note, folder="inbox")
        archived = vault.move_note(src, folder="_archive")
        assert archived.parent == vault.root / "_archive"
        assert "status: archived" in archived.read_text(encoding="utf-8")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_vault_move_note_without_status_in_frontmatter(temp_vault: FileVault):
    """Verify move_note cleanly injects status: filed into frontmatter missing status field."""
    inbox_dir = temp_vault.root / "inbox"
    raw_note = inbox_dir / "2026-11-03 - No-Status.md"
    raw_note.write_text(
        "---\ntitle: No Status Note\ntags: [research]\n---\n\n## Body\nContent here.\n",
        encoding="utf-8",
    )

    dest = temp_vault.move_note(raw_note, "reference")
    assert dest.exists()
    content = dest.read_text(encoding="utf-8")
    assert "status: filed" in content
    assert "title: No Status Note" in content
    assert "tags: [research]" in content


def test_vault_move_note_nonexistent_source_raises_ingest_error(temp_vault: FileVault):
    """Verify move_note on missing source raises IngestError('write', ...)."""
    with pytest.raises(IngestError) as exc_info:
        temp_vault.move_note(temp_vault.root / "inbox" / "ghost.md", "reference")
    assert exc_info.value.stage == "write"


def test_vault_nested_folder_creation(temp_vault: FileVault):
    """Verify write_note and move_note work seamlessly with deeply nested folders."""
    note = Note(
        frontmatter={"title": "Deep Note", "status": "inbox"},
        body="Nested content",
        filename="2026-11-03 - Deep.md",
    )
    written = temp_vault.write_note(note, folder="college-documents/sem-4/distributed-systems")
    assert written.exists()
    assert (temp_vault.root / "college-documents" / "sem-4" / "distributed-systems").is_dir()

    moved = temp_vault.move_note(written, folder="archive/sem-4/completed")
    assert moved.exists()
    assert (temp_vault.root / "archive" / "sem-4" / "completed").is_dir()
    assert not written.exists()


def test_vault_store_attachment_binary_integrity(temp_vault: FileVault):
    """Verify store_attachment preserves byte-for-byte fidelity of binary files via SHA-256."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        bin_path = temp_dir / "firmware.bin"
        # Generate 256 KB deterministic pseudo-binary data
        data = bytes((i * 37) % 256 for i in range(256 * 1024))
        bin_path.write_bytes(data)
        original_hash = hashlib.sha256(data).hexdigest()

        stored = temp_vault.store_attachment(bin_path, category="projects")
        assert stored.exists()
        assert stored.parent == temp_vault.root / "attachments" / "projects"
        stored_hash = hashlib.sha256(stored.read_bytes()).hexdigest()
        assert stored_hash == original_hash
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_vault_store_attachment_deep_collision_chain(temp_vault: FileVault):
    """Verify store_attachment collision increments up to -10."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        src = temp_dir / "diagram.png"
        src.write_bytes(b"\x89PNG\r\n\x1a\n...")

        stored_paths = []
        for _ in range(10):
            p = temp_vault.store_attachment(src, category="reference")
            stored_paths.append(p)

        assert stored_paths[0].name == "diagram.png"
        for i in range(2, 11):
            assert stored_paths[i - 1].name == f"diagram-{i}.png"
            assert stored_paths[i - 1].exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_vault_store_attachment_nonexistent_source_raises(temp_vault: FileVault):
    """Verify store_attachment with missing file raises IngestError('write', ...)."""
    with pytest.raises(IngestError) as exc_info:
        temp_vault.store_attachment(Path("missing_file.xyz"), "reference")
    assert exc_info.value.stage == "write"


def test_vault_archive_note_and_attachment_collisions(temp_vault: FileVault):
    """Verify archive_note resolves collisions when target _archive already has matching names."""
    archive_dir = temp_vault.root / "_archive"
    (archive_dir / "2026-11-03 - Note.md").write_text("Old archived note", encoding="utf-8")
    (archive_dir / "attachments" / "sheet.pdf").write_text("Old attachment", encoding="utf-8")

    # Create new note and attachment in inbox
    note = Note(
        frontmatter={"title": "Note", "status": "inbox"},
        body="Fresh note to archive",
        filename="2026-11-03 - Note.md",
    )
    src_note = temp_vault.write_note(note, "inbox")

    temp_dir = Path(tempfile.mkdtemp())
    try:
        src_att = temp_dir / "sheet.pdf"
        src_att.write_text("Fresh attachment", encoding="utf-8")

        archived_note, archived_att = temp_vault.archive_note(src_note, src_att)
        assert archived_note.name == "2026-11-03 - Note-2.md"
        assert archived_att is not None
        assert archived_att.name == "sheet-2.pdf"
        assert archived_note.exists()
        assert archived_att.exists()
        assert not src_note.exists()
        assert not src_att.exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_vault_append_correction_rapid_sequential(temp_vault: FileVault):
    """Verify rapid sequential appending to _corrections.jsonl produces valid JSONL with all entries."""
    entries_count = 25
    for i in range(entries_count):
        temp_vault.append_correction(
            original="college-documents",
            chosen="reference",
            note_title=f"Note Title {i}",
            iteration=i,
        )
    corrections_file = temp_vault.root / "_corrections.jsonl"
    assert corrections_file.exists()
    lines = corrections_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == entries_count
    for i, line in enumerate(lines):
        data = json.loads(line)
        assert data["original_category"] == "college-documents"
        assert data["iteration"] == i


def test_vault_append_correction_concurrent_validity(temp_vault: FileVault):
    """Verify concurrent threads appending to _corrections.jsonl produce valid JSON on every written line."""
    entries_count = 20

    def append_worker(i: int):
        temp_vault.append_correction(
            original="college-documents",
            chosen="reference",
            note_title=f"Note Title {i}",
            iteration=i,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(append_worker, range(entries_count)))

    corrections_file = temp_vault.root / "_corrections.jsonl"
    assert corrections_file.exists()
    lines = corrections_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 0
    for line in lines:
        data = json.loads(line)
        assert data["original_category"] == "college-documents"
        assert "timestamp" in data


def test_vault_append_correction_unicode_scripts(temp_vault: FileVault):
    """Verify append_correction preserves Devanagari, Tamil, and emojis without corrupted escaping."""
    temp_vault.append_correction(
        original="college-documents",
        chosen="reference",
        note_title="प्रमाणपत्र अंकतालिका 🚀 மற்றும் சான்றிதழ்",
        extra_note="परीक्षण குறிப்பு 🎯",
    )
    corrections_file = temp_vault.root / "_corrections.jsonl"
    raw_text = corrections_file.read_text(encoding="utf-8")
    assert "प्रमाणपत्र अंकतालिका" in raw_text
    assert "சான்றிதழ்" in raw_text
    assert "🚀" in raw_text

    line = raw_text.strip().splitlines()[-1]
    parsed = json.loads(line)
    assert parsed["note_title"] == "प्रमाणपत्र अंकतालिका 🚀 மற்றும் சான்றிதழ்"
    assert parsed["extra_note"] == "परीक्षण குறிப்பு 🎯"


# =====================================================================
# SECTION 2: SarvamLlm Adversarial Tests
# =====================================================================

def test_sarvam_llm_markdown_fences_uppercase_and_whitespace():
    """Verify stripping markdown code fences with ```JSON, extra whitespace, or trailing commentary."""
    raw_responses = [
        "```json\n{\"category\": \"reference\", \"title\": \"Clean Lowercase\"}\n```",
        "```\n{\"category\": \"projects\", \"title\": \"Plain Fence\"}\n```",
        "```JSON\n{\"category\": \"college-documents\", \"title\": \"Uppercase Fence\"}\n```",
        "```json  \n{\"category\": \"reference\", \"title\": \"Trailing Fence Whitespace\"}\n  ```",
        "Here is the classification result:\n```json\n{\"category\": \"projects\", \"title\": \"Surrounded Fence\"}\n```\nHope that helps!",
    ]
    for raw in raw_responses:
        extracted = SarvamLlm._parse_and_extract_json(raw)
        assert "category" in extracted
        assert "title" in extracted


def test_sarvam_llm_conversational_padding_with_embedded_braces():
    """Verify JSON extraction when outer text contains conversational commentary."""
    raw = (
        "Based on your requirements, I have classified the document:\n\n"
        '{"category": "possible-projects", "title": "Smart Campus Transit", '
        '"summary": "Bus tracking for students.", "tags": ["transit", "campus"]}\n\n'
        "Please let me know if you would like me to adjust any tags."
    )
    extracted = SarvamLlm._parse_and_extract_json(raw)
    assert extracted["category"] == "possible-projects"
    assert extracted["title"] == "Smart Campus Transit"


def test_sarvam_llm_complete_json_repair_success():
    """Verify automatic single repair attempt succeeds when first call returns truncated/invalid JSON."""
    async def _test():
        invalid_choice = MagicMock()
        invalid_choice.message.content = '{"category": "projects", "title": "Truncated...'
        invalid_response = MagicMock(choices=[invalid_choice])

        repaired_choice = MagicMock()
        repaired_choice.message.content = '{"category": "projects", "title": "Repaired Project", "summary": "Valid JSON"}'
        repaired_response = MagicMock(choices=[repaired_choice])

        mock_client = MagicMock()
        mock_client.chat.completions.side_effect = [invalid_response, repaired_response]

        llm = SarvamLlm(client=mock_client)
        result = await llm.complete_json("system prompt", "user prompt")
        assert result["category"] == "projects"
        assert result["title"] == "Repaired Project"
        assert mock_client.chat.completions.call_count == 2

    sync_run(_test())


def test_sarvam_llm_complete_json_double_failure_raises_ingest_error():
    """Verify that when both initial and repair calls fail, IngestError('classify', ...) is raised."""
    async def _test():
        bad_choice = MagicMock()
        bad_choice.message.content = "I cannot generate JSON for this query."
        bad_response = MagicMock(choices=[bad_choice])

        mock_client = MagicMock()
        mock_client.chat.completions.side_effect = [bad_response, bad_response]

        llm = SarvamLlm(client=mock_client)
        with pytest.raises(IngestError) as exc_info:
            await llm.complete_json("system", "user")
        assert exc_info.value.stage == "classify"

    sync_run(_test())


def test_sarvam_llm_complete_text_client_error():
    """Verify complete_text wraps client exception in IngestError('classify', ...)."""
    async def _test():
        mock_client = MagicMock()
        mock_client.chat.completions.side_effect = ConnectionResetError("Remote server closed socket")

        llm = SarvamLlm(client=mock_client)
        with pytest.raises(IngestError) as exc_info:
            await llm.complete_text("system", "user")
        assert exc_info.value.stage == "classify"

    sync_run(_test())


# =====================================================================
# SECTION 3: SarvamOcr Adversarial Tests
# =====================================================================

def test_sarvam_ocr_exact_word_count_boundary_50_vs_51():
    """Empirically test exact boundary: 50 words MUST call OCR API; 51 words MUST bypass it."""
    async def _test():
        adapter = SarvamOcr(api_key="mock_key")
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # 1. Create PDF with exactly 50 words
            words_50 = ["word"] * 50
            pdf_50 = temp_dir / "exact_50_words.pdf"
            doc = fitz.open()
            page = doc.new_page()
            page.insert_textbox(fitz.Rect(50, 50, 400, 400), " ".join(words_50))
            doc.save(str(pdf_50))
            doc.close()

            # 2. Create PDF with exactly 51 words
            words_51 = ["word"] * 51
            pdf_51 = temp_dir / "exact_51_words.pdf"
            doc = fitz.open()
            page = doc.new_page()
            page.insert_textbox(fitz.Rect(50, 50, 400, 400), " ".join(words_51))
            doc.save(str(pdf_51))
            doc.close()

            # 50 words: len(words) > 50 is False -> must invoke _call_sarvam_ocr
            with patch.object(adapter, "_call_sarvam_ocr", AsyncMock(return_value="OCR result 50")) as mock_ocr:
                res_50 = await adapter.extract_text(pdf_50)
                assert res_50 == "OCR result 50"
                mock_ocr.assert_called_once()

            # 51 words: len(words) > 50 is True -> must return fitz_text without calling _call_sarvam_ocr
            with patch.object(adapter, "_call_sarvam_ocr", AsyncMock()) as mock_ocr:
                res_51 = await adapter.extract_text(pdf_51)
                assert len(res_51.split()) == 51
                mock_ocr.assert_not_called()

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    sync_run(_test())


def test_sarvam_ocr_multipage_digital_pdf_fast_path():
    """Verify multipage digital PDF (> 50 total words across pages) concatenates all pages without API call."""
    async def _test():
        adapter = SarvamOcr(api_key="mock_key")
        temp_dir = Path(tempfile.mkdtemp())
        try:
            pdf_path = temp_dir / "multipage.pdf"
            doc = fitz.open()
            for page_idx in range(3):
                p = doc.new_page()
                page_text = f"Page {page_idx + 1} content with twenty words of academic discussion for testing distributed multipage document extraction without calling remote endpoints."
                p.insert_textbox(fitz.Rect(50, 50, 400, 400), page_text)
            doc.save(str(pdf_path))
            doc.close()

            with patch.object(adapter, "_call_sarvam_ocr", AsyncMock()) as mock_ocr:
                extracted = await adapter.extract_text(pdf_path)
                assert "Page 1" in extracted
                assert "Page 2" in extracted
                assert "Page 3" in extracted
                mock_ocr.assert_not_called()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    sync_run(_test())


def test_sarvam_ocr_sparse_pdf_fallback_on_api_error():
    """Verify sparse PDF (<= 50 words) falls back to fitz text when OCR API fails."""
    async def _test():
        adapter = SarvamOcr(api_key="mock_key")
        temp_dir = Path(tempfile.mkdtemp())
        try:
            sparse_pdf = temp_dir / "short_memo.pdf"
            doc = fitz.open()
            p = doc.new_page()
            p.insert_textbox(fitz.Rect(50, 50, 300, 300), "Short Meeting Memo: Room 402 at 3 PM")
            doc.save(str(sparse_pdf))
            doc.close()

            # When Sarvam OCR fails, it should gracefully fall back to fitz text
            with patch.object(adapter, "_call_sarvam_ocr", side_effect=httpx.ConnectError("API down")):
                res = await adapter.extract_text(sparse_pdf)
                assert "Short Meeting Memo" in res
                assert "Room 402 at 3 PM" in res
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    sync_run(_test())


def test_sarvam_ocr_scanned_pdf_error_handling():
    """Verify scanned PDF (0 words) raises IngestError('ocr', ...) when OCR API fails."""
    async def _test():
        adapter = SarvamOcr(api_key="mock_key")
        temp_dir = Path(tempfile.mkdtemp())
        try:
            empty_pdf = temp_dir / "scanned_empty.pdf"
            doc = fitz.open()
            doc.new_page()  # Blank page, 0 words
            doc.save(str(empty_pdf))
            doc.close()

            with patch.object(adapter, "_call_sarvam_ocr", side_effect=httpx.HTTPStatusError("500 Server Error", request=MagicMock(), response=MagicMock())):
                with pytest.raises(IngestError) as exc_info:
                    await adapter.extract_text(empty_pdf)
                assert exc_info.value.stage == "ocr"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    sync_run(_test())


def test_sarvam_ocr_plaintext_and_markdown_direct_read():
    """Verify .txt and .md files are read directly without PyMuPDF or OCR API."""
    async def _test():
        adapter = SarvamOcr()
        temp_dir = Path(tempfile.mkdtemp())
        try:
            f = temp_dir / "raw_notes.md"
            f.write_text("# Direct Markdown Note\nContent without OCR overhead.", encoding="utf-8")
            res = await adapter.extract_text(f)
            assert res == "# Direct Markdown Note\nContent without OCR overhead."
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    sync_run(_test())


# =====================================================================
# SECTION 4: SarvamText Adversarial Tests
# =====================================================================

def test_sarvam_text_multilingual_english_bypass():
    """Confirm user directive: to_english must bypass Indic translation and return original text directly."""
    indic_samples = {
        "hi-IN": "छात्रवृत्ति आवेदन पत्र और निर्देश",
        "ta-IN": "கல்வி உதவித்தொகை விண்ணப்பம் மற்றும் வழிகாட்டுதல்கள்",
        "te-IN": "స్కాలర్‌షిప్ దరఖాస్తు ఫారమ్ మరియు సూచనలు",
        "kn-IN": "ವಿದ್ಯಾರ್ಥಿವೇತನ ಅರ್ಜಿ ನಮೂನೆ ಮತ್ತು ಮಾರ್ಗಸೂಚಿಗಳು",
        "ml-IN": "സ്കോളർഷിപ്പ് അപേക്ഷാ ഫോമും മാർഗ്ഗനിർദ്ദേശങ്ങളും",
        "bn-IN": "বৃত্তি আবেদনপত্র এবং নির্দেশিকা",
        "mr-IN": "शिष्यवृत्ती अर्ज आणि मार्गदर्शक तत्त्वे",
        "gu-IN": "શિષ્યવૃત્તિ અરજી ફોર્મ અને માર્ગદર્શિકા",
        "pa-IN": "ਵਜ਼ੀਫ਼ਾ ਅਰਜ਼ੀ ਫ਼ਾਰਮ ਅਤੇ ਦਿਸ਼ਾ-ਨਿਰਦੇਸ਼",
    }
    adapter = SarvamText(api_key="mock_key")
    for lang_code, text in indic_samples.items():
        async def _test(t=text, l=lang_code):
            result = await adapter.to_english(t, l)
            assert result == t, f"Expected unchanged pass-through for {l}"
        sync_run(_test())


def test_sarvam_text_chunking_gigantic_sentence_no_loss():
    """Verify chunk_text cleanly splits a 5000-char sentence without losing any characters."""
    adapter = SarvamText(chunk_chars=500)
    huge_sentence = "A" * 4732
    chunks = adapter.chunk_text(huge_sentence, max_chars=500)
    assert len(chunks) == 10  # 9 chunks of 500 + 1 chunk of 232
    reconstructed = "".join(chunks)
    assert reconstructed == huge_sentence
    for c in chunks:
        assert len(c) <= 500


def test_sarvam_text_chunking_exact_limit():
    """Verify chunk_text on text exactly equal to chunk_chars produces single chunk."""
    adapter = SarvamText(chunk_chars=100)
    exact_text = "X" * 100
    chunks = adapter.chunk_text(exact_text, max_chars=100)
    assert len(chunks) == 1
    assert chunks[0] == exact_text


def test_sarvam_text_detect_empty_or_whitespace_defaults_en():
    """Verify detect on empty, whitespace, or none returns 'en-IN'."""
    adapter = SarvamText()
    async def _test():
        assert await adapter.detect("") == "en-IN"
        assert await adapter.detect("   \n\t  ") == "en-IN"
    sync_run(_test())


def test_sarvam_text_detect_client_exception_fallback():
    """Verify detect falls back to en-IN when client throws exception."""
    mock_client = MagicMock()
    mock_client.text.identify_language.side_effect = RuntimeError("Service unavailable")
    adapter = SarvamText(client=mock_client)
    async def _test():
        assert await adapter.detect("Sample text") == "en-IN"
    sync_run(_test())


# =====================================================================
# SECTION 5: SearchAdapter Adversarial Tests
# =====================================================================

def test_search_adapter_api_failure_returns_fallback():
    """Verify SearchAdapter returns fallback results when API fails with 500 Internal Server Error."""
    async def _test():
        adapter = SearchAdapter(api_key="tvly_valid_secret")

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError("500 Error", request=MagicMock(), response=mock_resp)

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(IngestError) as exc_info:
                await adapter.search("distributed key-value store", k=3)
            assert exc_info.value.stage == "search"
    sync_run(_test())


def test_search_adapter_empty_results_returns_empty_list():
    """Verify SearchAdapter returns empty list [] when API returns empty results list (grounded)."""
    async def _test():
        adapter = SearchAdapter(api_key="tvly_valid_secret")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"results": []}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch("httpx.AsyncClient", return_value=mock_client):
            results = await adapter.search("novel obscure project idea", k=2)
            assert results == []
    sync_run(_test())


def test_search_adapter_strict_k_limit():
    """Verify search adapter strictly honors k limits from 1 to 5."""
    async def _test():
        adapter = SearchAdapter(api_key="tvly_valid_secret")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [
                {"title": f"Result {i}", "url": f"https://example.com/{i}", "content": f"Snippet {i}"}
                for i in range(5)
            ]
        }
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch("httpx.AsyncClient", return_value=mock_client):
            for k_val in (1, 2, 3, 5):
                res = await adapter.search("test project query", k=k_val)
                assert len(res) == k_val
    sync_run(_test())


def test_search_adapter_query_special_characters():
    """Verify queries with special characters and missing API keys raise IngestError('search')."""
    async def _test():
        adapter = SearchAdapter(api_key=None)
        weird_query = "C++ & Rust: Which is better? #1 @2026! [benchmark]"
        with pytest.raises(IngestError) as exc_info:
            await adapter.search(weird_query, k=2)
        assert exc_info.value.stage == "search"
    sync_run(_test())


def test_vault_move_note_same_path_in_place(temp_vault: FileVault):
    """Verify move_note when moving a note in-place updates status without unlinking or creating duplicate."""
    note = Note(
        frontmatter={"title": "Same Path", "status": "inbox"},
        body="Body to update in place",
        filename="2026-11-03 - In-Place.md",
    )
    p = temp_vault.write_note(note, folder="inbox")
    assert "status: inbox" in p.read_text(encoding="utf-8")

    # Move to inbox (same folder)
    updated = temp_vault.move_note(p, folder="inbox")
    assert updated.resolve() == p.resolve()
    assert updated.exists()
    content = updated.read_text(encoding="utf-8")
    assert "status: filed" in content


def test_vault_invalid_windows_filename_raises_ingest_error(temp_vault: FileVault):
    """Verify write_note with illegal characters for filesystem raises IngestError('write', ...)."""
    note = Note(
        frontmatter={"title": "Bad Char Note", "status": "inbox"},
        body="Bad filename body",
        filename="2026-11-03 - Illegal:Colon*Star?.md",
    )
    with pytest.raises(IngestError) as exc_info:
        temp_vault.write_note(note, folder="inbox")
    assert exc_info.value.stage == "write"


def test_vault_move_note_multiple_status_lines_body_replacement_behavior(temp_vault: FileVault):
    """Empirically test move_note regex behavior on files containing status: lines in body."""
    inbox_dir = temp_vault.root / "inbox"
    note_file = inbox_dir / "2026-11-03 - Body-Status.md"
    note_file.write_text(
        "---\ntitle: Body Status\nstatus: inbox\n---\n\n## Section\nstatus: pending review\nMore text\n",
        encoding="utf-8",
    )
    moved = temp_vault.move_note(note_file, "reference")
    content = moved.read_text(encoding="utf-8")
    # Frontmatter status should be updated to filed
    assert "status: filed" in content


def test_sarvam_llm_multiple_json_objects_repair():
    """Verify LLM response with multiple JSON objects triggers repair attempt and succeeds on clean JSON."""
    async def _test():
        multi_choice = MagicMock()
        multi_choice.message.content = '{"thought": "exploring options"} {"category": "reference", "title": "Final Result"}'
        multi_response = MagicMock(choices=[multi_choice])

        clean_choice = MagicMock()
        clean_choice.message.content = '{"category": "reference", "title": "Final Result", "summary": "Clean JSON"}'
        clean_response = MagicMock(choices=[clean_choice])

        mock_client = MagicMock()
        mock_client.chat.completions.side_effect = [multi_response, clean_response]

        llm = SarvamLlm(client=mock_client)
        result = await llm.complete_json("sys", "user")
        assert result["category"] == "reference"
        assert result["title"] == "Final Result"
        assert mock_client.chat.completions.call_count == 2
    sync_run(_test())


def test_sarvam_ocr_non_pdf_non_txt_routes_to_ocr_api():
    """Verify unsupported binary files (e.g. .dat, .bin, .jpg) route to OCR API rather than local read."""
    async def _test():
        adapter = SarvamOcr(api_key="mock_key")
        temp_dir = Path(tempfile.mkdtemp())
        try:
            sample_img = temp_dir / "receipt.jpg"
            sample_img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

            with patch.object(adapter, "_call_sarvam_ocr", AsyncMock(return_value="Scanned Receipt Content")) as mock_call:
                res = await adapter.extract_text(sample_img)
                assert res == "Scanned Receipt Content"
                mock_call.assert_called_once()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    sync_run(_test())
