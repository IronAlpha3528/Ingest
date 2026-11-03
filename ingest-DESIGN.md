# Design Document

**Project:** `ingest` — Telegram → Obsidian capture pipeline
**Document type:** Technical design
**Version:** 1.0
**Date:** 3 November 2026
**Companion document:** `ingest-SRS.md`
**Audience:** the developer, and any AI coding assistant working on this repository

---

## 1. Design goals

The prototype must be **correct, small, and easy for a language model to extend one module at a time.** Every decision below serves those three goals.

| Goal | Consequence in the design |
|---|---|
| Each module is generatable in isolation | Modules communicate only through typed models defined in one file |
| No hidden state | Dependencies are passed in explicitly; nothing is imported for side effects |
| Behaviour is testable without network | Every external service sits behind a `Protocol`; fakes exist from day one |
| Prompts are reviewable artefacts | Prompt text lives in `prompts.py` as constants, never inline in handlers |
| Failures are attributable | Each stage raises a tagged error; handlers report the stage |
| No premature abstraction | Six modules, one interface per external service, nothing generic |

### Principles applied

**Functional core, imperative shell.** All logic that can be pure — prompt construction, note rendering, filename generation, response parsing — is a pure function with no I/O. The shell (bot handlers, adapters) does I/O and holds no logic. This is the single most important structural decision: it makes the interesting parts testable in milliseconds.

**Ports and adapters.** Sarvam, the search API, the Telegram API and the filesystem are each reached through a narrow `Protocol`. Swapping Sarvam Vision for a PyMuPDF fallback, or Tavily for Brave, is a one-file change.

**Parse, don't validate.** Untrusted data — LLM output, filenames, API responses — is converted into a typed model at the boundary and never travels as a raw `dict` afterwards.

**Explicit dependencies.** No module-level clients, no singletons, no `global`. A `Deps` dataclass is constructed once in `main.py` and passed down.

**Walking skeleton first.** Build a vertical slice that produces a note before any API exists, then add real adapters behind the same interfaces.

**YAGNI.** No repository pattern, no service layer, no event bus, no async concurrency. Sequential `await` is correct at this scale.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  bot.py — imperative shell                                   │
│  Telegram handlers → build Capture → call pipeline → render  │
│  keyboard → answer callback queries                          │
└───────────────┬─────────────────────────────────────────────┘
                │  Capture
┌───────────────▼─────────────────────────────────────────────┐
│  pipeline.py — orchestration (async, no protocol details)    │
│  normalise() → classify() → build_note() → expand() → file() │
└───────┬───────────────────────────────────────────────┬──────┘
        │                                               │
┌───────▼──────────────┐                     ┌──────────▼───────┐
│  prompts.py          │                     │  render.py        │
│  pure: str → str     │                     │  pure: Note       │
└───────┬──────────────┘                     └──────────┬───────┘
        │                                               │
┌───────▼───────────────────────────────────────────────▼──────┐
│  adapters/ — the only place I/O happens                      │
│  sarvam_llm · sarvam_ocr · sarvam_text · search · vault      │
└──────────────────────────────────────────────────────────────┘
```

Rules that hold across the whole codebase:

1. `prompts.py` and `render.py` import nothing but `models.py` and the standard library. They perform no I/O.
2. `adapters/*` are the only files that import `sarvamai`, `httpx`, `aiohttp` or `pathlib`-write operations.
3. `pipeline.py` may import adapters for type hints only; it receives instances via `Deps`.
4. `bot.py` knows about Telegram and nothing about Sarvam.
5. `models.py` imports nothing from the project.

### 2.1 Sarvam component map

Every model call in this project goes to Sarvam. One key, one SDK (`sarvamai`), one credit balance; the only third-party service is web search.

| Stage | Component | Endpoint | Why this one |
|---|---|---|---|
| OCR | **Sarvam Vision** (Document Intelligence) | Document intelligence API | 23 languages with structured output; reads Devanagari and Tamil scans that English-first OCR mangles |
| Language check | **Language Identification** | `POST /text-lid` | One cheap call decides whether translation is needed at all |
| Normalisation | **Sarvam-Translate** `sarvam-translate:v1` | `POST /translate` | All 22 scheduled languages, formal register, bidirectional |
| Colloquial fallback | **Mayura** `mayura:v1` | `POST /translate` | Handles code-mixed chat-style text; offers `output_script` and `numerals_format` |
| Classification and field extraction | **Sarvam-105B** `sarvam-105b` | `POST /v2/chat/completions` | Flagship Indic reasoning, 128K context, JSON output mode |
| Brief writing | **Sarvam-105B** | same | One model for both jobs, so prompt behaviour stays consistent |
| Voice notes (extension) | **Saaras v4** | `POST /speech-to-text` | Accepts OGG/Opus directly, no transcoding |
| Spoken summary (extension) | **Bulbul v4 Flash** | `POST /text-to-speech` | 200+ persona voices |

Three consequences for the design:

1. **`sarvam_text.py` is a real module, not a nicety.** Detection and translation run before classification, so the classifier reasons over English while field values keep their original script. Without it, a Hindi marksheet produces either an English note that has lost the issuer's name, or a Hindi note the user cannot scan quickly.
2. **The model is configuration, not code.** `/v2` is OpenAI-compatible, so `model` is a `Settings` field. Moving the brief-writing call to `glm5.3` is a one-line change, and `GET /v2/models` lists what the key can actually reach.
3. **Chunking belongs to the adapter.** `/translate` accepts 2,000 characters per request on `sarvam-translate:v1` (1,000 on `mayura:v1`), so long documents are split at sentence boundaries inside `sarvam_text.py`. Nothing above the adapter knows the limit exists.

---

## 3. Module map

| File | Responsibility | Must not |
|---|---|---|
| `models.py` | All data contracts and enums | Import anything from the project |
| `config.py` | Environment loading, path validation | Contain logic |
| `prompts.py` | Prompt constants and pure prompt builders | Perform I/O |
| `render.py` | `Classification` + `Capture` → `Note`; filename generation | Call an API or touch disk |
| `adapters/sarvam_llm.py` | Chat completion; JSON mode; schema validation | Contain prompts |
| `adapters/sarvam_ocr.py` | Sarvam Vision document intelligence; PyMuPDF fallback | Contain prompts |
| `adapters/sarvam_text.py` | Language identification, translation, chunking | Contain prompts |
| `adapters/search.py` | Web search provider | Contain prompts |
| `adapters/vault.py` | All filesystem writes, moves, folder creation | Contain logic |
| `pipeline.py` | The stages, error tagging | Know Telegram or Sarvam types |
| `bot.py` | Handlers, keyboards, message edits | Contain business logic |
| `main.py` | Assemble `Deps`, start polling | — |

---

## 4. Data contracts

`models.py` — the only shared vocabulary. Everything else is local.

```python
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path

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

    @field_validator("title")
    @classmethod
    def title_is_single_line(cls, v: str) -> str:
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
    frontmatter: dict
    body: str
    filename: str


class Proposal(BaseModel):
    """What bot.py needs to render a message and its buttons."""
    note: Note
    category: Category
    preview: str
```

Design notes:

- `Classification` is both the parsing target and the description of what the LLM must return. One definition, no drift.
- `Proposal` exists so `pipeline.py` never returns a Telegram object.
- `Capture` is normalised at the boundary — `pipeline.py` never sees a `telegram.Message`.

---

## 5. Interfaces

```python
# adapters/ports.py  (Protocols only, no implementations)
from typing import Protocol

class LlmPort(Protocol):
    async def complete_json(self, system: str, user: str) -> dict: ...
    async def complete_text(self, system: str, user: str) -> str: ...

class OcrPort(Protocol):
    async def extract_text(self, path: Path) -> str: ...

class TextPort(Protocol):
    """Sarvam language identification and translation."""
    async def detect(self, text: str) -> str:                  # BCP-47, e.g. "hi-IN"
    async def to_english(self, text: str, source: str) -> str: ...

class SearchPort(Protocol):
    async def search(self, query: str, k: int = 5) -> list[SearchResult]: ...

class VaultPort(Protocol):
    def write_note(self, note: Note, folder: str) -> Path: ...
    def move_note(self, src: Path, folder: str) -> Path: ...
    def store_attachment(self, src: Path, category: str) -> Path: ...
```

```python
# config.py
class Settings(BaseSettings):
    telegram_bot_token: str
    sarvam_api_key: str
    search_api_key: str
    vault_path: Path

    # Model choice is configuration, not code — /v2 is OpenAI-compatible.
    model: str = "sarvam-105b"
    translate_model: str = "sarvam-translate:v1"   # or "mayura:v1" for colloquial input
    ocr_model: str = "sarvam-vision"

    max_download_mb: int = 20
    translate_chunk_chars: int = 2000

    model_config = SettingsConfigDict(env_file=".env")
```

```python
# pipeline.py — the public surface of the domain
@dataclass(frozen=True)
class Deps:
    llm: LlmPort
    ocr: OcrPort
    text: TextPort
    search: SearchPort
    vault: VaultPort

async def normalise(capture: Capture, deps: Deps) -> tuple[str, str | None]
    # -> (original_text, english_translation or None)
async def classify(capture: Capture, original: str, translation: str | None,
                   deps: Deps) -> Classification
async def build_note(capture: Capture, cls: Classification, deps: Deps) -> Note
async def expand_project(idea: str, deps: Deps) -> str
async def ingest(capture: Capture, deps: Deps) -> Proposal
def file_proposal(proposal: Proposal, category: Category, deps: Deps) -> Path
```

```python
# render.py — pure, no I/O, fully unit-testable
def slug(title: str) -> str
def filename_for(title: str, day: date) -> str
def build_frontmatter(capture: Capture, cls: Classification, attachment: str | None) -> dict
def render_body(capture: Capture, cls: Classification, extra: str | None = None) -> str
def build_note(capture: Capture, cls: Classification,
               attachment: str | None, body_extra: str | None) -> Note
```

```python
# prompts.py — pure
def classify_prompt(capture: Capture, original: str,
                    translation: str | None) -> tuple[str, str]
def project_prompt(idea: str, results: list[SearchResult]) -> tuple[str, str]
def extract_prompt(original: str, translation: str | None) -> tuple[str, str]
```

Every builder returns `(system, user)`. That tuple is the unit of review — you can print it, diff it, and test it without a network.

---

## 6. Prompt specification

Prompts are the highest-leverage code in this project and the most likely to be regenerated. They live as module constants with a version comment, and their **only** job is to make `Classification` come back valid.

### 6.1 Classifier system prompt

```python
CLASSIFY_SYSTEM = """\
You file content into a personal knowledge base. You return one JSON object and nothing else.

Categories, exactly one:
- "college-documents": academic or institutional paperwork. Marksheets, transcripts,
  certificates, fee receipts, admit cards, ID cards, admission or bonafide letters.
- "possible-projects": an idea for something the user could build. Often one line.
  Software, hardware, a product, a service, a research project.
- "reference": anything else. Articles, notes, screenshots, quotes, general reading.

Rules:
- Return only JSON matching the schema. No prose, no code fences, no explanation.
- "title": 3 to 8 words, title case, no dates, no file extensions, no punctuation
  except hyphens. It must read as a note name, not a filename.
- "summary": 2 to 4 sentences describing what this is and why it matters.
  Write in English, always.
- If the input contains both an ORIGINAL block and an ENGLISH TRANSLATION block,
  summarise from the translation, but take every field value from the ORIGINAL,
  copied character for character in its own script.
- "tags": 2 to 5 lowercase hyphenated tags. Reuse plain words a person would search.
- For "college-documents" also fill "document_type", "issuer", "issued_date"
  (YYYY-MM-DD if visible, else null) and "reference_ids" (roll numbers, certificate
  numbers, application IDs). Copy these values verbatim in their original script;
  do not translate them.
- For other categories set those four fields to null or [].
- If the content is too short or ambiguous to classify, choose "reference".
- Never invent a value. A null is always better than a guess.
"""
```

The JSON schema is appended by `classify_prompt`, generated from `Classification.model_json_schema()`, so the prompt and the parser can never disagree.

### 6.2 Project expansion prompt

```python
PROJECT_SYSTEM = """\
You expand a one-line project idea into a short brief for the person who had it.

Write Markdown with exactly these sections, in this order:
## Description
## Similar projects and competitors
## Research areas
## Anticipated problems
## Open questions

Rules:
- Description: 2 short paragraphs. Concrete. Say what it does and who it is for.
- Similar projects and competitors: 3 to 5 items, each a bullet in the form
  "- **Name** — one line on what it does and how this idea differs. <URL>"
  Use only URLs from the search results provided. If the results support fewer than
  three, write fewer. Never invent a product, a company, or a URL.
- Research areas: 4 to 6 bullets, each a specific thing to learn or investigate.
- Anticipated problems: 3 to 5 bullets in the form "- **Problem** — mitigation."
- Open questions: 3 to 4 bullets, questions the idea leaves undecided.
- No preamble, no closing summary, no restating the idea.
"""
```

`project_prompt` formats the search results into the user message as `title — snippet — url` blocks. The "use only these URLs" constraint is what keeps the competitor section honest; it is the difference between a grounded brief and a plausible fiction.

### 6.3 Prompt hygiene

- One prompt, one job. Classification and expansion are separate calls.
- Never ask for prose and JSON in the same call.
- Keep the schema in the prompt generated, not hand-written.
- Version prompts by comment (`# v3 — tightened title rule`), and when a change alters output noticeably, note it in the commit message.

---

## 7. Note rendering

`render.py` holds three body templates as constants. They are the visible product; treat them as design artefacts, not string concatenation.

```python
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

{extra}"""

PROJECT_BODY = """\
## One-liner
> {idea}

{extra}"""

REFERENCE_BODY = """\
## Summary
{summary}

## Key points
{points}

{extra}"""
```

Frontmatter is built as an ordered `dict` — Obsidian preserves insertion order in the properties panel, and a stable order makes diffs readable:

```python
def build_frontmatter(capture, cls, attachment):
    fm = {
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
    if cls.category is Category.COLLEGE:
        fm |= {
            "document_type": cls.document_type,
            "issuer": cls.issuer,
            "issued_date": cls.issued_date,
            "reference_ids": cls.reference_ids,
        }
    return fm
```

`usage` in the college template ("what this is for") is a fixed lookup by `document_type`, not an LLM call — a dict of twelve entries covers marksheets, certificates, receipts and the rest, and it is deterministic and free.

Filename generation:

```python
def slug(title: str) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", title).strip()
    return "-".join(cleaned.split())[:60] or "untitled"

def filename_for(title: str, day: date) -> str:
    return f"{day.isoformat()} - {slug(title)}.md"
```

Collision handling belongs to `VaultPort.write_note`, which appends `-2`, `-3` as needed.

---

## 8. Flows

### 8.1 File capture

```
User        bot.py          pipeline.py        adapters            vault
 │  PDF       │                  │                 │                 │
 ├───────────►│ download         │                 │                 │
 │            ├──────────────────┼────────────────►│                 │
 │  "reading" ◄──────────────────┤                 │                 │
 │            │ Capture          │                 │                 │
 │            ├─────────────────►│ ocr.extract     │                 │
 │            │                  ├────────────────►│                 │
 │            │                  │ text.detect     │                 │
 │            │                  ├────────────────►│                 │
 │            │                  │ text.to_english │ (if not en)     │
 │            │                  ├────────────────►│                 │
 │            │                  │ classify        │                 │
 │            │                  ├────────────────►│                 │
 │            │                  │ render (pure)   │                 │
 │            │                  │ write to inbox  │                 │
 │            │                  ├─────────────────┼────────────────►│
 │  proposal  ◄──────────────────┤                 │                 │
 │            │                  │                 │                 │
 │  Approve   │                  │                 │                 │
 ├───────────►│ file_proposal    │                 │                 │
 │            ├──────────────────┼─────────────────┼────────────────►│
 │  "filed"   ◄──────────────────┤                 │                 │
```

### 8.2 Project idea

Text capture skips OCR. `ingest()` detects `Category.PROJECT`, calls `search.search(idea, k=5)`, then `expand_project()` with those results, and injects the returned Markdown as `extra` into `PROJECT_BODY`.

### 8.3 Callback handling

Button payloads are `f"{action}:{note_path}"` — for example `approve:inbox/2026-11-03 - Mess-Bills.md`. The handler parses the action, resolves the path, and calls the matching vault operation. **Never** store note content in callback data: it is capped at 64 bytes and it makes state invisible.

---

## 9. Error policy

```python
class IngestError(Exception):
    def __init__(self, stage: str, cause: Exception):
        self.stage = stage
        self.cause = cause
        super().__init__(f"{stage}: {cause}")
```

Rules:

- Every adapter wraps its own failures in `IngestError(stage, e)`. `stage` is one of `download`, `ocr`, `classify`, `search`, `render`, `write`.
- `pipeline.py` raises; it never catches. `bot.py` catches once, at the handler boundary, and replies `"Couldn't process that — {stage} failed. Send it again."`
- **No partial notes.** The note is written only after classification succeeds, and `write_note` writes to a temporary file then `os.replace`s it. A crash leaves either the old state or the new one.
- No bare `except:`. Catching `Exception` is permitted only at the handler boundary, and the message must name the stage.
- LLM JSON that fails validation gets exactly one repair attempt with the validation error appended to the user message. A second failure raises `IngestError("classify", ...)`.

---

## 10. Repository layout

```
ingest/
├── .env.example
├── .gitignore                 # .env, __pycache__, *.pdf
├── pyproject.toml
├── README.md
├── src/ingest/
│   ├── __init__.py
│   ├── config.py
│   ├── models.py
│   ├── prompts.py
│   ├── render.py
│   ├── pipeline.py
│   ├── bot.py
│   ├── main.py
│   └── adapters/
│       ├── ports.py
│       ├── sarvam_llm.py
│       ├── sarvam_ocr.py
│       ├── search.py
│       └── vault.py
└── tests/
    ├── fakes.py               # FakeLlm, FakeOcr, FakeSearch, TempVault
    ├── test_render.py
    ├── test_prompts.py
    ├── test_pipeline.py
    └── golden/
        ├── college.md
        ├── project.md
        └── reference.md
```

```toml
# pyproject.toml
[project]
name = "ingest"
requires-python = ">=3.12"
dependencies = [
  "python-telegram-bot[job-queue]>=21",
  "sarvamai",
  "pydantic>=2",
  "pydantic-settings",
  "python-frontmatter",
  "pymupdf",
  "httpx",
]
```

---

## 11. Build sequence

Each step is a vertical slice with a check. Do not start the next until the check passes — this is what keeps AI-assisted development from drifting.

| Step | Build | Check |
|---|---|---|
| **S0** | `models.py`, `config.py`, `.env.example` | `python -c "from ingest.models import Classification; print(Classification.model_json_schema())"` prints a schema |
| **S1** | `render.py` + `tests/fakes.py` + golden tests | Given a hand-written `Classification`, `build_note()` produces exactly `tests/golden/college.md`. **No network, no API keys.** |
| **S2** | `prompts.py` + `test_prompts.py` | Classifier prompt contains the generated schema; project prompt contains every supplied URL |
| **S3** | `adapters/sarvam_llm.py` + fake in tests | `pipeline.classify()` returns a `Classification` against `FakeLlm`; one live call succeeds manually |
| **S4** | `adapters/vault.py` | `write_note()` creates the file in `inbox/`, second call with the same title produces `-2` |
| **S5** | `adapters/sarvam_ocr.py` | A real marksheet photo returns text containing the issuer |
| **S5b** | `adapters/sarvam_text.py` | `/text-lid` returns `hi-IN` for a Hindi document and an English translation comes back; a 6,000-character document translates without a length error |
| **S6** | `bot.py`, `main.py` | Send a PDF from the phone; note appears in `inbox/`; Approve moves it |
| **S7** | `adapters/search.py` + project expansion | A one-liner returns a brief with three real URLs |

S1 is the important one. Building the renderer against a hand-written `Classification` means you can see the finished product before a single API call exists, and every later bug is attributable to the adapter that just came online.

---

## 12. Testing strategy

The seams that make testing cheap:

```python
# tests/fakes.py
class FakeLlm:
    def __init__(self, payload: dict): self.payload = payload
    async def complete_json(self, system, user): return self.payload
    async def complete_text(self, system, user): return "## Description\nA test."

class FakeOcr:
    def __init__(self, text: str): self.text = text
    async def extract_text(self, path): return self.text

class FakeSearch:
    async def search(self, query, k=5):
        return [SearchResult(title="Example", url="https://example.com", snippet="s")]

class TempVault:            # real filesystem, temporary directory
    ...
```

Three test layers, in order of value:

1. **Golden files** — `render.build_note()` against three fixtures. Catches every template regression. Update deliberately by regenerating and reading the diff.
2. **Pipeline with fakes** — `ingest()` end to end with `FakeLlm`/`FakeOcr`/`FakeSearch` and `TempVault`. Fast, no network, covers the branching.
3. **One live smoke test** — a single marked-slow test that hits Sarvam once. Run manually before a demo, not in CI.

No mocking library. Hand-written fakes are shorter and read better.

---

## 13. Vibe-coding playbook

Rules for working with an AI assistant on this repository. These are constraints on the assistant, not preferences.

### 13.1 One module per request

Ask for exactly one file. Provide: the target file's contract from §5, the signatures of the files it imports from, and the constraint list below. Ask for the implementation and its test in the same response.

Bad: "build the project". Good: "implement `render.py` to satisfy these five signatures; it may import only `models.py` and the standard library; here are the three golden files it must reproduce."

### 13.2 Constraints to paste into every prompt

```
Constraints:
- Python 3.12, type hints on every function.
- No new dependencies beyond pyproject.toml.
- No global state, no module-level clients, no singletons.
- No I/O in prompts.py or render.py.
- No bare except. Raise IngestError(stage, cause).
- No prompts inline in bot.py or pipeline.py.
- No dicts crossing module boundaries — use the models in models.py.
- Return the full file, not a diff.
```

### 13.3 Review checklist

Before accepting generated code, check five things:

1. Does it import anything it shouldn't? (the dependency rules in §2)
2. Are the types the ones from `models.py`, not look-alike local classes?
3. Is any I/O outside `adapters/`?
4. Is there a bare `except`, or an `except Exception` outside `bot.py`?
5. Does the test fail if you break the implementation? If not, the test is decorative.

### 13.4 Order of work

Always build pure before impure. If a change requires editing `bot.py` and `render.py`, change `render.py` first and re-run the golden tests. Handlers are where logic hides; keep them boring.

### 13.5 When the model drifts

If generated code adds an abstraction that wasn't asked for — a `NoteRepository`, a `BaseHandler`, a config loader for two values — delete it. Six modules is the design. Adding a seventh needs a reason that survives being said out loud.

### 13.6 Use Sarvam's own tooling while building

Sarvam ships tooling aimed at this workflow, and it removes the most common source of broken integration code: an assistant inventing SDK method names from memory.

- **Agent Skills** generate correct chat-completion code inside your editor:

  ```bash
  npx skills add sarvamai/skills --skill chat
  ```

- **The documentation MCP server** lets the assistant read the live docs rather than recall them: `https://docs.sarvam.ai/_mcp/server`
- **Every docs page has a Markdown twin** — append `.md` to any URL — and `docs.sarvam.ai/llms.txt` is a complete index. Paste the relevant page into the prompt instead of describing the API from memory.

Two rules for the adapters: never accept a Sarvam call written without the matching docs page in context, and always write the fake first. If the fake matches the port, a wrong SDK call fails inside one file.

---

## 14. Extension points

The design anticipates four additions without building them.

**Calendar lane.** Add `calendar.py` with a `CalendarPort` protocol and an `EventProposal` model. `bot.py` gains one handler and one keyboard. No change to `render.py`, `models.py` or the existing pipeline.

**Voice notes.** Saaras v4 accepts Telegram's OGG/Opus voice notes without transcoding. Add `CaptureKind.VOICE`, a `TranscribePort`, and one adapter call in `ingest()` before classification. Everything downstream is unchanged because `Capture.text` already carries text.

**Spoken summaries.** Bulbul v4 Flash turns a filed note's summary into a voice reply — a `TtsPort` with `speak(text, language) -> bytes`, plus one call in `bot.py` after filing. It is the highest-impact demo addition available: hearing a Hindi marksheet summarised aloud proves the Indic stack in a way no slide can.

**Duplicate detection.** Add `find_similar(text) -> Path | None` to `VaultPort`; `ingest()` checks it before writing. One call site.

Each is additive because the seams already exist. That is the test of whether the decomposition was right.

---

## 15. Anti-patterns

| Don't | Why |
|---|---|
| Parsing prose from the LLM | Use JSON mode and validate; regex on prose breaks on the first variation |
| Putting note content in callback data | 64-byte cap, invisible state, unrecoverable |
| Writing to the vault outside `VaultPort` | Breaks the fallback path and the temp-file atomicity |
| Making `render.py` async | It is pure; async there invites I/O |
| A `utils.py` | It becomes the place where logic goes to hide |
| Catching exceptions per stage in `pipeline.py` | Failures become invisible; catch once at the boundary |
| Regenerating golden files to make tests pass | The diff is the review |
| Building the calendar lane before filing works | The core loop is the demo |

---

## 16. Glossary

**Capture** — one inbound Telegram message, normalised into a typed model.
**Classification** — the LLM's verdict: category, title, summary, tags and optional document fields.
**Note** — rendered Markdown with frontmatter, not yet written to disk.
**Proposal** — a note plus the UI affordances needed to approve it.
**Port** — a `Protocol` describing one external capability.
**Adapter** — the concrete implementation of a port.
**Deps** — the frozen dataclass holding the five adapters, constructed once in `main.py`.
**Walking skeleton** — the thinnest end-to-end slice that produces a real output.

---

*End of design document.*
