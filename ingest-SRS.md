# Software Requirements Specification

**Project:** `ingest` — Telegram → Obsidian capture pipeline
**Document type:** SRS (prototype scope)
**Version:** 1.0 FINAL
**Date:** 3 November 2026
**Owner:** Abhay
**Companion document:** `ingest-DESIGN.md`

---

## 1. Purpose

`ingest` is a prototype that demonstrates one claim: **content sent to a chat can be read, understood, and filed into a structured personal knowledge base without manual sorting.**

A user forwards a document, photo or idea to a Telegram bot. The system extracts the content, decides what it is, writes a structured Markdown note into an Obsidian vault, and asks the user to confirm the filing with one tap.

This document defines the scope, requirements and acceptance criteria for the prototype. It deliberately excludes production concerns. The design that satisfies these requirements is specified separately.

---

## 2. Scope

### 2.1 In scope

| ID | Capability |
|---|---|
| C1 | Receive documents, photos and text messages over Telegram |
| C2 | Extract text from documents and images via OCR |
| C3 | Classify each capture into one of three categories and extract metadata |
| C4 | Render a structured Markdown note with YAML frontmatter |
| C5 | Present the proposal with Approve / Change folder / Discard controls |
| C6 | File the note and its attachment into the correct vault folder on approval |
| C7 | Expand a one-line project idea into a researched brief with cited sources |
| C8 | Detect the source language and normalise non-English content for summarisation |

### 2.2 Out of scope

Explicitly excluded from this version. Each is a candidate for a later increment, listed in §10.

Access control and user allowlisting; voice notes; photo albums; the Google Calendar lane; duplicate detection; undo; retry with backoff; spend caps; vault search; automatic related-note linking; project status lifecycle; recurring events; URL capture; internationalisation; packaging and deployment; any database.

### 2.3 System context

```
┌────────┐   Telegram    ┌──────────────┐   HTTPS    ┌─────────────┐
│  User  │──────────────►│  ingest bot  │───────────►│  Sarvam AI  │
└────────┘               │  (local)     │            │  LLM + OCR  │
     ▲                   └──────┬───────┘            └─────────────┘
     │                          │                    ┌─────────────┐
     │  proposal + buttons      ├───────────────────►│ Search API  │
     └──────────────────────────┤                    └─────────────┘
                                │  writes
                                ▼
                         ┌─────────────┐
                         │ Obsidian    │
                         │ vault (fs)  │
                         └─────────────┘
```

Actors: **User** (single, the owner). **Operator** (same person; runs the process).

External systems: Telegram Bot API; Sarvam AI (chat completion, document intelligence, language identification and translation); one web search API; the local filesystem.

---

## 3. Functional requirements

Each requirement carries a testable acceptance criterion. A requirement is met only when its criterion can be demonstrated.

### FR-1 — Capture intake
The system **shall** accept Telegram messages of type `document`, `photo` and `text`, including any caption.

*Acceptance:* sending a PDF, a JPEG and a plain sentence each produce a system response within 2 seconds acknowledging receipt.

### FR-2 — Media retrieval
The system **shall** download document and photo content to a local temporary path before processing.

*Acceptance:* after sending a PDF, the file exists on disk with the original filename preserved.

### FR-3 — Text extraction
The system **shall** extract text from documents and images using OCR. For text messages, the message body is the extracted text.

*Acceptance:* a photographed marksheet yields a text string containing the issuer name visible in the image.

### FR-4 — Classification
The system **shall** classify each capture into exactly one of `college-documents`, `possible-projects`, `reference`, and **shall** return a title, a summary, and a list of tags. For `college-documents` it **shall additionally** return document type, issuer, issue date and any reference numbers.

*Acceptance:* the response validates against the `Classification` schema; no free-text parsing is required to read it.

### FR-5 — Note rendering
The system **shall** render a Markdown note containing YAML frontmatter and a category-specific body, and **shall** write it to `inbox/`.

*Acceptance:* the written file opens in Obsidian with frontmatter recognised as properties, and the filename follows `YYYY-MM-DD - Title.md`.

### FR-6 — Proposal
The system **shall** present the proposal to the user containing the suggested category, title, a body preview, and exactly three inline controls: Approve, Change folder, Discard.

*Acceptance:* one message appears containing all three controls; tapping any produces a state change.

### FR-7 — Approval and filing
On Approve, the system **shall** move the note from `inbox/` to the target category folder, move the attachment to `attachments/<category>/`, and set frontmatter `status: filed`.

*Acceptance:* after approval the note is absent from `inbox/`, present in the target folder, and its attachment path resolves.

### FR-8 — Correction
On Change folder, the system **shall** present the three categories, move the note to the chosen one, and append a record of the correction to `_corrections.jsonl`.

*Acceptance:* the chosen folder contains the note; `_corrections.jsonl` has one new line naming the original and chosen category.

### FR-9 — Discard
On Discard, the system **shall** move the note and attachment to `_archive/` without deleting either.

*Acceptance:* files exist under `_archive/` after discard.

### FR-10 — Project expansion
For captures classified as `possible-projects`, the system **shall** perform a web search and generate a brief containing: the original one-liner verbatim, a description, three to five named comparable projects or competitors **with source URLs**, research areas, anticipated problems, and open questions.

*Acceptance:* the produced note contains at least three distinct URLs, and every named comparable has an adjacent URL.

### FR-11 — Failure reporting
When any external call fails, the system **shall** reply to the user with a short explanation and **shall not** write a partial note.

*Acceptance:* with the API key invalidated, sending a document produces an error message and leaves the vault unchanged.

### FR-12 — Language normalisation
The system **shall** detect the language of extracted text and, where it is not English, obtain an English translation before classification, while preserving original-script values for extracted fields.

*Acceptance:* a Hindi marksheet yields an English summary and an `issuer` value in Devanagari, and its frontmatter records `source_language: hi-IN`.

---

## 4. Non-functional requirements

| ID | Requirement | Measure |
|---|---|---|
| NFR-1 | Runs on Python 3.12 on macOS or Linux | Starts with one command on a clean machine with dependencies installed |
| NFR-2 | No secrets in source | API keys read from environment; repository contains no key material |
| NFR-3 | No database | State lives in the vault filesystem only |
| NFR-4 | Small surface | Total implementation under roughly 500 lines across all modules |
| NFR-5 | Single-user, single-process | One bot instance, foreground process |
| NFR-6 | Observable | Each pipeline stage logs one line: stage name, duration, outcome |
| NFR-7 | Responsive | Acknowledges every inbound message within 2 seconds; long work happens after acknowledgement |
| NFR-8 | Recoverable by resend | Any failure is resolved by re-sending the message; no hidden state to repair |
| NFR-9 | Data locality | Originals and notes stay on the local filesystem; extracted text is sent only to Sarvam and, for project briefs, search queries only |

Latency of the full pipeline is **not** a requirement at this stage. It is measured, not promised.

---

## 5. External interface requirements

### 5.1 Telegram Bot API
Long polling. Bot token from `TELEGRAM_BOT_TOKEN`. No webhook, no public endpoint, no TLS termination.

### 5.2 Sarvam AI

One API key covers every model used, through the `sarvamai` SDK.

| Purpose | Endpoint | Model |
|---|---|---|
| Classification, field extraction, brief writing | `POST /v2/chat/completions` | `sarvam-105b` |
| OCR | Document intelligence | Sarvam Vision |
| Language detection | `POST /text-lid` | — |
| Translation to English | `POST /translate` | `sarvam-translate:v1` (default) or `mayura:v1` |
| Model discovery | `GET /v2/models` | — |

Authentication via `SARVAM_API_KEY`. All model output consumed by the system is JSON and is validated before use. `/translate` accepts 2,000 characters per request on `sarvam-translate:v1` and 1,000 on `mayura:v1`; longer text is chunked at sentence boundaries by the adapter.

### 5.3 Web search
One provider, one API key, accessed through a single adapter. Queries contain only the project idea text — never document contents or personal identifiers.

### 5.4 Filesystem
Read and write access to one vault directory, configured by `VAULT_PATH`. The system creates missing subdirectories at start-up and refuses to start if the root is absent.

---

## 6. Data requirements

Every note carries frontmatter with these fields:

| Field | Type | Required |
|---|---|---|
| `title` | string | yes |
| `category` | enum | yes |
| `created` | ISO date | yes |
| `source` | string (`telegram`) | yes |
| `status` | enum (`inbox`, `filed`) | yes |
| `original_filename` | string | when a file was sent |
| `attachment` | vault-relative path | when a file was sent |
| `tags` | list of strings | yes |
| `document_type`, `issuer`, `issued_date`, `reference_ids` | string / list | college documents only |

Non-English sources record `source_language` as a BCP-47 code (for example `hi-IN`). Extracted values retain their original script; prose is written in English.

---

## 7. Acceptance criteria — definition of done

The prototype is complete when this script runs end to end without intervention:

1. Start the bot. It reports ready.
2. Photograph a marksheet and send it. Within one message cycle a proposal appears naming the issuer and date.
3. Tap Approve. The note appears in `college-documents/` and its attachment in `attachments/college-documents/`.
4. Send the sentence *"an app that tracks my hostel mess bills"*. A brief returns containing named competitors with URLs.
5. Tap Change folder, choose `reference`. The note moves and `_corrections.jsonl` gains a line.
6. Send a document with an invalid API key. An error is reported and the vault is unchanged.

---

## 8. Assumptions

- The user has a Sarvam account, a Telegram account and an Obsidian vault on the same machine.
- One user, one vault, one bot. No concurrency.
- The laptop is awake and online during use; messages sent while offline are not recovered beyond Telegram's own 24-hour update window.
- Documents are under the Bot API's 20 MB download limit.
- English or Indian-language content; no right-to-left layout handling beyond what OCR returns.

---

## 9. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Sarvam Vision call signature differs from assumption | Blocks FR-3 | OCR sits behind an adapter; a PyMuPDF text-extraction fallback covers digital PDFs |
| OCR quality on photographs is poor | Weak demo | Pre-test with real photographs; show a digital PDF if photos disappoint |
| Search API returns thin results | Weak brief | Prompt requires URLs only for comparables actually returned by search |
| Classification misfiles | Visible failure | Change folder control makes correction a one-tap recovery |
| Venue network unreliable | Demo fails | Keep one successful run of each pipeline saved and narratable |

---

## 10. Future increments

In rough order of value: voice notes via speech-to-text; the Google Calendar lane for forwarded events; duplicate detection; access control; related-note linking; project status lifecycle; vault search; recurring events.

---

## 11. Appendix — Sarvam component utilisation

| Component | Role in this system | Requirement |
|---|---|---|
| Sarvam Vision (Document Intelligence) | OCR of PDFs, photos and scans | FR-3 |
| Language Identification (`/text-lid`) | Detect the source language | FR-12 |
| Sarvam-Translate (`sarvam-translate:v1`) | English normalisation of non-English text | FR-12 |
| Mayura (`mayura:v1`) | Alternative translation for colloquial or code-mixed input | FR-12 |
| Sarvam-105B (`sarvam-105b`) | Classification, field extraction, brief writing | FR-4, FR-10 |
| Saaras v4 | Voice notes — future increment | §10 |
| Bulbul v4 Flash | Spoken summaries — future increment | §10 |

All are reached with a single `SARVAM_API_KEY` through the `sarvamai` SDK. Model identifiers are configuration rather than code: `/v2` is OpenAI-compatible, so any model the key can reach is substitutable without a code change.

---

*End of specification.*
