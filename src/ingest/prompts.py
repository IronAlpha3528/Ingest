from __future__ import annotations

import json

from ingest.models import Capture, Classification, SearchResult

# v1 — from ingest-DESIGN.md §6.1
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

# v1 — from ingest-DESIGN.md §6.2
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

EXTRACT_SYSTEM = """\
You extract metadata and factual details from the provided document text verbatim in its original script.
Return the extracted key details accurately.
"""


def classify_prompt(
    capture: Capture,
    original: str,
    translation: str | None,
) -> tuple[str, str]:
    """Appends Classification.model_json_schema() to system prompt.
    Formats user message with metadata, ORIGINAL block and optional ENGLISH TRANSLATION block.
    """
    schema = json.dumps(Classification.model_json_schema(), indent=2)
    system = f"{CLASSIFY_SYSTEM}\n\nSchema:\n{schema}"

    user_parts: list[str] = []
    if capture.filename:
        user_parts.append(f"Filename: {capture.filename}")
    if capture.caption:
        user_parts.append(f"Caption: {capture.caption}")

    user_parts.append(f"ORIGINAL:\n{original}")
    if translation and translation.strip():
        user_parts.append(f"ENGLISH TRANSLATION:\n{translation.strip()}")

    user = "\n\n".join(user_parts)
    return system, user


def project_prompt(
    idea: str,
    results: list[SearchResult],
) -> tuple[str, str]:
    """Formats project idea and search results into user message with 'title — snippet — url' blocks."""
    system = PROJECT_SYSTEM

    user_lines: list[str] = [f"Idea: {idea}", "", "Search results:"]
    if not results:
        user_lines.append("(No search results available)")
    else:
        for r in results:
            user_lines.append(f"- {r.title} — {r.snippet} — {r.url}")

    user = "\n".join(user_lines)
    return system, user


def extract_prompt(
    original: str,
    translation: str | None,
) -> tuple[str, str]:
    """Prompt for additional field extraction if needed."""
    system = EXTRACT_SYSTEM
    user_parts: list[str] = [f"ORIGINAL:\n{original}"]
    if translation and translation.strip():
        user_parts.append(f"ENGLISH TRANSLATION:\n{translation.strip()}")

    user = "\n\n".join(user_parts)
    return system, user
