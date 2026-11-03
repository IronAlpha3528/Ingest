# `ingest` — Telegram → Obsidian Capture Pipeline

Content sent to a Telegram chat is read, classified, translated (if Indic), and filed into a structured Obsidian knowledge base without manual sorting.

Powered by **Sarvam AI** (Sarvam-105B LLM, Sarvam Vision OCR, Sarvam-Translate, and Language Identification).

## Setup

### Prerequisites
- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) package manager

### Environment Setup

```powershell
# 1. Install Python 3.12 via uv
uv python install 3.12

# 2. Create virtual environment
uv venv --python 3.12 .venv

# 3. Install dependencies in editable mode
uv pip install -e ".[dev]"

# 4. Configure environment variables
# Copy .env.example to .env and configure keys
```

### Running Tests

```powershell
# Run unit tests (offline, no network required)
uv run pytest tests/ -m "not slow" -v

# Run type checking
uv run mypy src
```
