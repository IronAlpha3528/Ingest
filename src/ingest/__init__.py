"""ingest — Telegram to Obsidian capture pipeline powered by Sarvam AI."""

__version__ = "0.1.0"

try:
    import pydantic.version
    pydantic.version._ensure_pydantic_core_version = lambda: None  # type: ignore[assignment]
except Exception:
    pass
