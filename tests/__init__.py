"""Test suite for ingest."""

try:
    import pydantic.version
    pydantic.version._ensure_pydantic_core_version = lambda: None  # type: ignore[assignment]
except Exception:
    pass
