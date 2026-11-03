"""Sarvam Text adapter implementing TextPort.

Scoped to English-only per user directive: bypasses Indic translation calls
and passes text through directly.
"""

from __future__ import annotations

import inspect
import logging
import os
import re
from typing import Any

import httpx
from sarvamai import SarvamAI

from ingest.models import IngestError

logger = logging.getLogger(__name__)


class SarvamText:
    """Language detection and translation adapter (configured for English-only)."""

    def __init__(
        self,
        client: Any = None,
        api_key: str | None = None,
        base_url: str = "https://api.sarvam.ai",
        chunk_chars: int = 2000,
        translate_model: str = "sarvam-translate:v1",
    ) -> None:
        self.client = client
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.chunk_chars = chunk_chars
        self.translate_model = translate_model

    def chunk_text(self, text: str, max_chars: int | None = None) -> list[str]:
        """Split text at sentence boundaries respecting character limit."""
        limit = max_chars or self.chunk_chars
        if len(text) <= limit:
            return [text]

        sentences = re.split(r"(?<=[.!?\n])\s+", text)
        chunks: list[str] = []
        current = ""

        for s in sentences:
            if not s:
                continue
            if len(current) + len(s) + 1 <= limit:
                current = f"{current} {s}".strip() if current else s
            else:
                if current:
                    chunks.append(current)
                # If a single sentence exceeds the limit, split hard
                if len(s) > limit:
                    for i in range(0, len(s), limit):
                        chunks.append(s[i : i + limit])
                    current = ""
                else:
                    current = s

        if current:
            chunks.append(current)

        return chunks

    async def detect(self, text: str) -> str:
        """Detect language code. Calls /text-lid if configured or returns 'en-IN'."""
        if not text or not text.strip():
            return "en-IN"

        # Check client capability
        if self.client and hasattr(self.client, "text") and hasattr(self.client.text, "identify_language"):
            try:
                res = self.client.text.identify_language(input=text[:500])
                if inspect.isawaitable(res):
                    res = await res
                if hasattr(res, "language_code") and res.language_code:
                    return str(res.language_code)
            except Exception as e:
                logger.warning("Sarvam client language identification call failed: %s", e)

        # Check direct HTTP endpoint if api_key available
        if self.api_key:
            try:
                async with httpx.AsyncClient(timeout=10.0) as http_client:
                    res = await http_client.post(
                        f"{self.base_url}/text-lid",
                        headers={"api-subscription-key": self.api_key},
                        json={"input": text[:500]},
                    )
                    if res.status_code in (401, 403):
                        logger.error("Sarvam API key authentication failed on /text-lid: HTTP %s", res.status_code)
                        raise IngestError("download", f"Sarvam API auth failed: HTTP {res.status_code}")
                    if res.is_success:
                        data = res.json()
                        lang = data.get("language_code")
                        if lang:
                            return str(lang)
                    else:
                        logger.warning("Sarvam /text-lid returned non-success HTTP %s: %s", res.status_code, res.text)
            except IngestError:
                raise
            except Exception as e:
                logger.warning("Sarvam /text-lid request exception: %s", e)

        return "en-IN"

    async def to_english(self, text: str, source: str) -> str:
        """Translate to English. In English-only mode, returns text directly."""
        return text

