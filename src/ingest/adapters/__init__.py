"""Adapters module for external service integration and filesystem persistence."""

from ingest.adapters.ports import LlmPort, OcrPort, SearchPort, TextPort, VaultPort
from ingest.adapters.sarvam_llm import SarvamLlm
from ingest.adapters.sarvam_ocr import SarvamOcr
from ingest.adapters.sarvam_text import SarvamText
from ingest.adapters.search import TavilySearch
from ingest.adapters.vault import FileVault

__all__ = [
    "LlmPort",
    "OcrPort",
    "TextPort",
    "SearchPort",
    "VaultPort",
    "SarvamLlm",
    "SarvamOcr",
    "SarvamText",
    "TavilySearch",
    "FileVault",
]

