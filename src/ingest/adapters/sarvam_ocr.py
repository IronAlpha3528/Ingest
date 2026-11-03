"""Sarvam OCR adapter implementing OcrPort with Sarvam DocAI (Document Intelligence) and PyMuPDF fallback."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import time
from typing import Any

import httpx

from ingest.models import IngestError

logger = logging.getLogger(__name__)


class SarvamOcr:
    """Extracts text from documents and images using Sarvam DocAI and PyMuPDF."""

    def __init__(
        self,
        client: Any = None,
        api_key: str | None = None,
        base_url: str = "https://api.sarvam.ai",
        ocr_model: str = "sarvam-vision",
    ) -> None:
        self.client = client
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.ocr_model = ocr_model

    async def _call_sarvam_ocr(self, path: Path) -> str:
        """Submit a document/image to Sarvam DocAI and poll for extracted text blocks."""
        if not self.api_key:
            raise ValueError("SARVAM_API_KEY is not configured for DocAI")

        suffix = path.suffix.lower()
        mime_types = {
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".tiff": "image/tiff",
        }
        content_type = mime_types.get(suffix, "application/octet-stream")

        headers = {"api-subscription-key": self.api_key}

        async with httpx.AsyncClient(timeout=45.0) as http_client:
            logger.info("Uploading %s to Sarvam DocAI for OCR extraction...", path.name)
            with open(path, "rb") as f:
                file_bytes = f.read()

            files = {"file": (path.name, file_bytes, content_type)}
            data = {"language": "en-IN"}

            # Start DocAI parse job
            parse_url = f"{self.base_url}/doc-ai/v1/job/parse"
            res = await http_client.post(parse_url, headers=headers, files=files, data=data)
            if res.status_code != 200:
                logger.error("Sarvam DocAI parse initiation failed: HTTP %s - %s", res.status_code, res.text)
                res.raise_for_status()

            job_data = res.json()
            job_id = job_data.get("job_id")
            if not job_id:
                raise ValueError(f"No job_id returned by Sarvam DocAI: {job_data}")

            logger.info("Sarvam DocAI job started (job_id=%s). Polling for completion...", job_id)

            # Poll for completion (up to 30 seconds)
            status_url = f"{self.base_url}/doc-ai/v1/job/{job_id}/status"
            results_url = f"{self.base_url}/doc-ai/v1/job/{job_id}/results"

            terminal_status = None
            for attempt in range(15):
                await asyncio.sleep(2.0)
                st_res = await http_client.get(status_url, headers=headers)
                if st_res.status_code == 200:
                    status_info = st_res.json()
                    current_status = status_info.get("status")
                    logger.debug("DocAI job %s status: %s (attempt %d/15)", job_id, current_status, attempt + 1)
                    if current_status in ("completed", "partially_completed"):
                        terminal_status = current_status
                        break
                    elif current_status in ("failed", "rejected"):
                        raise ValueError(f"Sarvam DocAI job {job_id} terminated with status: {current_status}")

            if not terminal_status:
                raise TimeoutError(f"Sarvam DocAI job {job_id} timed out waiting for completion")

            # Fetch results
            res_data_resp = await http_client.get(results_url, headers=headers)
            res_data_resp.raise_for_status()
            results_data = res_data_resp.json()

            extracted_lines: list[str] = []
            for doc in results_data.get("documents", []):
                for page in doc.get("pages", []):
                    for block in page.get("blocks", []):
                        text_val = block.get("text", "").strip()
                        if text_val:
                            extracted_lines.append(text_val)

            full_text = "\n".join(extracted_lines).strip()
            logger.info("Sarvam DocAI successfully extracted %d text blocks (%d chars)", len(extracted_lines), len(full_text))
            return full_text

    # Alias for backwards compatibility with any internal references
    _call_sarvam_docai = _call_sarvam_ocr

    async def extract_text(self, path: Path) -> str:
        """Asynchronously extract text from a document or image path."""
        path = Path(path)
        if not path.exists():
            raise IngestError("ocr", FileNotFoundError(f"File not found: {path}"))

        suffix = path.suffix.lower()

        # Fast path 1: Plain text files
        if suffix in (".txt", ".md", ".csv", ".json"):
            return path.read_text(encoding="utf-8").strip()

        # Fast path 2: Digital PDF with clean text (> 50 words) via PyMuPDF
        fitz_text = ""
        if suffix == ".pdf":
            try:
                import pymupdf
                doc = pymupdf.open(path)
                pages_text = [page.get_text() for page in doc]
                doc.close()
                fitz_text = "\n\n".join(t.strip() for t in pages_text if t.strip())
                words = fitz_text.split()
                if len(words) > 50:
                    logger.info("Extracted %d words directly from digital PDF %s via PyMuPDF", len(words), path.name)
                    return fitz_text
            except Exception as e:
                logger.warning("PyMuPDF PDF extraction failed on %s: %s", path.name, e)

        # Scanned documents, photos, images: call Sarvam DocAI
        try:
            extracted = await self._call_sarvam_ocr(path)
            if extracted and extracted.strip():
                return extracted.strip()
        except IngestError:
            raise
        except Exception as ocr_err:
            logger.error("Sarvam DocAI extraction failed for %s: %s", path.name, ocr_err, exc_info=True)
            if fitz_text.strip():
                logger.info("Falling back to sparse text extracted by PyMuPDF (%d chars)", len(fitz_text))
                return fitz_text.strip()
            raise IngestError("ocr", ocr_err) from ocr_err

        if fitz_text.strip():
            return fitz_text.strip()

        raise IngestError("ocr", f"No readable text extracted from {path.name}")
