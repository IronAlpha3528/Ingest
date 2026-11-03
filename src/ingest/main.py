"""Application entrypoint for the ingest Telegram bot pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from sarvamai import SarvamAI
from telegram.ext import Application

from ingest.adapters.sarvam_llm import SarvamLlm
from ingest.adapters.sarvam_ocr import SarvamOcr
from ingest.adapters.sarvam_text import SarvamText
from ingest.adapters.search import TavilySearch
from ingest.adapters.vault import FileVault
from ingest.bot import setup_bot_handlers
from ingest.config import Settings
from ingest.pipeline import Deps

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("ingest")


def build_deps(settings: Settings) -> Deps:
    """Instantiate and wire concrete adapters into the Deps container."""
    client = SarvamAI(api_subscription_key=settings.sarvam_api_key)

    llm = SarvamLlm(
        client=client,
        api_key=settings.sarvam_api_key,
        model=settings.model,
    )
    ocr = SarvamOcr(
        client=client,
        api_key=settings.sarvam_api_key,
        ocr_model=settings.ocr_model,
    )
    text = SarvamText(
        client=client,
        api_key=settings.sarvam_api_key,
        chunk_chars=settings.translate_chunk_chars,
        translate_model=settings.translate_model,
    )
    search = TavilySearch(api_key=settings.search_api_key)
    vault = FileVault(root=settings.vault_path)

    return Deps(llm=llm, ocr=ocr, text=text, search=search, vault=vault)


def main() -> None:
    """Start the Telegram polling loop with graceful shutdown."""
    try:
        settings = Settings()
    except Exception as e:
        logger.error("Configuration error. Please verify your .env file: %s", e)
        sys.exit(1)

    logger.info("Initializing vault at %s", settings.vault_path)
    deps = build_deps(settings)

    logger.info("Initializing Telegram bot application...")
    app = Application.builder().token(settings.telegram_bot_token).build()
    setup_bot_handlers(app, deps)

    logger.info("Bot ready! Listening for messages...")
    try:
        app.run_polling()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopping cleanly...")


if __name__ == "__main__":
    main()
