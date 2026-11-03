from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    telegram_bot_token: str = ""
    sarvam_api_key: str = ""
    search_api_key: str = ""
    vault_path: Path = Path("vault")

    # Model choice is configuration, not code — /v2 is OpenAI-compatible.
    model: str = "sarvam-105b-conversations"
    translate_model: str = "sarvam-translate:v1"   # or "mayura:v1" for colloquial input
    ocr_model: str = "sarvam-vision"

    max_download_mb: int = 20
    translate_chunk_chars: int = 2000

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
