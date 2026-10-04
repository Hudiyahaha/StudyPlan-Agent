from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8000
    model_provider: str = "grok"
    model_name: str = "grok-4.3"
    xai_api_key: str = ""
    max_steps: int = Field(default=6, ge=1, le=6)
    max_tool_retries: int = Field(default=2, ge=0, le=2)
    max_output_tokens: int = Field(default=512, ge=1)
    run_timeout_seconds: float = Field(default=40, gt=0, le=40)
    tool_timeout_seconds: float = Field(default=10, gt=0, le=30)
    study_block_minutes: int = Field(default=90, ge=30, le=180)
    preferred_start: str = "16:00"
    preferred_end: str = "22:00"


settings = Settings()
