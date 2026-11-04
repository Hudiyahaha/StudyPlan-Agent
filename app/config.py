from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8000
    model_provider: str = "groq"
    model_name: str = "qwen/qwen3.8-27b"
    fallback_models: str = "openai/gpt-oss-120b,openai/gpt-oss-20b"
    groq_api_key: str = ""
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_timeout_seconds: float = Field(default=15, gt=0, le=40)
    max_rate_limit_wait_seconds: float = Field(default=8, ge=0, le=30)
    max_steps: int = Field(default=6, ge=1, le=6)
    max_tool_retries: int = Field(default=2, ge=0, le=2)
    max_output_tokens: int = Field(default=900, ge=1)
    max_history_messages: int = Field(default=12, ge=0, le=40)
    max_history_chars: int = Field(default=24000, ge=0)
    max_external_chars: int = Field(default=6000, ge=0)
    run_timeout_seconds: float = Field(default=40, gt=0, le=40)
    tool_timeout_seconds: float = Field(default=10, gt=0, le=30)
    study_block_minutes: int = Field(default=90, ge=30, le=180)
    planning_horizon_days: int = Field(default=28, ge=1, le=120)
    preferred_start: str = "16:00"
    preferred_end: str = "22:00"
    timezone: str = "Asia/Karachi"
    reference_date: str = ""


settings = Settings()


def current_time() -> datetime:
    """Local wall-clock time. REFERENCE_DATE pins it for tests and repeatable experiments."""
    if settings.reference_date:
        return datetime.fromisoformat(settings.reference_date)
    return datetime.now(ZoneInfo(settings.timezone)).replace(tzinfo=None)
