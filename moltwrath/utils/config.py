"""Configuration management using pydantic-settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App
    app_name: str = "moltwrath"
    debug: bool = False
    log_level: str = "INFO"
    secret_key: str = "change-me-in-production"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # LLM
    llm_provider: str = "openai"
    openai_api_key: str = ""
    openai_base_url: str = ""
    anthropic_api_key: str = ""

    # Database
    db_url: str = "sqlite+aiosqlite:///moltwrath.db"

    # Agent defaults
    default_model: str = "gpt-4"
    default_temperature: float = 0.7
    default_max_tokens: int = 4096
    max_agent_iterations: int = 10

    model_config = {"env_prefix": "MOLTWRATH_", "env_file": ".env", "extra": "ignore"}


# Singleton
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
