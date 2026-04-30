"""Runtime configuration.

Pydantic-settings sourcing: ``AIOS_TELEGRAM_*`` env vars for connector-specific
settings. All may be overridden on the CLI (see ``__main__.py``).
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AIOS_TELEGRAM_",
        extra="ignore",
        populate_by_name=True,
    )

    # Telegram
    bot_token: str

    # MCP server
    mcp_bind: str = "127.0.0.1:9200"
    mcp_token: str
