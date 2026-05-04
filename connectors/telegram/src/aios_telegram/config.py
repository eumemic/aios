"""Runtime configuration.

``AIOS_TELEGRAM_*`` env vars feed pydantic-settings.  Pre-PR3 the
connector also took ``AIOS_URL`` / ``AIOS_API_KEY`` /
``AIOS_CONNECTION_ID`` because it ran as a separate process and POSTed
inbound to aios over HTTP; PR3's stdio model makes the parent process
aios itself, so those fields are gone.

MCP bind/token are also gone — the MCP server now runs over stdio
(no HTTP bind, no bearer token).
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AIOS_TELEGRAM_",
        extra="ignore",
        populate_by_name=True,
    )

    bot_token: str
