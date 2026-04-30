"""Runtime configuration.

Pydantic-settings sourcing: ``AIOS_SIGNAL_*`` env vars for connector-specific
settings, plus three shared aios env vars via explicit aliases
All may be overridden on the CLI (see ``__main__.py``).
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AIOS_SIGNAL_",
        extra="ignore",
        populate_by_name=True,
    )

    # Signal / daemon
    phone: str
    config_dir: Path
    cli_bin: str = "signal-cli"
    daemon_host: str = "127.0.0.1"
    daemon_port: int = 7583

    # MCP server
    mcp_bind: str = "127.0.0.1:9100"
    mcp_token: str
