"""Runtime configuration.

``AIOS_SIGNAL_*`` env vars feed pydantic-settings.  These are *deployment-shape*
fields — host paths and ports baked into how the container is wired —
not credentials.  The phone (the account identity) lives on the
connection record's encrypted secrets, fetched at ``setup()`` time via
``HttpConnector.secrets()``.
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

    config_dir: Path
    cli_bin: str = "signal-cli"
    daemon_host: str = "127.0.0.1"
    daemon_port: int = 7583
