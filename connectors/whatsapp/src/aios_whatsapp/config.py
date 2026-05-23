"""Runtime configuration.

``AIOS_WHATSAPP_*`` env vars feed pydantic-settings.  Deployment-shape
only — the phone (account identity) lives on each connection record's
encrypted secrets and arrives per-connection via ``HttpConnector.secrets()``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AIOS_WHATSAPP_",
        extra="ignore",
        populate_by_name=True,
    )

    data_dir: Path
    daemon_bin: str = "whatsapp-daemon"
    daemon_host: str = "127.0.0.1"
