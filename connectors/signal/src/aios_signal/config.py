"""Runtime configuration.

``AIOS_SIGNAL_*`` env vars feed pydantic-settings.  In the new
HTTP-client architecture (#301) each connector container runs one
account; multi-phone deployments use multiple containers, each with
its own ``AIOS_CONNECTOR_TOKEN`` and ``AIOS_SIGNAL_PHONE``.
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

    phone: str
    config_dir: Path
    cli_bin: str = "signal-cli"
    daemon_host: str = "127.0.0.1"
    daemon_port: int = 7583
