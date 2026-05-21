"""Runtime configuration.

``AIOS_WHATSAPP_*`` env vars feed pydantic-settings.  These are
*deployment-shape* fields — host paths and binary names baked into how
the container is wired — not credentials.  The phone (the account
identity) lives on each connection record's encrypted secrets and is
fetched per-connection via ``HttpConnector.secrets()``.
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

    # Host directory bind-mounted into the container; per-phone
    # subdirectories hold whatsmeow's sqlstore (``store.db``) plus the
    # daemon's downloaded media cache (``media/``).  Required: without a
    # persistent store, every container restart forces a fresh pairing.
    data_dir: Path

    # Daemon binary name/path.  Default assumes ``whatsapp-daemon`` is on
    # ``$PATH`` (the Dockerfile installs it to ``/usr/local/bin/``).
    daemon_bin: str = "whatsapp-daemon"

    # Loopback host the daemon listens on; only ever 127.0.0.1 in production.
    daemon_host: str = "127.0.0.1"

    # Default port for single-daemon smoke testing.  The connector
    # allocates ephemeral ports per-connection in production; this
    # default exists so a single ``whatsapp-daemon`` can be run by hand
    # and probed without a wrapping Python process.
    daemon_port: int = 7584
