"""Runtime configuration.

``AIOS_SIGNAL_*`` env vars feed pydantic-settings.  Pre-PR3 the
connector also took ``AIOS_URL`` / ``AIOS_API_KEY`` /
``AIOS_CONNECTION_ID`` because it ran as a separate process and POSTed
inbound to aios over HTTP; PR3's stdio model makes the parent process
aios itself, so those fields are gone.

MCP bind/token are also gone — the MCP server now runs over stdio
(no HTTP bind, no bearer token).  Daemon host/port are still
configurable since signal-cli's TCP daemon is internal to this
subprocess and operators may need to avoid local port collisions.

Multi-account: ``phones`` is a CSV of registered numbers (e.g.
``AIOS_SIGNAL_PHONES=+1555,+1666``).  signal-cli's daemon is launched
without ``-a`` so it serves all registered accounts; every RPC call
includes the target account in its params.  Single-phone setups still
work — set ``AIOS_SIGNAL_PHONES=+1555`` (a one-element list).
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
    phones: list[str]
    config_dir: Path
    cli_bin: str = "signal-cli"
    daemon_host: str = "127.0.0.1"
    daemon_port: int = 7583
