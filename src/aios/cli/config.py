"""Client-side CLI configuration.

Reads from environment + ``.env`` just like the server config. The CLI shares
the ``AIOS_API_KEY`` env var with the server: on a dev box, the server reads
it to *check* bearer tokens and the CLI reads it to *send* one. No second
config surface.
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CliSettings(BaseSettings):
    """Env-driven CLI configuration.

    The ``AIOS_`` prefix matches the server's settings. ``CliSettings`` reads
    only the handful of keys the client needs; extras are ignored so a single
    ``.env`` can power both server and client.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIOS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    url: str = Field(
        default="http://localhost:8080",
        description="Base URL of the aios API server.",
    )
    api_key: str | None = Field(
        default=None,
        description="Bearer token sent to the API. Required for all protected endpoints.",
    )
    api_port: int = Field(
        default=8080,
        ge=1,
        le=65535,
        description="Port the server listens on; used to synthesize a default URL "
        "when AIOS_URL is not set explicitly.",
    )


@lru_cache(maxsize=1)
def get_cli_settings() -> CliSettings:
    """Process-wide CLI settings singleton."""
    return CliSettings()


def resolve_base_url(override: str | None) -> str:
    """Pick the effective API base URL.

    Priority: explicit flag > ``AIOS_URL`` env > ``http://127.0.0.1:{AIOS_API_PORT}``.
    The last fallback mirrors the playground pattern where the operator sets
    ``AIOS_API_PORT`` (8090) but never ``AIOS_URL`` explicitly.
    """
    if override is not None:
        return override.rstrip("/")
    if "AIOS_URL" in os.environ:
        return os.environ["AIOS_URL"].rstrip("/")
    return f"http://127.0.0.1:{get_cli_settings().api_port}"
