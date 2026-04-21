"""FastAPI app factory.

Builds the app, wires in the routers, the exception handlers, and the
lifespan that opens/closes the asyncpg pool and constructs the CryptoBox.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from aios.api.routers import (
    agents,
    channel_bindings,
    connection_routing_rules,
    connections,
    environments,
    health,
    sessions,
    skills,
    vaults,
)
from aios.config import get_settings
from aios.crypto.vault import CryptoBox
from aios.db.pool import close_pool, create_pool
from aios.errors import install_exception_handlers
from aios.harness.procrastinate_app import app as procrastinate_app
from aios.logging import configure_logging, get_logger


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger("aios.api")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        log.info("api.startup", db_url=_redact_dsn(settings.db_url))
        pool = await create_pool(settings.db_url, max_size=settings.db_pool_max_size)
        crypto_box = CryptoBox.from_base64(settings.vault_key.get_secret_value())
        await procrastinate_app.open_async()
        app.state.pool = pool
        app.state.crypto_box = crypto_box
        app.state.procrastinate = procrastinate_app
        app.state.db_url = settings.db_url
        try:
            yield
        finally:
            log.info("api.shutdown")
            await procrastinate_app.close_async()
            await pool.close()
            await close_pool()

    app = FastAPI(
        title="aios",
        version="0.1.0",
        description="Open-source agent runtime: Postgres-backed sessions, "
        "Docker sandbox, any LiteLLM model.",
        lifespan=lifespan,
    )
    install_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(environments.router)
    app.include_router(agents.router)
    app.include_router(sessions.router)
    app.include_router(skills.router)
    app.include_router(vaults.router)
    app.include_router(connections.router)
    app.include_router(channel_bindings.router)
    app.include_router(connection_routing_rules.router)
    return app


def _redact_dsn(dsn: str) -> str:
    """Strip the password from a Postgres DSN for safe logging."""
    if "@" not in dsn or "://" not in dsn:
        return dsn
    scheme, _, rest = dsn.partition("://")
    auth, _, host_part = rest.partition("@")
    if ":" in auth:
        user, _, _ = auth.partition(":")
        return f"{scheme}://{user}:***@{host_part}"
    return dsn


# An app instance importable as ``aios.api.app:app`` for ``uvicorn``.
app: Any = create_app()
