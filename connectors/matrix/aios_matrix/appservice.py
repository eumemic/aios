from __future__ import annotations

from typing import cast

from mautrix.appservice import AppService
from mautrix.appservice.state_store import ASStateStore
from mautrix.appservice.state_store.asyncpg import PgASStateStore
from mautrix.util.async_db import Database

from .config import MatrixConfig

_USE_POSTGRES = object()


class PostgresStateStore(PgASStateStore):
    """PgASStateStore whose lifecycle owns the underlying asyncpg pool."""

    async def open(self) -> None:
        await self.db.start()

    async def close(self) -> None:
        await self.db.stop()


def create_appservice(
    config: MatrixConfig, state_store: ASStateStore | None | object = _USE_POSTGRES
) -> AppService:
    """Build the receiver, using a durable mautrix state store by default."""
    if state_store is _USE_POSTGRES:
        database = Database.create(config.database_url, owner_name="aios-matrix")
        state_store = PostgresStateStore(database)
    return AppService(
        server=config.hs_url,
        domain=config.server_name,
        as_token=config.as_token,
        hs_token=config.hs_token,
        bot_localpart=config.sender_localpart,
        id="aios-matrix",
        state_store=cast(ASStateStore, state_store),
    )
