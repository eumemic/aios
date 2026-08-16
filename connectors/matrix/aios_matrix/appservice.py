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
    config: MatrixConfig,
    state_store: ASStateStore | object | None = _USE_POSTGRES,
    appservice_class: type[AppService] = AppService,
) -> AppService:
    """Build the receiver, using a durable mautrix state store by default.

    ``appservice_class`` MUST be supplied by callers that override any HTTP
    route handler.  ``AppService.__init__`` calls ``register_routes()``, which
    binds ``self._http_handle_transaction`` into the aiohttp router at
    construction time; reassigning ``__class__`` afterwards does not rebind
    the already-bound handler, so a subclass installed post-construction is
    silently dead code.  The class must therefore be chosen up front.
    """
    if state_store is _USE_POSTGRES:
        database = Database.create(
            config.database_url,
            upgrade_table=PgASStateStore.upgrade_table,
            owner_name="aios-matrix",
        )
        state_store = PostgresStateStore(database)
    return appservice_class(
        server=config.hs_url,
        domain=config.server_name,
        as_token=config.as_token,
        hs_token=config.hs_token,
        bot_localpart=config.sender_localpart,
        id="aios-matrix",
        state_store=cast(ASStateStore, state_store),
    )
