"""FastAPI dependencies: auth, DB pool, crypto box.

Each dependency is async and resolved per request via FastAPI's dep injection.
The pool and crypto box are stored on ``request.app.state`` at startup, then
exposed here as typed accessors.
"""

from __future__ import annotations

import secrets
from typing import Annotated, cast

import asyncpg
from fastapi import Depends, Header, HTTPException, Request, status
from procrastinate import App as ProcrastinateApp

from aios.config import Settings, get_settings
from aios.crypto.vault import CryptoBox
from aios.errors import UnauthorizedError
from aios.models.connections import Connection


def get_pool(request: Request) -> asyncpg.Pool:
    return cast("asyncpg.Pool", request.app.state.pool)


def get_crypto_box(request: Request) -> CryptoBox:
    crypto_box: CryptoBox = request.app.state.crypto_box
    return crypto_box


def get_procrastinate(request: Request) -> ProcrastinateApp:
    app: ProcrastinateApp = request.app.state.procrastinate
    return app


def get_db_url(request: Request) -> str:
    db_url: str = request.app.state.db_url
    return db_url


def get_settings_dep() -> Settings:
    return get_settings()


def require_bearer_auth(
    settings: Annotated[Settings, Depends(get_settings_dep)],
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> None:
    """Verify the request carries a valid bearer token.

    Compares the supplied token against ``AIOS_API_KEY`` using a
    constant-time comparison so the comparison itself doesn't leak timing
    information about the key.
    """
    if authorization is None:
        raise UnauthorizedError("missing Authorization header")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="expected `Authorization: Bearer <key>`",
        )
    expected = settings.api_key.get_secret_value()
    if not secrets.compare_digest(token, expected):
        raise UnauthorizedError("invalid api key")


# Type aliases for clarity at the route definitions.
# Note: asyncpg.Pool isn't subscriptable at runtime, so we annotate with the
# bare class. FastAPI's dependency injection ignores generic parameters.
PoolDep = Annotated[asyncpg.Pool, Depends(get_pool)]
CryptoBoxDep = Annotated[CryptoBox, Depends(get_crypto_box)]
ProcrastinateDep = Annotated[ProcrastinateApp, Depends(get_procrastinate)]
DbUrlDep = Annotated[str, Depends(get_db_url)]
AuthDep = Annotated[None, Depends(require_bearer_auth)]


async def resolve_connection(connection_id: str, pool: PoolDep) -> Connection:
    """Resolve a ``{connection_id}`` path param to a ``Connection`` row.

    Used by all endpoints nested under ``/v1/connections/{connection_id}/...``
    to fail fast with 404 before any business logic runs.
    """
    from aios.services import connections as service

    return await service.get_connection(pool, connection_id)


ConnectionDep = Annotated[Connection, Depends(resolve_connection)]
