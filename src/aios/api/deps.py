"""FastAPI dependencies: auth, DB pool, crypto box.

Each dependency is async and resolved per request via FastAPI's dep injection.
The pool and crypto box are stored on ``request.app.state`` at startup, then
exposed here as typed accessors.
"""

from __future__ import annotations

import secrets
from typing import Annotated, cast

import asyncpg
from fastapi import Depends, Header, Request
from procrastinate import App as ProcrastinateApp

from aios.config import Settings, get_settings
from aios.crypto.vault import CryptoBox
from aios.errors import UnauthorizedError
from aios.services import runtime_tokens as runtime_tokens_service


def _extract_bearer_token(authorization: str | None) -> str:
    """Parse a ``Bearer <token>`` header.  Raises 401 on absent / malformed."""
    if authorization is None:
        raise UnauthorizedError("missing Authorization header")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise UnauthorizedError("expected `Authorization: Bearer <token>`")
    return token


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
    """Verify the request carries the operator API key.

    Constant-time compare so the comparison itself doesn't leak timing
    information about the key.
    """
    token = _extract_bearer_token(authorization)
    expected = settings.api_key.get_secret_value()
    if not secrets.compare_digest(token, expected):
        raise UnauthorizedError("invalid api key")


async def require_runtime_auth(
    pool: Annotated[asyncpg.Pool, Depends(get_pool)],
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> tuple[str, str]:
    """Resolve a bearer runtime token to ``(runtime_token_id, connector)``.

    Accepts only tokens issued via ``POST /v1/runtime-tokens`` (#328 PR 5).
    Routes that take ``RuntimeAuthDep`` receive the resolved tuple — the
    ``connector`` half is used to scope the runtime-facing routes
    (``/connectors/runtime/...`` family) to one connector type.
    """
    token = _extract_bearer_token(authorization)
    resolved = await runtime_tokens_service.resolve(pool, token)
    if resolved is None:
        raise UnauthorizedError("invalid or revoked runtime token")
    return (resolved.token_id, resolved.connector)


# Type aliases for clarity at the route definitions.
# Note: asyncpg.Pool isn't subscriptable at runtime, so we annotate with the
# bare class. FastAPI's dependency injection ignores generic parameters.
PoolDep = Annotated[asyncpg.Pool, Depends(get_pool)]
CryptoBoxDep = Annotated[CryptoBox, Depends(get_crypto_box)]
ProcrastinateDep = Annotated[ProcrastinateApp, Depends(get_procrastinate)]
DbUrlDep = Annotated[str, Depends(get_db_url)]
AuthDep = Annotated[None, Depends(require_bearer_auth)]
RuntimeAuthDep = Annotated[tuple[str, str], Depends(require_runtime_auth)]
