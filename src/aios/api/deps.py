"""FastAPI dependencies: auth, DB pool, vault.

Each dependency is async and resolved per request via FastAPI's dep injection.
The pool and vault are stored on ``request.app.state`` at startup, then
exposed here as typed accessors.
"""

from __future__ import annotations

import secrets
from typing import Annotated, cast

import asyncpg
from fastapi import Depends, Header, HTTPException, Request, status

from aios.config import Settings, get_settings
from aios.crypto.vault import Vault
from aios.errors import UnauthorizedError


def get_pool(request: Request) -> asyncpg.Pool:
    return cast("asyncpg.Pool", request.app.state.pool)


def get_vault(request: Request) -> Vault:
    vault: Vault = request.app.state.vault
    return vault


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
VaultDep = Annotated[Vault, Depends(get_vault)]
AuthDep = Annotated[None, Depends(require_bearer_auth)]
