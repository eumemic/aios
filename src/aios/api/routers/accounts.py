"""Account management endpoints.

PR 1 of the multi-tenancy stack (issue #367) ships only the bootstrap
route. The full ``/v1/accounts/*`` surface (create child, list, mint
key, archive, by-path lookup, usage) lands in PR 6 once the auth dep
returns ``(account_id, key_id, can_mint_children)``.
"""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import APIRouter, Header, status

from aios.api.deps import PoolDep
from aios.config import Settings, get_settings
from aios.db import queries
from aios.errors import NotFoundError, UnauthorizedError
from aios.logging import get_logger
from aios.models.accounts import BootstrapRequest, BootstrapResponse
from aios.services import accounts as service

router = APIRouter(prefix="/v1/accounts", tags=["accounts"])

log = get_logger("aios.api.accounts")


def _extract_bootstrap_token(authorization: str | None) -> str:
    """Pull the bearer token from ``Authorization`` or 401."""
    if authorization is None:
        raise UnauthorizedError("missing Authorization header")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise UnauthorizedError("expected `Authorization: Bearer <token>`")
    return token


@router.post(
    "/bootstrap",
    operation_id="bootstrap_root_account",
    status_code=status.HTTP_201_CREATED,
    # The bootstrap endpoint is operator-side ceremony, not part of the
    # agent-facing management plane. Excluding it from the MCP surface
    # keeps the tool list focused on what a session can actually do.
    openapi_extra={"x-codegen": {"targets": ["sdk"]}},
)
async def bootstrap(
    body: BootstrapRequest,
    pool: PoolDep,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> BootstrapResponse:
    """One-shot endpoint that mints the root account and its first API key.

    Gated by ``AIOS_BOOTSTRAP_TOKEN`` (env var). When that env is unset
    or empty, the endpoint is 401 regardless of header value — a fresh
    deployment must explicitly opt in to bootstrap by setting the env.

    Once a non-archived root account exists, the endpoint is 404
    regardless of token validity. The ``accounts_one_active_root``
    partial unique index in migration 0040 enforces the invariant at
    the DB layer too — the 404 here is the friendly upstream answer.

    The ``plaintext_key`` field of the response is the *only* time the
    operator key is returned in plaintext. After this call, every
    subsequent use of that key authenticates against the stored
    ``sha256`` hash.
    """
    # Root-exists check fires first so a probe with no/wrong token can't
    # distinguish "no bootstrap" (404) from "wrong token but bootstrap is
    # still open" (401). Once a root is in place the endpoint behaves
    # like it doesn't exist at all, regardless of what the caller sent.
    async with pool.acquire() as conn:
        if await queries.has_active_root_account(conn):
            raise NotFoundError("bootstrap endpoint closed: root account already exists")
    settings = get_settings()
    _require_bootstrap_token(settings, authorization)
    response = await service.bootstrap_root(pool, display_name=body.display_name)
    log.info(
        "account.operation",
        actor_account_id=None,
        actor_key_id=None,
        target_account_id=response.account_id,
        action="account.bootstrap",
        outcome="success",
    )
    return response


def _require_bootstrap_token(settings: Settings, authorization: str | None) -> None:
    """Authenticate a bootstrap request against ``AIOS_BOOTSTRAP_TOKEN``.

    Separated so the unset-env case (401) is indistinguishable from
    the wrong-token case (401) — the response body is identical and
    no length-leak via early-return-on-missing-env is possible.
    """
    token = _extract_bootstrap_token(authorization)
    expected_secret = settings.bootstrap_token
    expected = expected_secret.get_secret_value() if expected_secret is not None else ""
    if not expected or not secrets.compare_digest(token, expected):
        raise UnauthorizedError("invalid bootstrap token")
