"""Account management endpoints."""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import APIRouter, Header, status

from aios.api.deps import PoolDep, _extract_bearer_token
from aios.config import get_settings
from aios.db import queries
from aios.errors import NotFoundError, UnauthorizedError
from aios.logging import get_logger
from aios.models.accounts import BootstrapRequest, BootstrapResponse
from aios.services import accounts as service

router = APIRouter(prefix="/v1/accounts", tags=["accounts"])

log = get_logger("aios.api.accounts")


@router.post(
    "/bootstrap",
    operation_id="bootstrap_root_account",
    status_code=status.HTTP_201_CREATED,
    # Operator-side ceremony; not part of the agent-facing management plane.
    openapi_extra={"x-codegen": {"targets": ["sdk"]}},
)
async def bootstrap(
    body: BootstrapRequest,
    pool: PoolDep,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> BootstrapResponse:
    """Mint the root account and its first API key.

    Gated by ``AIOS_BOOTSTRAP_TOKEN`` (env var). When that env is unset
    or empty, the endpoint is 401 regardless of header value.

    Root-exists check fires before the token check so a probe with no/
    wrong token can't distinguish "no bootstrap" (404) from "wrong token
    but bootstrap is still open" (401). Once a root is in place the
    endpoint behaves like it doesn't exist at all.

    ``plaintext_key`` in the response is the only time the operator key
    is returned in plaintext.
    """
    async with pool.acquire() as conn:
        if await queries.has_active_root_account(conn):
            raise NotFoundError("bootstrap endpoint closed: root account already exists")
    token = _extract_bearer_token(authorization)
    expected_secret = get_settings().bootstrap_token
    expected = expected_secret.get_secret_value() if expected_secret is not None else ""
    if not expected or not secrets.compare_digest(token, expected):
        raise UnauthorizedError("invalid bootstrap token")
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
