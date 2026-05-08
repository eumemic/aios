"""Connector token endpoints — operator-scoped issue / list / revoke.

These endpoints are guarded by the ordinary ``AuthDep`` (global
``AIOS_API_KEY``) — they're for operators provisioning connector
containers, not for the connectors themselves.  Connector-facing
endpoints take ``ConnectorAuthDep`` and resolve the token to one
``connection_id``.

Plaintext is returned ONCE on issue (in ``ConnectorTokenIssued``); all
subsequent reads expose only the ``ConnectorToken`` view.
"""

from __future__ import annotations

from fastapi import APIRouter, status
from pydantic import BaseModel

from aios.api.deps import AuthDep, ConnectorAuthDep, PoolDep
from aios.models.common import ListResponse
from aios.models.connector_tokens import (
    ConnectorToken,
    ConnectorTokenIssue,
    ConnectorTokenIssued,
)
from aios.services import connector_tokens as service


class WhoAmI(BaseModel):
    """Response for ``GET /v1/connector-tokens/whoami``."""

    connection_id: str


router = APIRouter(prefix="/v1/connector-tokens", tags=["connector-tokens"])


@router.post("", operation_id="issue_connector_token", status_code=status.HTTP_201_CREATED)
async def issue(body: ConnectorTokenIssue, pool: PoolDep, _auth: AuthDep) -> ConnectorTokenIssued:
    """Mint a new bearer token for ``body.connection_id``.

    The plaintext is included in the response and CANNOT be recovered
    later — operators must save it at issue time.  Subsequent ``GET``
    on this resource returns the read view without plaintext.
    """
    token, plaintext = await service.issue(pool, connection_id=body.connection_id, label=body.label)
    return ConnectorTokenIssued(
        id=token.id,
        connection_id=token.connection_id,
        label=token.label,
        plaintext=plaintext,
        created_at=token.created_at,
    )


@router.get("", operation_id="list_connector_tokens")
async def list_(
    connection_id: str,
    pool: PoolDep,
    _auth: AuthDep,
) -> ListResponse[ConnectorToken]:
    """All tokens (revoked included) for ``connection_id``, newest first.

    Revoked tokens stay in the listing for audit; clients filter by
    ``revoked_at IS NULL`` if they only want live tokens.
    """
    items = await service.list_for_connection(pool, connection_id)
    return ListResponse[ConnectorToken](data=items, has_more=False, next_after=None)


@router.post("/{token_id}/revoke", operation_id="revoke_connector_token")
async def revoke(token_id: str, pool: PoolDep, _auth: AuthDep) -> ConnectorToken:
    """Soft-delete a token.  Idempotent — re-revoking is a no-op."""
    return await service.revoke(pool, token_id)


@router.get("/whoami", operation_id="connector_token_whoami")
async def whoami(connection_id: ConnectorAuthDep) -> WhoAmI:
    """Resolve the bearer token to its ``connection_id``.

    Sanity check / debugging endpoint for connector containers — call
    once at startup to confirm the token is valid and points where the
    operator intended.  Authed by ``ConnectorAuthDep`` (token, NOT
    operator key), so any side-channel access from the operator surface
    is impossible.
    """
    return WhoAmI(connection_id=connection_id)
