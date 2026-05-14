"""Runtime token endpoints — operator-scoped issue / list / revoke (#328 PR 5).

Bearer tokens scope per ``connector`` type rather than per
``connection_id``.  One runtime container hosts N connections of one
type and authenticates with one token.

Plaintext is returned ONCE on issue.  All subsequent reads expose only
the :class:`RuntimeToken` view.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from aios.api.deps import AuthDep, PoolDep
from aios.models.common import ListResponse
from aios.models.runtime_tokens import (
    RuntimeToken,
    RuntimeTokenIssue,
    RuntimeTokenIssued,
)
from aios.services import runtime_tokens as service

router = APIRouter(prefix="/v1/runtime-tokens", tags=["runtime-tokens"])


@router.post("", operation_id="issue_runtime_token", status_code=status.HTTP_201_CREATED)
async def issue(body: RuntimeTokenIssue, pool: PoolDep, _auth: AuthDep) -> RuntimeTokenIssued:
    """Mint a new bearer token scoped to ``body.connector``.

    The plaintext is included in the response and CANNOT be recovered
    later — operators must save it at issue time.
    """
    account_id, _, _ = _auth
    token, plaintext = await service.issue(
        pool, connector=body.connector, label=body.label, account_id=account_id
    )
    return RuntimeTokenIssued(
        id=token.id,
        connector=token.connector,
        label=token.label,
        plaintext=plaintext,
        created_at=token.created_at,
    )


@router.get("", operation_id="list_runtime_tokens")
async def list_(
    connector: str,
    pool: PoolDep,
    _auth: AuthDep,
) -> ListResponse[RuntimeToken]:
    """All tokens (revoked included) for ``connector``, newest first."""
    account_id, _, _ = _auth
    items = await service.list_tokens(pool, connector=connector, account_id=account_id)
    return ListResponse[RuntimeToken](data=items, has_more=False, next_after=None)


@router.post("/{token_id}/revoke", operation_id="revoke_runtime_token")
async def revoke(token_id: str, pool: PoolDep, _auth: AuthDep) -> RuntimeToken:
    """Soft-delete a token.  Idempotent — re-revoking is a no-op."""
    account_id, _, _ = _auth
    return await service.revoke(pool, token_id, account_id=account_id)
