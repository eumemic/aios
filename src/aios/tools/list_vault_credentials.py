"""Session-scoped, metadata-only vault credential self-visibility (#1945)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic import ValidationError as PydanticValidationError

from aios.db import queries
from aios.harness import runtime
from aios.services import sessions as sessions_service
from aios.tools.invoke import ToolBail
from aios.tools.registry import ToolResult, registry


class _ListVaultCredentialsArgs(BaseModel):
    """No caller-supplied identity: the harness supplies the executing session."""

    model_config = ConfigDict(extra="forbid")


async def list_vault_credentials_handler(
    session_id: str, arguments: dict[str, Any]
) -> dict[str, Any] | ToolResult:
    """Return non-secret metadata for credentials in this session's vaults."""
    try:
        _ListVaultCredentialsArgs.model_validate(arguments)
    except PydanticValidationError as exc:
        raise ToolBail(f"invalid arguments: {exc}") from exc

    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    async with pool.acquire() as conn:
        rows = await queries.list_session_vault_credentials(  # pooled-connection-await: allow eumemic/aios#1945
            conn, session_id, account_id=account_id
        )
    return {
        "credentials": [
            {
                "credential_id": row.credential_id,
                "vault_id": row.vault_id,
                "display_name": row.display_name,
                "auth_type": row.auth_type,
                "secret_name": row.secret_name,
                "allowed_hosts": list(row.allowed_hosts),
                "target_url": row.target_url,
                "created_at": row.created_at.isoformat().replace("+00:00", "Z"),
                "archived_at": (
                    row.archived_at.isoformat().replace("+00:00", "Z")
                    if row.archived_at is not None
                    else None
                ),
            }
            for row in rows
        ]
    }


_DESCRIPTION = (
    "List metadata for every active or archived credential in vaults attached to "
    "your current session. Returns credential_id, vault_id, display_name, auth_type, "
    "secret_name, allowed_hosts, target_url, created_at, and archived_at. This is "
    "read-only and never returns secret material, ciphertext, nonce, or credential "
    "metadata payloads. Use it to diagnose credential identity, host scope, and stale "
    "archived references."
)


def _register() -> None:
    registry.register(
        name="list_vault_credentials",
        description=_DESCRIPTION,
        parameters_schema=_ListVaultCredentialsArgs.model_json_schema(),
        handler=list_vault_credentials_handler,
        transport="agent_tool",
    )


_register()
