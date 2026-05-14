"""Account and account-key resource models.

PR 1 of the multi-tenancy stack: only the minimum needed for the
bootstrap endpoint. Read models for accounts and the bootstrap
request/response live here; the full CRUD surface (create child,
list children, mint key, archive, etc.) lands in PR 6 once the auth
dep is account-aware.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Account(BaseModel):
    """Read view of an account.

    ``parent_account_id`` is ``None`` only for the singular root account.
    ``can_mint_children`` distinguishes accounts that can provision
    sub-accounts from leaf accounts that cannot.
    """

    id: str
    parent_account_id: str | None
    can_mint_children: bool
    display_name: str
    metadata: dict[str, Any]
    created_at: datetime
    archived_at: datetime | None = None


class BootstrapRequest(BaseModel):
    """Body for ``POST /v1/accounts/bootstrap``.

    Only the human-readable display_name is required at bootstrap time;
    metadata can be added later via a PATCH endpoint (lands in PR 6).
    """

    model_config = ConfigDict(extra="forbid")

    display_name: str = Field(min_length=1, max_length=128)


class BootstrapResponse(BaseModel):
    """Response from ``POST /v1/accounts/bootstrap``.

    ``plaintext_key`` is returned exactly once — it's never recoverable
    after this response. The operator must capture it (env, secret store,
    password manager). All subsequent API calls use it as a bearer token.
    """

    account_id: str
    key_id: str
    plaintext_key: str
